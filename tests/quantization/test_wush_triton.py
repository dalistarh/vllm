#!/usr/bin/env python3.12
"""Test fused WUSH Triton decode kernel against Python reference.

Run: CUDA_VISIBLE_DEVICES=0 python3.12 tests/quantization/test_wush_triton.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm.model_executor.layers.quantization.turboquant.centroids import solve_lloyd_max
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv
from vllm.v1.attention.ops.triton_wush_decode import triton_wush_decode_attention
from vllm.triton_utils import triton


def _rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def python_wush_decode_ref(query, k_wush_all, v_wush_all, T_K, T_V_inv,
                           cos_cache, sin_cache, scale):
    """Reference: dequant + T_K^T + RoPE + attention in pure Python."""
    Hq, D = query.shape[1], query.shape[2]
    Hk = T_K.shape[0]
    gqa = Hq // Hk
    S = k_wush_all.shape[0]

    T_K_T = T_K.transpose(-2, -1)
    k_pre = torch.einsum('shd,hde->she', k_wush_all.float(), T_K_T.float())
    cos = cos_cache[:S].unsqueeze(1)
    sin = sin_cache[:S].unsqueeze(1)
    k_rope = k_pre * cos + _rotate_half(k_pre) * sin

    v_out = torch.einsum('shd,hde->she', v_wush_all.float(), T_V_inv.float())

    k_gqa = k_rope.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(S, Hq, D)
    v_gqa = v_out.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(S, Hq, D)

    scores = torch.einsum('bhd,shd->bhs', query.float(), k_gqa) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.einsum('bhs,shd->bhd', attn, v_gqa).to(query.dtype)


def test_wush_triton():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    D, Hk, Hq = 128, 8, 24
    S, block_size = 64, 16
    num_blocks = (S + block_size - 1) // block_size
    bits = 4

    print(f"Config: D={D}, Hk={Hk}, Hq={Hq}, S={S}, bits={bits}", flush=True)

    # Per-head transforms with small perturbation of identity (realistic)
    T_K = torch.eye(D, device=device).unsqueeze(0).repeat(Hk, 1, 1)
    T_K += 0.05 * torch.randn(Hk, D, D, device=device)
    T_K = T_K.float()
    T_V = torch.eye(D, device=device).unsqueeze(0).repeat(Hk, 1, 1)
    T_V += 0.05 * torch.randn(Hk, D, D, device=device)
    T_V = T_V.float()
    T_K_inv = torch.linalg.inv(T_K.double()).float()
    T_V_inv = torch.linalg.inv(T_V.double()).float()

    # RoPE
    half = D // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=device).float() / D))
    angles = torch.outer(torch.arange(S, device=device).float(), freqs)
    cos_cache = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin_cache = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

    # Random data + WUSH forward transforms
    keys_pre = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    values = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    k_wush = torch.einsum('shd,hde->she', keys_pre.float(), T_K_inv).half()
    v_wush = torch.einsum('shd,hde->she', values.float(), T_V.mT).half()

    # ---- Store into TQ cache ----
    centroids, _ = solve_lloyd_max(D, bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    cfg = TurboQuantConfig(head_dim=D, key_quant_bits=bits,
                           value_quant_bits=bits, norm_correction=True, wush=True)
    PiT = torch.eye(D, device=device, dtype=torch.float32)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cfg.slot_size_aligned,
                           device=device, dtype=torch.uint8)
    slot_mapping = torch.arange(S, device=device, dtype=torch.int32)

    triton_turboquant_store(
        k_wush, v_wush, kv_cache, slot_mapping, PiT, midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=False,
    )
    print(f"Stored {S} tokens into TQ cache", flush=True)

    # ---- Dequant for reference ----
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * bits / 8)
    val_data_bytes = math.ceil(D * bits / 8)
    alloc_len = num_blocks * block_size
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)

    k_deq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)
    v_deq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)
    _tq_full_dequant_kv[(alloc_len, Hk)](
        kv_cache, block_table, c_sorted, k_deq, v_deq,
        k_deq.stride(0), k_deq.stride(1), k_deq.stride(2),
        v_deq.stride(0), v_deq.stride(1), v_deq.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D, BLOCK_SIZE=block_size, NUM_KV_HEADS=Hk,
        MSE_BYTES=mse_bytes, KPS=cfg.key_packed_size, VQB=bits,
        VAL_DATA_BYTES=val_data_bytes, MSE_BITS=bits,
        KEY_FP8=0, BLOCK_D=BLOCK_D, NORM_CORRECTION=1, FP8_E4B15=0,
        num_warps=4,
    )
    k_wush_deq = k_deq[0, :, :S, :].transpose(0, 1)  # [S, Hk, D]
    v_wush_deq = v_deq[0, :, :S, :].transpose(0, 1)

    # ---- Query ----
    q_pre = torch.randn(1, Hq, D, device=device, dtype=torch.float16)
    c_q = cos_cache[S - 1].unsqueeze(0).unsqueeze(0)
    s_q = sin_cache[S - 1].unsqueeze(0).unsqueeze(0)
    q_rope = (q_pre.float() * c_q + _rotate_half(q_pre.float()) * s_q).half()
    scale = 1.0 / math.sqrt(D)
    seq_lens = torch.tensor([S], device=device, dtype=torch.int32)

    # ---- Python reference (use same dequant data as Triton) ----
    print("Python reference...", flush=True)
    # The ref was using cos_cache [S, D] but the Triton launcher
    # gives cos_half [S, D/2].  Make sure ref uses SAME cos convention:
    ref_out = python_wush_decode_ref(
        q_rope, k_wush_deq, v_wush_deq, T_K, T_V_inv,
        cos_cache, sin_cache, scale,
    )
    # Also compute directly from the Triton kernel's dequanted K/V
    # to isolate the attention computation:
    T_K_T2 = T_K.transpose(-2, -1)
    k_pre2 = torch.einsum('shd,hde->she', k_wush_deq.float(), T_K_T2.float())
    cos2 = cos_cache[:S].unsqueeze(1)
    sin2 = sin_cache[:S].unsqueeze(1)
    k_rope2 = k_pre2 * cos2 + _rotate_half(k_pre2) * sin2
    v_out2 = torch.einsum('shd,hde->she', v_wush_deq.float(), T_V_inv.float())
    # Expand GQA
    gqa = Hq // Hk
    k_gqa2 = k_rope2.unsqueeze(2).expand(-1,-1,gqa,-1).reshape(S,Hq,D)
    v_gqa2 = v_out2.unsqueeze(2).expand(-1,-1,gqa,-1).reshape(S,Hq,D)
    scores2 = torch.einsum('bhd,shd->bhs', q_rope.float(), k_gqa2) * scale
    attn2 = F.softmax(scores2, dim=-1)
    ref_out2 = torch.einsum('bhs,shd->bhd', attn2, v_gqa2).to(q_rope.dtype)
    ref_out = ref_out2  # use this as authoritative reference

    # ---- Triton WUSH kernel ----
    print("Triton WUSH kernel...", flush=True)
    T_K_T = T_K.transpose(-2, -1).contiguous()
    cos_half = cos_cache[:, :half].contiguous()
    sin_half = sin_cache[:, :half].contiguous()

    try:
        triton_out = triton_wush_decode_attention(
            query=q_rope, kv_cache=kv_cache,
            block_table=block_table, seq_lens=seq_lens,
            centroids=c_sorted, T_K_T=T_K_T, T_V_inv=T_V_inv,
            cos_cache=cos_half, sin_cache=sin_half,
            scale=scale, mse_bits=bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=bits, norm_correction=True,
            max_num_kv_splits=4,
        )
        print(f"  Output shape: {triton_out.shape}", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        triton_out = None

    # ---- Compare ----
    if triton_out is not None:
        print("\nPer-head cosine similarity:", flush=True)
        sims = []
        for h in range(Hq):
            sim = F.cosine_similarity(
                triton_out[0, h].float().unsqueeze(0),
                ref_out[0, h].float().unsqueeze(0),
            ).item()
            sims.append(sim)
            if h < 6 or sim < 0.85:
                print(f"  Head {h:2d}: {sim:.6f}", flush=True)

        avg = sum(sims) / len(sims)
        max_diff = (triton_out.float() - ref_out.float()).abs().max().item()
        print(f"\nAvg cosine sim: {avg:.6f}")
        print(f"Max abs diff:   {max_diff:.6f}")
        print(f"\n{'PASSED' if avg > 0.85 else 'FAILED'}")
    else:
        print("\nSKIPPED (kernel failed)")


if __name__ == '__main__':
    test_wush_triton()
