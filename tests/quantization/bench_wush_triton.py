#!/usr/bin/env python3.12
"""Benchmark fused WUSH Triton decode kernel vs baselines.

Compares three decode implementations:
  1. TQ Hadamard  — standard TurboQuant (Hadamard rotation, fused Triton)
  2. WUSH Python   — Python dequant + T_K matmul + RoPE + FlashAttention
  3. WUSH Triton   — fused Triton kernel (this PR)

Sweeps over sequence lengths to show scaling behavior.

Run: CUDA_VISIBLE_DEVICES=0 python3.12 tests/quantization/bench_wush_triton.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm.model_executor.layers.quantization.turboquant.centroids import solve_lloyd_max
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_wush_decode import triton_wush_decode_attention
from vllm.triton_utils import triton


def _rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def build_hadamard(d, device):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device)


def setup_cache(S, D, Hk, bits, block_size, device):
    """Store S tokens into TQ cache with WUSH transforms."""
    cfg = TurboQuantConfig(head_dim=D, key_quant_bits=bits,
                           value_quant_bits=bits, norm_correction=True, wush=True)
    num_blocks = (S + block_size - 1) // block_size

    # Transforms
    T_K = torch.randn(Hk, D, D, device=device).double()
    T_K = (T_K @ T_K.mT + 5 * torch.eye(D, device=device)).float()
    T_V = torch.randn(Hk, D, D, device=device).double()
    T_V = (T_V @ T_V.mT + 5 * torch.eye(D, device=device)).float()
    T_K_inv = torch.linalg.inv(T_K.double()).float()
    T_V_inv = torch.linalg.inv(T_V.double()).float()

    # RoPE
    half = D // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=device).float() / D))
    angles = torch.outer(torch.arange(S, device=device).float(), freqs)
    cos_cache = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin_cache = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

    # Random data + WUSH forward
    keys_pre = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    values = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    k_wush = torch.einsum('shd,hde->she', keys_pre.float(), T_K_inv).half()
    v_wush = torch.einsum('shd,hde->she', values.float(), T_V.mT).half()

    # Centroids
    centroids, _ = solve_lloyd_max(D, bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    # Store
    PiT = torch.eye(D, device=device, dtype=torch.float32)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cfg.slot_size_aligned,
                           device=device, dtype=torch.uint8)
    slot_mapping = torch.arange(S, device=device, dtype=torch.int32)
    triton_turboquant_store(
        k_wush, v_wush, kv_cache, slot_mapping, PiT, midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=False,
    )

    # Also store with Hadamard rotation for TQ baseline
    H = build_hadamard(D, device)
    kv_cache_tq = torch.zeros_like(kv_cache)
    k_tq = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    v_tq = torch.randn(S, Hk, D, device=device, dtype=torch.float16)
    triton_turboquant_store(
        k_tq, v_tq, kv_cache_tq, slot_mapping, H, midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=False,
    )

    return {
        'cfg': cfg, 'kv_cache': kv_cache, 'kv_cache_tq': kv_cache_tq,
        'c_sorted': c_sorted, 'midpoints': midpoints,
        'T_K': T_K, 'T_K_inv': T_K_inv, 'T_V': T_V, 'T_V_inv': T_V_inv,
        'cos_cache': cos_cache, 'sin_cache': sin_cache,
        'H': H, 'num_blocks': num_blocks,
    }


def bench_tq_hadamard(query, ctx, S, B, Hq, D, block_size, device, warmup=5, iters=50):
    """Standard TurboQuant decode (Hadamard, fused Triton)."""
    cfg = ctx['cfg']
    num_blocks = ctx['num_blocks']
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1)
    seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)

    for _ in range(warmup):
        triton_turboquant_decode_attention(
            query=query, kv_cache=ctx['kv_cache_tq'],
            block_table=block_table, seq_lens=seq_lens,
            Pi=ctx['H'], centroids=ctx['c_sorted'], scale=1.0/math.sqrt(D),
            mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=False, norm_correction=True, PiT=ctx['H'],
            max_num_kv_splits=32,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        triton_turboquant_decode_attention(
            query=query, kv_cache=ctx['kv_cache_tq'],
            block_table=block_table, seq_lens=seq_lens,
            Pi=ctx['H'], centroids=ctx['c_sorted'], scale=1.0/math.sqrt(D),
            mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=False, norm_correction=True, PiT=ctx['H'],
            max_num_kv_splits=32,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms


def bench_wush_triton(query, ctx, S, B, Hq, D, block_size, device, warmup=5, iters=50):
    """Fused WUSH Triton decode."""
    cfg = ctx['cfg']
    num_blocks = ctx['num_blocks']
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1)
    seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)
    half = D // 2
    T_K_T = ctx['T_K'].transpose(-2, -1).contiguous()
    cos_half = ctx['cos_cache'][:, :half].contiguous()
    sin_half = ctx['sin_cache'][:, :half].contiguous()

    for _ in range(warmup):
        triton_wush_decode_attention(
            query=query, kv_cache=ctx['kv_cache'],
            block_table=block_table, seq_lens=seq_lens,
            centroids=ctx['c_sorted'], T_K_T=T_K_T, T_V_inv=ctx['T_V_inv'],
            cos_cache=cos_half, sin_cache=sin_half,
            scale=1.0/math.sqrt(D), mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            norm_correction=True, max_num_kv_splits=32,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        triton_wush_decode_attention(
            query=query, kv_cache=ctx['kv_cache'],
            block_table=block_table, seq_lens=seq_lens,
            centroids=ctx['c_sorted'], T_K_T=T_K_T, T_V_inv=ctx['T_V_inv'],
            cos_cache=cos_half, sin_cache=sin_half,
            scale=1.0/math.sqrt(D), mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            norm_correction=True, max_num_kv_splits=32,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_wush_python(query, ctx, S, B, Hq, Hk, D, block_size, device, warmup=3, iters=20):
    """Python dequant + T_K matmul + RoPE + SDPA."""
    cfg = ctx['cfg']
    num_blocks = ctx['num_blocks']
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)

    def run_once():
        # Dequant
        alloc_len = num_blocks * block_size
        k_deq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)
        v_deq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)
        _tq_full_dequant_kv[(alloc_len, Hk)](
            ctx['kv_cache'], block_table, ctx['c_sorted'], k_deq, v_deq,
            k_deq.stride(0), k_deq.stride(1), k_deq.stride(2),
            v_deq.stride(0), v_deq.stride(1), v_deq.stride(2),
            ctx['kv_cache'].stride(0), ctx['kv_cache'].stride(1), ctx['kv_cache'].stride(2),
            block_table.stride(0),
            HEAD_DIM=D, BLOCK_SIZE=block_size, NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes, KPS=cfg.key_packed_size, VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes, MSE_BITS=cfg.key_mse_bits,
            KEY_FP8=0, BLOCK_D=BLOCK_D, NORM_CORRECTION=1, FP8_E4B15=0, num_warps=4,
        )
        k_w = k_deq[0, :, :S, :].transpose(0, 1)  # [S, Hk, D]
        v_w = v_deq[0, :, :S, :].transpose(0, 1)

        # Inverse WUSH K
        T_K_T = ctx['T_K'].transpose(-2, -1)
        k_pre = torch.einsum('shd,hde->she', k_w.float(), T_K_T.float())
        # RoPE
        cos = ctx['cos_cache'][:S].unsqueeze(1)
        sin = ctx['sin_cache'][:S].unsqueeze(1)
        k_rope = k_pre * cos + _rotate_half(k_pre) * sin
        # Inverse WUSH V
        v_out = torch.einsum('shd,hde->she', v_w.float(), ctx['T_V_inv'].float())

        # GQA expand + attention
        gqa = Hq // Hk
        k_gqa = k_rope.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(S, Hq, D)
        v_gqa = v_out.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(S, Hq, D)
        scores = torch.einsum('bhd,shd->bhs', query[:1].float(), k_gqa) * (1.0/math.sqrt(D))
        attn = F.softmax(scores, dim=-1)
        return torch.einsum('bhs,shd->bhd', attn, v_gqa)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    D, Hk, Hq = 128, 8, 24
    B = 1
    bits = 4
    block_size = 16
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"Config: D={D}, Hk={Hk}, Hq={Hq}, B={B}, bits={bits}")
    print(f"{'S':>6s}  {'TQ Hadamard':>12s}  {'WUSH Triton':>12s}  {'WUSH Python':>12s}  {'Triton/TQ':>10s}")
    print("-" * 70)

    for S in seq_lengths:
        # Fresh setup per sequence length (RoPE tables sized to S)
        ctx = setup_cache(S, D, Hk, bits, block_size, device)
        q_pre = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        c_q = ctx['cos_cache'][S-1].unsqueeze(0).unsqueeze(0)
        s_q = ctx['sin_cache'][S-1].unsqueeze(0).unsqueeze(0)
        query = (q_pre.float() * c_q + _rotate_half(q_pre.float()) * s_q).half()

        t_tq = bench_tq_hadamard(query, ctx, S, B, Hq, D, block_size, device)
        t_wt = bench_wush_triton(query, ctx, S, B, Hq, D, block_size, device)

        # Python path is slow; use fewer iters for long seqs
        py_iters = 20 if S <= 512 else (10 if S <= 2048 else 5)
        t_wp = bench_wush_python(query, ctx, S, B, Hq, Hk, D, block_size, device,
                                 warmup=2, iters=py_iters)

        ratio = t_wt / t_tq if t_tq > 0 else float('inf')
        print(f"{S:6d}  {t_tq:10.3f}ms  {t_wt:10.3f}ms  {t_wp:10.3f}ms  {ratio:9.2f}x")

    print()
    print("TQ Hadamard  = standard TurboQuant (fused Triton, orthogonal rotation)")
    print("WUSH Triton  = fused WUSH kernel (this PR)")
    print("WUSH Python  = Python dequant + matmul + RoPE + SDPA (old path)")
    print("Triton/TQ    = WUSH Triton slowdown vs TQ Hadamard")


if __name__ == '__main__':
    main()
