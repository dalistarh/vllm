# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused WUSH-KV decode: two-kernel pipeline.

Kernel 1 (_wush_dequant_transform_kv): parallel across (pos, head)
  K-side: dequant centroids*norm → T_K^T matmul → RoPE → fp16 output
  V-side: dequant → fp16 output (stays in WUSH space)

Kernel 2: standard flash attention / SDPA on dequanted K/V

Post-kernel: T_V^{-1} applied once to attention output (one matmul
per head, not per token — saves O(S*D^2) vs per-token transform).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


# ---- Kernel 1: Fused dequant + T_K^T + RoPE (parallel across tokens) ----

@triton.jit
def _wush_dequant_transform_kv(
    KV_cache_ptr,       # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,    # [B, max_num_blocks] int32
    Centroids_ptr,      # [n_centroids] float32
    T_K_T_ptr,          # [Hk, D, D] float32 — T_K transposed
    Cos_ptr,            # [max_pos, D/2] float32
    Sin_ptr,            # [max_pos, D/2] float32
    K_out_ptr,          # [B, Hk, alloc_len, D] float16 — post-RoPE keys
    V_out_ptr,          # [B, Hk, alloc_len, D] float16 — WUSH-space values
    Scratch_ptr,        # [num_blocks_total, Hk, D] float32 — for rotate_half
    # Strides
    stride_ko_b, stride_ko_h, stride_ko_s,
    stride_vo_b, stride_vo_h, stride_vo_s,
    stride_cb, stride_cp, stride_ch,
    stride_bt,
    stride_th, stride_tr, stride_tc,
    stride_cos_pos,
    stride_scratch_p, stride_scratch_h,
    # Constexpr
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    TILE_K: tl.constexpr = 32,
):
    """Dequant one (pos, head) slot: K gets T_K^T + RoPE, V stays raw."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    # Page lookup
    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt + page_idx).to(tl.int64)
    slot_base = (block_num * stride_cb
                 + tl.cast(page_off, tl.int64) * stride_cp
                 + tl.cast(hid, tl.int64) * stride_ch)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # ======== K: dequant ========
    mse_bit_off = d_offs * MSE_BITS
    mse_byte_idx = mse_bit_off // 8
    mse_bit_shift = mse_bit_off % 8
    mse_umask = (1 << MSE_BITS) - 1

    mr0 = tl.load(KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0).to(tl.int32)
    mr1 = tl.load(KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0).to(tl.int32)
    mse_idx = ((mr0 | (mr1 << 8)) >> mse_bit_shift) & mse_umask
    k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

    if NORM_CORRECTION:
        cn2 = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0))
        k_mse = k_mse * (1.0 / tl.sqrt(cn2 + 1e-16))

    nb = slot_base + MSE_BYTES
    nlo = tl.load(KV_cache_ptr + nb).to(tl.uint16)
    nhi = tl.load(KV_cache_ptr + nb + 1).to(tl.uint16)
    vnorm = (nlo | (nhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_wush = k_mse * vnorm  # [BLOCK_D]

    # ======== K: T_K^T matmul (tiled) ========
    # k_pre[d] = sum_{d_in} k_wush[d_in] * T_K_T[hid, d_in, d]
    # Write k_wush to scratch for tiled reads
    scratch_base = pos * stride_scratch_p + hid * stride_scratch_h
    tl.store(Scratch_ptr + scratch_base + d_offs, k_wush, mask=d_mask)

    tk_base = hid * stride_th
    k_pre = tl.zeros([BLOCK_D], dtype=tl.float32)
    for tile in range(0, HEAD_DIM, TILE_K):
        t_offs = tile + tl.arange(0, TILE_K)
        t_mask = t_offs < HEAD_DIM

        # T_K_T[hid, tile:tile+TILE_K, :] — [TILE_K, BLOCK_D]
        tk_addrs = tk_base + t_offs[:, None] * stride_tr + d_offs[None, :] * stride_tc
        tk_tile = tl.load(T_K_T_ptr + tk_addrs,
                          mask=t_mask[:, None] & d_mask[None, :], other=0.0)

        # k_wush[tile:tile+TILE_K] from scratch
        k_slice = tl.load(Scratch_ptr + scratch_base + t_offs,
                          mask=t_mask, other=0.0)

        k_pre += tl.sum(k_slice[:, None] * tk_tile, axis=0)

    # ======== K: RoPE ========
    # Write k_pre to scratch for rotate_half gather
    tl.store(Scratch_ptr + scratch_base + d_offs, k_pre, mask=d_mask)

    cos_idx = d_offs % HALF_DIM
    cos_val = tl.load(Cos_ptr + pos * stride_cos_pos + cos_idx, mask=d_mask, other=1.0)
    sin_val = tl.load(Sin_ptr + pos * stride_cos_pos + cos_idx, mask=d_mask, other=0.0)

    rot_idx = tl.where(d_offs < HALF_DIM, d_offs + HALF_DIM, d_offs - HALF_DIM)
    rot_sign = tl.where(d_offs < HALF_DIM, -1.0, 1.0)
    k_rot = rot_sign * tl.load(Scratch_ptr + scratch_base + rot_idx, mask=d_mask, other=0.0)

    k_rope = k_pre * cos_val + k_rot * sin_val

    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_rope.to(tl.float16), mask=d_mask)

    # ======== V: dequant only (stays in WUSH space) ========
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        vr = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(tl.int32)
        vi = ((vr >> vb_shift) & 0xF).to(tl.float32)
    elif VQB == 3:
        vbo = d_offs * 3
        vbi = vbo // 8
        vbs = vbo % 8
        vr0 = tl.load(KV_cache_ptr + val_base + vbi, mask=d_mask, other=0).to(tl.int32)
        vr1 = tl.load(KV_cache_ptr + val_base + vbi + 1, mask=d_mask, other=0).to(tl.int32)
        vi = (((vr0 | (vr1 << 8)) >> vbs) & 0x7).to(tl.float32)
    else:
        vi = tl.zeros([BLOCK_D], dtype=tl.float32)

    sb = val_base + VAL_DATA_BYTES
    slo = tl.load(KV_cache_ptr + sb).to(tl.uint16)
    shi = tl.load(KV_cache_ptr + sb + 1).to(tl.uint16)
    vs = (slo | (shi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    zlo = tl.load(KV_cache_ptr + sb + 2).to(tl.uint16)
    zhi = tl.load(KV_cache_ptr + sb + 3).to(tl.uint16)
    vz = (zlo | (zhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    v_vals = vi * vs + vz

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---- Launcher: two-kernel pipeline ----

def triton_wush_decode_attention(
    query: torch.Tensor,        # [B, Hq, D] post-RoPE
    kv_cache: torch.Tensor,     # [num_blocks, block_size, Hk, padded_slot]
    block_table: torch.Tensor,  # [B, max_num_blocks]
    seq_lens: torch.Tensor,     # [B]
    centroids: torch.Tensor,    # [n_centroids]
    T_K_T: torch.Tensor,        # [Hk, D, D] — T_K transposed
    T_V_inv: torch.Tensor,      # [Hk, D, D] — T_V^{-1}
    cos_cache: torch.Tensor,    # [max_pos, D/2]
    sin_cache: torch.Tensor,    # [max_pos, D/2]
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    norm_correction: bool = False,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Two-kernel WUSH decode attention.

    Kernel 1: Parallel dequant + T_K^T + RoPE (one thread block per token)
    Kernel 2: Flash attention / SDPA on dequanted K/V
    Post: T_V^{-1} applied once to attention output
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    gqa = Hq // Hk
    device = query.device
    half_dim = D // 2

    mse_bytes = math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    BLOCK_D = triton.next_power_of_2(D)

    # Max sequence length across batch (avoid GPU sync — use tensor op)
    max_seq = int(seq_lens.max().item())
    alloc_len = math.ceil(max_seq / block_size) * block_size

    # Reuse pre-allocated buffers via buf_holder to avoid per-call allocation
    _alloc = alloc_len * Hk * D
    k_deq = getattr(buf_holder, '_wush_k_deq', None) if buf_holder else None
    if k_deq is None or k_deq.numel() < B * _alloc:
        k_deq = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)
        v_deq = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)
        scratch = torch.empty(alloc_len, Hk, D, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._wush_k_deq = k_deq
            buf_holder._wush_v_deq = v_deq
            buf_holder._wush_scratch = scratch
    else:
        v_deq = buf_holder._wush_v_deq
        scratch = buf_holder._wush_scratch
        k_deq = k_deq[:B, :, :alloc_len, :]
        v_deq = v_deq[:B, :, :alloc_len, :]
        scratch = scratch[:alloc_len, :, :]

    T_K_T = T_K_T.contiguous().float()
    cos_cache = cos_cache.contiguous().float()
    sin_cache = sin_cache.contiguous().float()

    # ---- Kernel 1: fused dequant + T_K^T + RoPE ----
    grid = (alloc_len, B * Hk)
    _wush_dequant_transform_kv[grid](
        kv_cache, block_table, centroids,
        T_K_T, cos_cache, sin_cache,
        k_deq, v_deq, scratch,
        # Strides
        k_deq.stride(0), k_deq.stride(1), k_deq.stride(2),
        v_deq.stride(0), v_deq.stride(1), v_deq.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        T_K_T.stride(0), T_K_T.stride(1), T_K_T.stride(2),
        cos_cache.stride(0),
        scratch.stride(0), scratch.stride(1),
        # Constexpr
        HEAD_DIM=D, HALF_DIM=half_dim,
        BLOCK_SIZE=block_size, NUM_KV_HEADS=Hk,
        MSE_BITS=mse_bits, MSE_BYTES=mse_bytes,
        KPS=key_packed_size, VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes, BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if norm_correction else 0,
        num_warps=4,
    )

    # ---- Kernel 2: batched attention via SDPA ----
    # For decode (B queries, each attending to S cached tokens), batch across B.
    # k_deq [B, Hk, alloc_len, D], v_deq [B, Hk, alloc_len, D]
    # GQA expand Hk → Hq
    k_exp = k_deq[:, :, :max_seq, :].repeat_interleave(gqa, dim=1)  # [B, Hq, max_seq, D]
    v_exp = v_deq[:, :, :max_seq, :].repeat_interleave(gqa, dim=1)  # [B, Hq, max_seq, D]

    output = F.scaled_dot_product_attention(
        query.float().unsqueeze(2),  # [B, Hq, 1, D]
        k_exp.float(),               # [B, Hq, max_seq, D]
        v_exp.float(),               # [B, Hq, max_seq, D]
        scale=scale,
    )[:, :, 0, :]  # [B, Hq, D]

    # ---- Post: apply T_V^{-1} to output ----
    # output is in WUSH V-space; apply per-head T_V^{-1}
    T_V_inv_q = T_V_inv.float().repeat_interleave(gqa, dim=0)  # [Hq, D, D]
    result = torch.einsum('bhd,hde->bhe', output.float(), T_V_inv_q)

    return result.to(query.dtype)
