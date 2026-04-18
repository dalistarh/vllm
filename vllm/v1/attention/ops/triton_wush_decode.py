# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused WUSH-KV decode attention.

WUSH variant of the TurboQuant decode kernel.  Keys stored in
pre-RoPE WUSH-transformed space; kernel applies inverse WUSH
(T_K^T matmul) and RoPE per cached token during scoring.

Values accumulated in WUSH space; T_V^{-1} applied once after
stage-2 reduction — saves O(seq_len * D^2) vs per-token dequant.

Cache layout identical to TurboQuant (shared store kernel).
"""

import math
from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)


@triton.jit
def _wush_decode_stage1(
    Q_ptr,              # [B, Hq, D] float32 — post-RoPE query
    KV_cache_ptr,       # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,    # [B, max_num_blocks] int32
    Seq_lens_ptr,       # [B] int32
    Centroids_ptr,      # [n_centroids] float32
    T_K_T_ptr,          # [Hk, D, D] float32 — per-head T_K transposed
    Cos_ptr,            # [max_pos, D/2] float32
    Sin_ptr,            # [max_pos, D/2] float32
    Scratch_ptr,        # [B, Hq, D] float32 — scratch for rotate_half
    Mid_o_ptr,          # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb, stride_qh,
    stride_cb, stride_cp, stride_ch,
    stride_bt,
    stride_mb, stride_mh, stride_ms,
    stride_th, stride_tr, stride_tc,
    stride_sb, stride_sh,
    stride_cos_pos,
    # Constexpr
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    TILE_K: tl.constexpr = 32,
):
    """Process one KV-split: dequant K + T_K^T + RoPE + score; accumulate V."""
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)
    kv_head = hid // KV_GROUP_SIZE

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # Load query once (post-RoPE)
    q_base = bid * stride_qb + hid * stride_qh
    q_vec = tl.load(Q_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Scratch buffer base for this (batch, q-head) — used for rotate_half
    scratch_base = bid * stride_sb + hid * stride_sh

    # MSE bit-unpacking precompute
    mse_bit_off = d_offs * MSE_BITS
    mse_byte_idx = mse_bit_off // 8
    mse_bit_shift = mse_bit_off % 8
    mse_umask = (1 << MSE_BITS) - 1

    # Value bit-unpacking precompute
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # RoPE rotate_half index: for d < D/2 read from d+D/2, else d-D/2
    rot_idx = tl.where(d_offs < HALF_DIM, d_offs + HALF_DIM, d_offs - HALF_DIM)
    rot_sign = tl.where(d_offs < HALF_DIM, -1.0, 1.0)

    # T_K^T base for this KV-head
    tk_base = kv_head * stride_th

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt

    # ================================================================
    # Process one token at a time (BLOCK_KV=1 for K-side matmul)
    # ================================================================
    for pos in range(split_start, split_end):
        # Page lookup
        page_idx = pos // BLOCK_SIZE
        page_off = pos % BLOCK_SIZE
        block_num = tl.load(Block_table_ptr + bt_base + page_idx).to(tl.int64)
        slot_base = (block_num * stride_cb
                     + tl.cast(page_off, tl.int64) * stride_cp
                     + tl.cast(kv_head, tl.int64) * stride_ch)

        # ---- K dequant: centroid lookup + norm ----
        mse_addrs = slot_base + mse_byte_idx
        mr0 = tl.load(KV_cache_ptr + mse_addrs, mask=d_mask, other=0).to(tl.int32)
        mr1 = tl.load(KV_cache_ptr + mse_addrs + 1, mask=d_mask, other=0).to(tl.int32)
        idx = ((mr0 | (mr1 << 8)) >> mse_bit_shift) & mse_umask
        c = tl.load(Centroids_ptr + idx, mask=d_mask, other=0.0)

        if NORM_CORRECTION:
            cn2 = tl.sum(tl.where(d_mask, c * c, 0.0))
            c = c * (1.0 / tl.sqrt(cn2 + 1e-16))

        nb = slot_base + MSE_BYTES
        nlo = tl.load(KV_cache_ptr + nb).to(tl.uint16)
        nhi = tl.load(KV_cache_ptr + nb + 1).to(tl.uint16)
        vnorm = (nlo | (nhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_wush = c * vnorm  # [BLOCK_D] — key in WUSH space

        # ---- K: T_K^T matmul (tiled over contraction dim) ----
        # k_pre[d_out] = sum_{d_in} k_wush[d_in] * T_K_T[kv_head, d_in, d_out]
        k_pre = tl.zeros([BLOCK_D], dtype=tl.float32)
        for tile in range(0, HEAD_DIM, TILE_K):
            t_offs = tile + tl.arange(0, TILE_K)
            t_mask = t_offs < HEAD_DIM

            # Load T_K_T tile: [TILE_K, BLOCK_D]
            tk_addrs = (tk_base
                        + t_offs[:, None] * stride_tr
                        + d_offs[None, :] * stride_tc)
            tk_tile = tl.load(
                T_K_T_ptr + tk_addrs,
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            )

            # Extract k_wush elements for this tile
            k_tile = tl.load(
                Centroids_ptr + tl.where(t_mask, idx, 0),
                mask=t_mask, other=0.0,
            )
            # Need the actual k_wush values, not re-gathered centroids
            # Write k_wush to scratch, read back the tile
            tl.store(Scratch_ptr + scratch_base + d_offs, k_wush, mask=d_mask)
            k_slice = tl.load(
                Scratch_ptr + scratch_base + t_offs,
                mask=t_mask, other=0.0,
            )

            # Accumulate: k_pre += k_slice @ tk_tile
            k_pre += tl.sum(k_slice[:, None] * tk_tile, axis=0)

        # ---- K: apply RoPE ----
        # Write k_pre to scratch for rotate_half gather
        tl.store(Scratch_ptr + scratch_base + d_offs, k_pre, mask=d_mask)

        # Load cos/sin for this position (half-dim, duplicated)
        half_offs = tl.arange(0, BLOCK_D)
        h_mask = half_offs < HALF_DIM
        cos_h = tl.load(Cos_ptr + pos * stride_cos_pos + half_offs,
                        mask=h_mask, other=1.0)
        sin_h = tl.load(Sin_ptr + pos * stride_cos_pos + half_offs,
                        mask=h_mask, other=0.0)
        # Expand to full dim: cos[d] = cos_h[d % HALF_DIM]
        cos_full = tl.where(d_offs < HALF_DIM, cos_h, cos_h)
        sin_full = tl.where(d_offs < HALF_DIM, sin_h, sin_h)

        # rotate_half via scratch gather
        k_rot = rot_sign * tl.load(
            Scratch_ptr + scratch_base + rot_idx,
            mask=d_mask, other=0.0,
        )
        k_rope = k_pre * cos_full + k_rot * sin_full

        # ---- Score ----
        score = tl.sum(tl.where(d_mask, q_vec * k_rope, 0.0)) * ATTN_SCALE

        # ---- Online softmax ----
        n_e_max = tl.maximum(score, m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(score - n_e_max)

        # ---- V dequant + accumulate (in WUSH space) ----
        vb = slot_base + KPS
        if VQB == 3:
            va0 = vb + val_byte_idx
            vr0 = tl.load(KV_cache_ptr + va0, mask=d_mask, other=0).to(tl.int32)
            vr1 = tl.load(KV_cache_ptr + va0 + 1, mask=d_mask, other=0).to(tl.int32)
            vi = (((vr0 | (vr1 << 8)) >> val_bit_shift) & 0x7).to(tl.float32)
            sb = vb + VAL_DATA_BYTES
            slo = tl.load(KV_cache_ptr + sb).to(tl.uint16)
            shi = tl.load(KV_cache_ptr + sb + 1).to(tl.uint16)
            vs = (slo | (shi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zlo = tl.load(KV_cache_ptr + sb + 2).to(tl.uint16)
            zhi = tl.load(KV_cache_ptr + sb + 3).to(tl.uint16)
            vz = (zlo | (zhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            vals = vi * vs + vz
        else:
            vbi = d_offs // 2
            vbs = (d_offs % 2) * 4
            vr = tl.load(KV_cache_ptr + vb + vbi, mask=d_mask, other=0).to(tl.int32)
            vi = ((vr >> vbs) & 0xF).to(tl.float32)
            sb = vb + VAL_DATA_BYTES
            slo = tl.load(KV_cache_ptr + sb).to(tl.uint16)
            shi = tl.load(KV_cache_ptr + sb + 1).to(tl.uint16)
            vs = (slo | (shi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zlo = tl.load(KV_cache_ptr + sb + 2).to(tl.uint16)
            zhi = tl.load(KV_cache_ptr + sb + 3).to(tl.uint16)
            vz = (zlo | (zhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            vals = vi * vs + vz

        acc = acc * re_scale + p * vals
        l_prev = l_prev * re_scale + p
        m_prev = n_e_max

    # Write partial result (V still in WUSH space)
    out_base = bid * stride_mb + hid * stride_mh + sid * stride_ms
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

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
    """Fused WUSH decode attention.

    K-side: dequant -> T_K^T (tiled matmul) -> RoPE -> score (all in-kernel)
    V-side: dequant -> accumulate in WUSH space -> T_V^{-1} applied once
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device
    half_dim = D // 2

    mse_bytes = math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = max_num_kv_splits

    # Scratch buffer for rotate_half (one vector per (batch, q-head))
    scratch = torch.empty(B, Hq, D, dtype=torch.float32, device=device)

    # Intermediate buffer
    if (mid_o_buf is not None
            and mid_o_buf.shape[0] >= B
            and mid_o_buf.shape[2] >= NUM_KV_SPLITS):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1,
                            dtype=torch.float32, device=device)

    T_K_T = T_K_T.contiguous().float()
    cos_cache = cos_cache.contiguous().float()
    sin_cache = sin_cache.contiguous().float()

    grid = (B, Hq, NUM_KV_SPLITS)
    _wush_decode_stage1[grid](
        query.float().contiguous(),
        kv_cache, block_table, seq_lens, centroids,
        T_K_T, cos_cache, sin_cache,
        scratch, mid_o,
        # Strides
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        T_K_T.stride(0), T_K_T.stride(1), T_K_T.stride(2),
        scratch.stride(0), scratch.stride(1),
        cos_cache.stride(0),
        # Constexpr
        NUM_KV_HEADS=Hk, HEAD_DIM=D, HALF_DIM=half_dim,
        BLOCK_SIZE=block_size, NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits, MSE_BYTES=mse_bytes, KPS=key_packed_size,
        VQB=value_quant_bits, VAL_DATA_BYTES=val_data_bytes,
        ATTN_SCALE=scale, BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if norm_correction else 0,
        num_warps=4,
    )

    # Stage 2: reduce across splits
    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)

    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    _fwd_kernel_stage2[(B, Hq)](
        mid_o, output, lse,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        output.stride(0), output.stride(1),
        lse.stride(0),
        HEAD_DIM=D, BLOCK_D=BLOCK_D, NUM_KV_SPLITS=NUM_KV_SPLITS,
        num_warps=4,
    )

    # Apply T_V^{-1} to output (one matmul, not per-token)
    # output: [B, Hq, D], T_V_inv: [Hk, D, D]
    T_V_inv_q = T_V_inv.repeat_interleave(kv_group_size, dim=0)  # [Hq, D, D]
    result = torch.einsum('bhd,hde->bhe', output, T_V_inv_q.float())

    return result.to(query.dtype)
