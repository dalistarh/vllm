# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WUSH-KV transform utilities for data-dependent KV-cache quantization.

WUSH replaces the fixed Hadamard rotation used in TurboQuant with a
per-layer, per-KV-head linear transform T calibrated from model
statistics.  Keys are stored in the pre-RoPE transformed space; values
are stored in the output-projection-weighted transformed space.

Reference: Alistarh et al., "WUSH-KV: Data-Aware KV-Cache Quantization",
2025.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transform loading
# ---------------------------------------------------------------------------

def load_wush_transforms(
    path: str,
    num_layers: int,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    """Load pre-calibrated WUSH transforms from a safetensors or .pt file.

    Expected keys (for each layer index *i*):
      ``k.{i}``       – T_K,  shape [num_kv_heads, head_dim, head_dim]
      ``k_inv.{i}``   – T_K^{-1}
      ``v_t.{i}``     – T_V^T
      ``v_inv_t.{i}`` – T_V^{-T}

    If the inverse/transpose variants are absent, they are computed from
    the forward transforms (``k.{i}`` and ``v.{i}``).

    Returns ``{"k": {i: T_K}, "k_inv": {i: T_K_inv}, ...}`` dicts.
    """
    ext = Path(path).suffix.lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        raw = load_file(path, device=str(device))
    elif ext in (".pt", ".pth", ".bin"):
        raw = torch.load(path, map_location=device, weights_only=True)
    else:
        raise ValueError(f"Unsupported transform file format: {ext}")

    result: dict[str, dict[int, torch.Tensor]] = {
        "k": {}, "k_inv": {}, "v_t": {}, "v_inv_t": {},
    }

    for i in range(num_layers):
        # --- K-side ---
        k_key = f"k.{i}"
        k_inv_key = f"k_inv.{i}"
        if k_key in raw:
            T_K = raw[k_key].float().to(device)
            result["k"][i] = T_K
            if k_inv_key in raw:
                result["k_inv"][i] = raw[k_inv_key].float().to(device)
            else:
                result["k_inv"][i] = torch.linalg.inv(T_K)
        elif k_inv_key in raw:
            T_K_inv = raw[k_inv_key].float().to(device)
            result["k_inv"][i] = T_K_inv
            result["k"][i] = torch.linalg.inv(T_K_inv)

        # --- V-side ---
        v_key = f"v.{i}"
        v_t_key = f"v_t.{i}"
        v_inv_t_key = f"v_inv_t.{i}"
        if v_t_key in raw:
            result["v_t"][i] = raw[v_t_key].float().to(device)
        elif v_key in raw:
            result["v_t"][i] = raw[v_key].float().to(device).transpose(-2, -1)

        if v_inv_t_key in raw:
            result["v_inv_t"][i] = raw[v_inv_t_key].float().to(device)
        elif v_key in raw:
            T_V = raw[v_key].float().to(device)
            result["v_inv_t"][i] = torch.linalg.inv(T_V).transpose(-2, -1)
        elif v_t_key in raw:
            T_V_T = result["v_t"][i]
            result["v_inv_t"][i] = torch.linalg.inv(T_V_T.transpose(-2, -1)).transpose(-2, -1)

    return result


def discover_wush_transforms_path(model_path: str) -> Optional[str]:
    """Auto-discover WUSH transforms bundled with a model."""
    for name in ("wush_transforms.safetensors", "wush_transforms.pt"):
        candidate = os.path.join(model_path, name)
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def build_rope_cache(
    head_dim: int,
    max_position: int,
    rope_theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute [max_position, head_dim] cos/sin RoPE tables.

    Uses the standard "rotate_half" convention where the first half and
    second half of the head dimension form sin/cos pairs.
    """
    half = head_dim // 2
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_position, device=device).float()
    angles = torch.outer(positions, freqs)           # [max_position, half]
    cos = torch.cos(angles).to(torch.float32)        # [max_position, half]
    sin = torch.sin(angles).to(torch.float32)        # [max_position, half]
    # Expand to full head_dim: [max_position, head_dim] with repeated pairs
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation.  cos/sin broadcastable to x's last dims."""
    return x * cos + _rotate_half(x) * sin


def undo_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Invert RoPE rotation: applies RoPE with negated sin."""
    return x * cos + _rotate_half(x) * (-sin)


# ---------------------------------------------------------------------------
# Per-head transform application
# ---------------------------------------------------------------------------

def apply_wush_forward_k(
    key: torch.Tensor,
    T_K_inv: torch.Tensor,
) -> torch.Tensor:
    """K-side forward: k_wush = key @ T_K^{-1}, per KV-head.

    Args:
        key:     [num_tokens, num_kv_heads, head_dim]
        T_K_inv: [num_kv_heads, head_dim, head_dim]
    """
    return torch.einsum("thd,hde->the", key.float(), T_K_inv).to(key.dtype)


def apply_wush_forward_v(
    value: torch.Tensor,
    T_V_T: torch.Tensor,
) -> torch.Tensor:
    """V-side forward: v_wush = value @ T_V^T, per KV-head.

    Args:
        value: [num_tokens, num_kv_heads, head_dim]
        T_V_T: [num_kv_heads, head_dim, head_dim]
    """
    return torch.einsum("thd,hde->the", value.float(), T_V_T).to(value.dtype)


def apply_wush_inverse_k(
    k_wush: torch.Tensor,
    T_K: torch.Tensor,
) -> torch.Tensor:
    """K-side inverse: k_pre_rope = k_wush @ T_K, per KV-head.

    Args:
        k_wush: [*, num_kv_heads, head_dim]  (any leading dims)
        T_K:    [num_kv_heads, head_dim, head_dim]
    """
    return torch.einsum("...hd,hde->...he", k_wush.float(), T_K).to(k_wush.dtype)


def apply_wush_inverse_v(
    v_wush: torch.Tensor,
    T_V_inv_T: torch.Tensor,
) -> torch.Tensor:
    """V-side inverse: v = v_wush @ T_V^{-T}, per KV-head.

    Args:
        v_wush:    [*, num_kv_heads, head_dim]
        T_V_inv_T: [num_kv_heads, head_dim, head_dim]
    """
    return torch.einsum("...hd,hde->...he", v_wush.float(), T_V_inv_T).to(v_wush.dtype)
