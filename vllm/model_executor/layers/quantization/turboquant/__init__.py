# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant / WUSH-KV: KV-cache quantization for vLLM.

TurboQuant presets use a fixed Hadamard rotation; WUSH presets
(``wush_3bit``, ``wush_4bit``) replace it with a data-dependent
linear transform calibrated per layer and KV-head, significantly
reducing quantization error.

The Hadamard-rotation approach is the scalar case of the HIGGS
quantization method (Malinovskii et al., NAACL 2025;
arXiv:2411.17525), first applied to KV-cache compression in
"Cache Me If You Must" (Shutova et al., ICML 2025;
arXiv:2501.19392).  Both pre-date the TurboQuant paper
(Zandieh et al., ICLR 2026).

The WUSH data-dependent transform generalises this to the non-
isotropic case via joint weight-activation Hessian diagonalisation
(Alistarh et al., 2025).
"""

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
    is_tq_cache_dtype,
)

__all__ = ["TurboQuantConfig", "is_tq_cache_dtype"]
