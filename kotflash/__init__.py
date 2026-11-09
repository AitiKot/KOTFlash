# SPDX-License-Identifier: Apache-2.0
"""KOTFlash: independent experimental attention kernels for NVIDIA Turing GPUs."""
from .attention import attention
from .kernels.triton_forward import triton_available, triton_turing_supported
from .cache import append_kv_cache, decode_step

__all__ = ["attention", "append_kv_cache", "decode_step", "triton_available", "triton_turing_supported"]
__version__ = "0.2.0"
