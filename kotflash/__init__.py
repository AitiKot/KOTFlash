# SPDX-License-Identifier: Apache-2.0
"""KOTFlash: independent experimental attention kernels for NVIDIA Turing GPUs."""
from .attention import attention
from .kernels.triton_forward import triton_available, triton_turing_supported

__all__ = ["attention", "triton_available", "triton_turing_supported"]
__version__ = "0.1.1"
