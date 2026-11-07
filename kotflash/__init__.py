"""KOTFlash: experimental memory-efficient attention for NVIDIA Turing GPUs."""
from .attention import attention
from .kernels.triton_forward import triton_available

__all__ = ["attention", "triton_available"]
__version__ = "0.1.0"
