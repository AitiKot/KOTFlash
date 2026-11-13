# SPDX-License-Identifier: Apache-2.0
"""Optional JIT-built CUDA C++ operator for one-launch K/V cache appends."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_extension():
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch C++ extension support is unavailable") from exc

    source_dir = Path(__file__).resolve().parent / "csrc"
    cpp = source_dir / "kv_append.cpp"
    cu = source_dir / "kv_append_cuda.cu"
    if not cpp.is_file() or not cu.is_file():
        raise RuntimeError(f"KOTFlash CUDA append sources are missing under {source_dir}")
    return load(
        name="kotflash_kv_append",
        sources=[str(cpp), str(cu)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def append_kv_cuda(k_cache, v_cache, new_k, new_v, start: int) -> None:
    """Append K and V to preallocated FP16 CUDA buffers with one kernel launch."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the optional KOTFlash append extension")
    tensors = (k_cache, v_cache, new_k, new_v)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all cache and source tensors must be CUDA tensors")
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError("the optional CUDA append extension currently supports FP16 only")
    if any(tensor.device != k_cache.device for tensor in tensors):
        raise ValueError("all cache and source tensors must be on the same CUDA device")
    _load_extension().append_kv_(k_cache, v_cache, new_k, new_v, int(start))
