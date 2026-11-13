# SPDX-License-Identifier: Apache-2.0
"""Caller-managed KV-cache helpers for incremental attention decoding.

These helpers do not implement paging, eviction, batching or model projections.
"""
from __future__ import annotations

import os
from typing import Tuple, Union

import torch

from .attention import attention


def _validate_kv_tensors(k_cache, v_cache, new_k, new_v) -> None:
    tensors = (k_cache, v_cache, new_k, new_v)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("K/V cache and new K/V values must be torch.Tensor instances")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("All K/V tensors must use [B, Hkv, S, D] layout")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    if k_cache.shape[0] <= 0 or k_cache.shape[1] <= 0 or k_cache.shape[3] <= 0:
        raise ValueError("K/V cache batch, head count and head dimension must be non-zero")
    if new_k.shape != new_v.shape:
        raise ValueError("New K and V shapes must match")
    if (
        k_cache.shape[0] != new_k.shape[0]
        or k_cache.shape[1] != new_k.shape[1]
        or k_cache.shape[3] != new_k.shape[3]
    ):
        raise ValueError("New K/V batch, head count and head dimension must match the cache")
    if any(tensor.device != k_cache.device for tensor in tensors):
        raise ValueError("All K/V tensors must be on the same device")
    if any(tensor.dtype != k_cache.dtype for tensor in tensors):
        raise ValueError("All K/V tensors must have the same dtype")
    if new_k.shape[2] == 0:
        raise ValueError("Appended K/V sequence length must be non-zero")


def append_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append [B, Hkv, Snew, D] entries without mutating the old cache.

    This convenience function uses torch.cat and copies the complete cache.
    Use PreallocatedKVCache for fixed-capacity storage.
    """
    _validate_kv_tensors(k_cache, v_cache, new_k, new_v)
    return torch.cat((k_cache, new_k), dim=2), torch.cat((v_cache, new_v), dim=2)


def decode_step(
    q_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one K/V token and attend with one newest query token.

    The updated cache contains only positions visible to the new query, so the
    attention call uses causal=False. Never pass future K/V positions here.
    """
    _validate_kv_tensors(k_cache, v_cache, k_new, v_new)
    if k_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new K/V token")
    if not isinstance(q_new, torch.Tensor) or q_new.ndim != 4:
        raise ValueError("q_new must use [B, Hq, 1, D] layout")
    if q_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new query token")
    if q_new.shape[0] != k_cache.shape[0] or q_new.shape[-1] != k_cache.shape[-1]:
        raise ValueError("Query batch and head dimension must match the KV cache")
    if q_new.shape[1] <= 0 or q_new.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("Query heads must be non-zero and divisible by the cache KV-head count")
    if q_new.device != k_cache.device or q_new.dtype != k_cache.dtype:
        raise ValueError("Query device and dtype must match the KV cache")

    updated_k = torch.cat((k_cache, k_new), dim=2)
    updated_v = torch.cat((v_cache, v_new), dim=2)
    output = attention(q_new, updated_k, updated_v, causal=False, backend=backend)
    return output, updated_k, updated_v


class PreallocatedKVCache:
    """Fixed-capacity K/V storage with append-only writes.

    view() returns storage-backed prefix views, which may be non-contiguous.
    The SM75 inference decode kernel can read supported padded prefixes
    directly. Training/autograd and unprofiled shapes retain portable fallback
    paths. contiguous_view() explicitly copies the populated prefix.

    append() defaults to portable PyTorch copies. An optional JIT-built
    CUDA C++ operator can combine K/V writes into one kernel launch.

    This class is not thread-safe and does not implement paging or eviction.
    """

    def __init__(
        self,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        *,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        dims = {"batch": batch, "kv_heads": kv_heads, "max_len": max_len, "head_dim": head_dim}
        for name, value in dims.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.k = torch.empty((batch, kv_heads, max_len, head_dim), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.max_len = max_len
        self.length = 0

    def append(
        self, new_k: torch.Tensor, new_v: torch.Tensor, *, backend: str = "auto"
    ) -> None:
        """Append positions after validation.

        backend="torch" uses portable tensor copy operations. backend="cuda"
        forces the optional FP16 CUDA C++ extension. backend="auto" uses that
        extension only when KOTFLASH_ENABLE_CUDA_APPEND=1 is set.
        """
        if not isinstance(new_k, torch.Tensor) or not isinstance(new_v, torch.Tensor):
            raise TypeError("new_k and new_v must be torch.Tensor instances")
        if new_k.ndim != 4 or new_v.ndim != 4:
            raise ValueError("new K/V must use [B, Hkv, S, D] layout")
        if new_k.shape != new_v.shape:
            raise ValueError("new K and V shapes must match")
        if (
            new_k.shape[0] != self.k.shape[0]
            or new_k.shape[1] != self.k.shape[1]
            or new_k.shape[3] != self.k.shape[3]
        ):
            raise ValueError("new K/V batch, heads and head_dim must match the cache")
        if new_k.shape[2] <= 0:
            raise ValueError("cannot append an empty K/V sequence")
        if new_k.device != self.k.device or new_v.device != self.v.device:
            raise ValueError("new K/V must be on the cache device")
        if new_k.dtype != self.k.dtype or new_v.dtype != self.v.dtype:
            raise ValueError("new K/V dtype must match the cache dtype")

        if backend not in {"auto", "torch", "cuda"}:
            raise ValueError("append backend must be one of: auto, torch or cuda")
        end = self.length + new_k.shape[2]
        if end > self.max_len:
            raise ValueError(
                f"KV cache full: {self.length} + {new_k.shape[2]} tokens "
                f"exceeds max_len={self.max_len}"
            )

        # The optional C++/CUDA operator copies K and V in a single launch.
        # It is opt-in because compiling a CUDA extension requires a working
        # local CUDA toolkit (including nvcc). The PyTorch path stays portable.
        use_cuda = backend == "cuda" or (
            backend == "auto"
            and os.environ.get("KOTFLASH_ENABLE_CUDA_APPEND") == "1"
            and self.k.is_cuda
            and self.k.dtype == torch.float16
        )
        if use_cuda:
            from ._cuda_append import append_kv_cuda

            append_kv_cuda(self.k, self.v, new_k, new_v, self.length)
        else:
            # All user-facing validation is complete before either buffer is written.
            self.k[:, :, self.length:end].copy_(new_k)
            self.v[:, :, self.length:end].copy_(new_v)
        self.length = end

    def view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return storage-backed views of only the populated cache prefix."""
        if self.length == 0:
            raise ValueError("KV cache is empty")
        return self.k[:, :, :self.length, :], self.v[:, :, :self.length, :]

    def contiguous_view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous copies of only the populated cache prefix."""
        k, v = self.view()
        return k.contiguous(), v.contiguous()

    def append_and_attend(
        self,
        q_new: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        append_backend: str = "auto",
        attention_backend: str = "auto",
    ) -> torch.Tensor:
        """Append one token and attend without copying the existing cache prefix.

        On the exact measured T4 profile, append_backend="auto" fuses the new
        K/V write with Split-KV partial attention in one kernel launch, followed
        by a partial-result reduction. Other supported inference shapes use
        direct stride-aware attention after a normal cache append. The fused
        path is inference-only and has no backward implementation.
        """
        if attention_backend not in {"auto", "triton", "sdpa"}:
            raise ValueError("attention backend must be one of: auto, triton or sdpa")
        if append_backend not in {"auto", "torch", "cuda", "fused"}:
            raise ValueError("append backend must be one of: auto, torch, cuda or fused")
        if not isinstance(q_new, torch.Tensor) or q_new.ndim != 4 or q_new.shape[2] != 1:
            raise ValueError("q_new must use [B, Hq, 1, D] layout")
        if q_new.shape[0] != self.k.shape[0] or q_new.shape[-1] != self.k.shape[-1]:
            raise ValueError("Query batch and head dimension must match the cache")
        if q_new.shape[1] <= 0 or q_new.shape[1] % self.k.shape[1] != 0:
            raise ValueError("Query heads must be non-zero and divisible by the cache KV-head count")
        if q_new.device != self.k.device or q_new.dtype != self.k.dtype:
            raise ValueError("Query device and dtype must match the cache")

        _validate_kv_tensors(self.k, self.v, new_k, new_v)
        if new_k.shape[2] != 1:
            raise ValueError("append_and_attend accepts exactly one new K/V token")
        end = self.length + 1
        if end > self.max_len:
            raise ValueError(
                f"KV cache full: {self.length} + 1 tokens exceeds max_len={self.max_len}"
            )

        # On the exact measured T4 decode profile, auto combines the append
        # write with Split-KV partial attention in one Triton launch. A second
        # kernel merges partial results. Grad-enabled calls keep the portable
        # append + attention path; the fused inference kernel has no backward.
        use_fused = append_backend == "fused"
        if append_backend == "auto" and attention_backend in {"auto", "triton"}:
            from .attention import _is_measured_t4_runtime
            from .kernels.triton_decode import strided_decode_supported
            device_index = q_new.device.index
            if device_index is None and q_new.is_cuda:
                device_index = torch.cuda.current_device()
            use_fused = (
                q_new.is_cuda
                and tuple(q_new.shape) == (2, 8, 1, 64)
                and tuple(new_k.shape) == (2, 2, 1, 64)
                and self.length == 2047
                and strided_decode_supported(q_new, self.k, self.v)
                and _is_measured_t4_runtime(device_index)
                and os.environ.get("KOTFLASH_ENABLE_CUDA_APPEND") != "1"
            )

        has_grad_inputs = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q_new, new_k, new_v)
        )
        if append_backend == "auto" and has_grad_inputs:
            use_fused = False
        if use_fused:
            if attention_backend == "sdpa":
                raise ValueError("append_backend='fused' requires attention_backend='auto' or 'triton'")
            if has_grad_inputs:
                raise RuntimeError("append_backend='fused' is inference-only; disable gradients")
            from .kernels.triton_forward import triton_turing_supported
            from .kernels.triton_decode import attention_forward_strided_append
            if not triton_turing_supported():
                raise RuntimeError("fused append/decode requires Turing-compatible Triton 3.2.x")
            output = attention_forward_strided_append(
                q_new, self.k, self.v, new_k, new_v,
                past_length=self.length,
                scale=1.0 / (self.k.shape[-1] ** 0.5),
            )
            self.length = end
            return output

        self.append(
            new_k, new_v,
            backend="torch" if append_backend == "fused" else append_backend,
        )
        return self.attend(q_new, backend=attention_backend)

    def attend(self, q_new: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
        """Attend one newest query token to the currently populated cache."""
        if not isinstance(q_new, torch.Tensor) or q_new.ndim != 4 or q_new.shape[2] != 1:
            raise ValueError("q_new must use [B, Hq, 1, D] layout")
        if q_new.shape[0] != self.k.shape[0] or q_new.shape[-1] != self.k.shape[-1]:
            raise ValueError("Query batch and head dimension must match the cache")
        if q_new.shape[1] <= 0 or q_new.shape[1] % self.k.shape[1] != 0:
            raise ValueError("Query heads must be non-zero and divisible by the cache KV-head count")
        if q_new.device != self.k.device or q_new.dtype != self.k.dtype:
            raise ValueError("Query device and dtype must match the cache")
        k, v = self.view()
        return attention(q_new, k, v, causal=False, backend=backend)
