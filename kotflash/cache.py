# SPDX-License-Identifier: Apache-2.0
"""Caller-managed KV-cache helpers for incremental attention decoding.

These helpers do not implement paging, eviction, batching or model projections.
"""
from __future__ import annotations

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
    The current Triton attention wrapper makes contiguous inputs before launch;
    therefore this class avoids torch.cat reallocation but is not zero-copy
    end-to-end. contiguous_view() explicitly copies the populated prefix.

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

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append one or more positions after validating the complete write."""
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

        end = self.length + new_k.shape[2]
        if end > self.max_len:
            raise ValueError(
                f"KV cache full: {self.length} + {new_k.shape[2]} tokens "
                f"exceeds max_len={self.max_len}"
            )
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
