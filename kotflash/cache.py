# SPDX-License-Identifier: Apache-2.0
"""Small caller-managed KV-cache helpers.

These helpers use torch.cat for simplicity and correctness. Production
inference code should normally write into preallocated cache storage.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .attention import attention


def append_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append new [B, Hkv, Snew, D] entries without mutating the old cache."""
    tensors = (k_cache, v_cache, new_k, new_v)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("K/V cache and new K/V values must be torch.Tensor instances")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("All K/V tensors must use [B, Hkv, S, D] layout")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    if new_k.shape != new_v.shape:
        raise ValueError("New K and V shapes must match")
    if (
        k_cache.shape[0] != new_k.shape[0]
        or k_cache.shape[1] != new_k.shape[1]
        or k_cache.shape[3] != new_k.shape[3]
    ):
        raise ValueError("New K/V batch, head count and head dimension must match the cache")
    if (
        k_cache.device != v_cache.device
        or k_cache.device != new_k.device
        or new_k.device != new_v.device
    ):
        raise ValueError("All K/V tensors must be on the same device")
    if (
        k_cache.dtype != v_cache.dtype
        or k_cache.dtype != new_k.dtype
        or new_k.dtype != new_v.dtype
    ):
        raise ValueError("All K/V tensors must have the same dtype")
    if k_cache.shape[2] == 0 or new_k.shape[2] == 0:
        raise ValueError("Cache and appended K/V sequence lengths must be non-zero")

    return (
        torch.cat((k_cache, new_k), dim=2),
        torch.cat((v_cache, new_v), dim=2),
    )


def decode_step(
    q_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one token and attend to the now fully-visible cache.

    For one newest query token, every entry in the updated cache is visible.
    This uses causal=False. Never pass future K/V entries to this function.
    """
    if not isinstance(q_new, torch.Tensor) or q_new.ndim != 4:
        raise ValueError("q_new must use [B, Hq, 1, D] layout")
    if q_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new query token")
    if not isinstance(k_new, torch.Tensor) or k_new.ndim != 4 or k_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new K token")
    if not isinstance(v_new, torch.Tensor) or v_new.ndim != 4 or v_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new V token")

    updated_k, updated_v = append_kv_cache(k_cache, v_cache, k_new, v_new)
    if q_new.shape[0] != updated_k.shape[0] or q_new.shape[-1] != updated_k.shape[-1]:
        raise ValueError("Query batch and head dimension must match the KV cache")

    output = attention(
        q_new,
        updated_k,
        updated_v,
        causal=False,
        backend=backend,
    )
    return output, updated_k, updated_v
