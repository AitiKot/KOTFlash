# SPDX-License-Identifier: Apache-2.0
"""Tests for fixed-capacity, caller-managed KV cache storage."""
import pytest
import torch

from kotflash.cache import PreallocatedKVCache


def test_preallocated_cache_appends_in_order_and_exposes_only_valid_prefix():
    cache = PreallocatedKVCache(1, 2, 8, 4, device="cpu", dtype=torch.float32)
    k1 = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    v1 = -k1
    k2 = torch.full((1, 2, 1, 4), 100.0)
    v2 = -k2
    cache.append(k1, v1)
    cache.append(k2, v2)

    assert cache.length == 4
    actual_k, actual_v = cache.contiguous_view()
    torch.testing.assert_close(actual_k, torch.cat((k1, k2), dim=2))
    torch.testing.assert_close(actual_v, torch.cat((v1, v2), dim=2))
    assert actual_k.is_contiguous() and actual_v.is_contiguous()


def test_preallocated_cache_does_not_expose_unused_slots():
    cache = PreallocatedKVCache(1, 1, 5, 4, device="cpu", dtype=torch.float32)
    cache.k.fill_(12345.0)
    cache.v.fill_(-12345.0)
    k = torch.ones(1, 1, 2, 4)
    v = torch.full_like(k, 2.0)
    cache.append(k, v)
    actual_k, actual_v = cache.contiguous_view()
    assert actual_k.shape[2] == 2
    assert actual_v.shape[2] == 2
    assert torch.equal(actual_k, k)
    assert torch.equal(actual_v, v)


def test_preallocated_cache_rejects_invalid_append_and_overflow():
    cache = PreallocatedKVCache(1, 2, 3, 4, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="empty"):
        cache.contiguous_view()
    with pytest.raises(ValueError, match="batch, heads"):
        cache.append(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    with pytest.raises(ValueError, match="dtype"):
        cache.append(torch.zeros(1, 2, 1, 4, dtype=torch.float16),
                     torch.zeros(1, 2, 1, 4, dtype=torch.float16))
    cache.append(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))
    with pytest.raises(ValueError, match="full"):
        cache.append(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))


def test_preallocated_cache_rejects_shape_mismatch():
    cache = PreallocatedKVCache(1, 2, 4, 4, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="shapes must match"):
        cache.append(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 2, 4))
