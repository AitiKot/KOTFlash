# SPDX-License-Identifier: Apache-2.0
"""Tests for fixed-capacity, caller-managed KV cache storage."""
import pytest
import torch

from kotflash.cache import PreallocatedKVCache


def test_preallocated_cache_appends_and_returns_only_populated_prefix():
    cache = PreallocatedKVCache(1, 2, 8, 4, device="cpu", dtype=torch.float32)
    k1 = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    v1 = -k1
    k2 = torch.full((1, 2, 1, 4), 100.0)
    v2 = -k2
    cache.append(k1, v1)
    cache.append(k2, v2)
    assert cache.length == 4
    view_k, view_v = cache.view()
    torch.testing.assert_close(view_k, torch.cat((k1, k2), dim=2))
    torch.testing.assert_close(view_v, torch.cat((v1, v2), dim=2))
    assert view_k.untyped_storage().data_ptr() == cache.k.untyped_storage().data_ptr()
    copied_k, copied_v = cache.contiguous_view()
    assert copied_k.is_contiguous() and copied_v.is_contiguous()
    torch.testing.assert_close(copied_k, view_k)
    torch.testing.assert_close(copied_v, view_v)


def test_preallocated_cache_does_not_expose_unused_slots():
    cache = PreallocatedKVCache(1, 1, 5, 4, device="cpu", dtype=torch.float32)
    cache.k.fill_(12345.0)
    cache.v.fill_(-12345.0)
    k = torch.ones(1, 1, 2, 4)
    v = torch.full_like(k, 2.0)
    cache.append(k, v)
    actual_k, actual_v = cache.view()
    assert actual_k.shape[2] == actual_v.shape[2] == 2
    torch.testing.assert_close(actual_k, k)
    torch.testing.assert_close(actual_v, v)


def test_preallocated_cache_rejects_invalid_append_without_advancing():
    cache = PreallocatedKVCache(1, 2, 3, 4, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="empty"):
        cache.view()
    with pytest.raises(ValueError, match="batch, heads"):
        cache.append(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    with pytest.raises(ValueError, match="dtype"):
        cache.append(torch.zeros(1, 2, 1, 4, dtype=torch.float16),
                     torch.zeros(1, 2, 1, 4, dtype=torch.float16))
    assert cache.length == 0
    cache.append(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))
    with pytest.raises(ValueError, match="full"):
        cache.append(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))
    assert cache.length == 2


def test_preallocated_cache_rejects_shape_mismatch():
    cache = PreallocatedKVCache(1, 2, 4, 4, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="shapes must match"):
        cache.append(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 2, 4))
    assert cache.length == 0


def test_preallocated_cache_attention_matches_manual_sdpa():
    torch.manual_seed(17)
    cache = PreallocatedKVCache(1, 2, 8, 16, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 5, 16)
    v = torch.randn_like(k)
    cache.append(k, v)
    q = torch.randn(1, 4, 1, 16)
    out = cache.attend(q, backend="sdpa")
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=False
    )
    torch.testing.assert_close(out, expected)
