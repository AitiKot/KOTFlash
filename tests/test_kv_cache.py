# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the caller-managed KV-cache integration helpers."""
import pytest
import torch
import torch.nn.functional as F

from kotflash.cache import append_kv_cache, decode_step


def test_append_kv_cache_preserves_order_and_does_not_mutate_inputs():
    k_cache = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    v_cache = -k_cache
    new_k = torch.full((1, 2, 1, 4), 100.0)
    new_v = torch.full((1, 2, 1, 4), -100.0)
    old_k, old_v = k_cache.clone(), v_cache.clone()
    updated_k, updated_v = append_kv_cache(k_cache, v_cache, new_k, new_v)
    torch.testing.assert_close(updated_k, torch.cat((old_k, new_k), dim=2))
    torch.testing.assert_close(updated_v, torch.cat((old_v, new_v), dim=2))
    torch.testing.assert_close(k_cache, old_k)
    torch.testing.assert_close(v_cache, old_v)


@pytest.mark.parametrize("bad_field", ["batch", "heads", "dim", "dtype", "device"])
def test_append_kv_cache_rejects_incompatible_new_tensors(bad_field):
    k_cache = torch.zeros(1, 2, 4, 8)
    v_cache = torch.zeros_like(k_cache)
    new_k = torch.zeros(1, 2, 1, 8)
    new_v = torch.zeros_like(new_k)
    if bad_field == "batch":
        new_k = torch.zeros(2, 2, 1, 8); new_v = torch.zeros_like(new_k)
    elif bad_field == "heads":
        new_k = torch.zeros(1, 1, 1, 8); new_v = torch.zeros_like(new_k)
    elif bad_field == "dim":
        new_k = torch.zeros(1, 2, 1, 16); new_v = torch.zeros_like(new_k)
    elif bad_field == "dtype":
        new_k = new_k.half(); new_v = new_v.half()
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        new_k = new_k.cuda(); new_v = new_v.cuda()
    with pytest.raises(ValueError):
        append_kv_cache(k_cache, v_cache, new_k, new_v)


def test_decode_step_matches_expanded_sdpa_for_one_visible_token():
    torch.manual_seed(7)
    q = torch.randn(1, 4, 1, 16)
    k_cache = torch.randn(1, 2, 5, 16)
    v_cache = torch.randn_like(k_cache)
    k_new = torch.randn(1, 2, 1, 16)
    v_new = torch.randn_like(k_new)
    actual, updated_k, updated_v = decode_step(q, k_cache, v_cache, k_new, v_new, backend="sdpa")
    expected = F.scaled_dot_product_attention(
        q, updated_k.repeat_interleave(2, dim=1),
        updated_v.repeat_interleave(2, dim=1), is_causal=False,
    )
    torch.testing.assert_close(actual, expected)
    assert updated_k.shape[2] == 6 and updated_v.shape[2] == 6


def test_decode_step_rejects_multi_token_queries_before_appending():
    q = torch.randn(1, 4, 2, 16)
    cache = torch.randn(1, 2, 4, 16)
    new = torch.randn(1, 2, 1, 16)
    with pytest.raises(ValueError, match="exactly one new query token"):
        decode_step(q, cache, cache, new, new, backend="sdpa")
