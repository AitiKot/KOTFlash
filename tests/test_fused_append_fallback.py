# SPDX-License-Identifier: Apache-2.0
"""Regression tests for best-effort fused KV-cache append fallback."""

import pytest
import torch
import torch.nn.functional as F

from kotflash.cache import PreallocatedKVCache


@pytest.mark.parametrize(
    ("head_dim", "past_length"),
    [
        (64, 0),       # First generated token.
        (64, 1022),    # Just below the minimum fused prefix length.
        (128, 1022),   # KOT-V2 D=128, also below the prefix threshold.
        (128, 1023),   # KOT-V2 D=128 with a long enough prefix: D is the blocker.
    ],
)
def test_fused_append_falls_back_for_unsupported_shapes(head_dim, past_length):
    torch.manual_seed(101)
    batch, hq, hkv = 1, 4, 2
    cache = PreallocatedKVCache(
        batch, hkv, past_length + 2, head_dim,
        device="cpu", dtype=torch.float32,
    )
    if past_length:
        cache.append(
            torch.randn(batch, hkv, past_length, head_dim),
            torch.randn(batch, hkv, past_length, head_dim),
            backend="torch",
        )

    q = torch.randn(batch, hq, 1, head_dim)
    new_k = torch.randn(batch, hkv, 1, head_dim)
    new_v = torch.randn_like(new_k)

    actual = cache.append_and_attend(
        q, new_k, new_v,
        append_backend="fused",
        attention_backend="sdpa",
    )

    k, v = cache.view()
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(hq // hkv, dim=1),
        v.repeat_interleave(hq // hkv, dim=1),
        is_causal=False,
    )

    assert cache.length == past_length + 1
    torch.testing.assert_close(actual, expected)
