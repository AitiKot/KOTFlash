# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
import torch.nn.functional as F

from kotflash import attention
from kotflash.attention import _blockwise_backward


def test_sdpa_matches_torch_mha():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 7, 32)
    k = torch.randn(2, 4, 9, 32)
    v = torch.randn_like(k)
    actual = attention(q, k, v, backend="sdpa")
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_matches_torch_gqa_with_unequal_sequence_lengths(causal):
    torch.manual_seed(1)
    q = torch.randn(2, 8, 6, 32)
    k = torch.randn(2, 2, 9, 32)
    v = torch.randn_like(k)
    actual = attention(q, k, v, causal=causal, backend="sdpa")
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
        is_causal=causal,
    )
    torch.testing.assert_close(actual, expected)


def test_custom_scale_matches_explicit_q_scaling():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 5, 16)
    k = torch.randn(1, 2, 7, 16)
    v = torch.randn_like(k)
    scale = 0.125
    actual = attention(q, k, v, scale=scale, backend="sdpa")
    expected = F.scaled_dot_product_attention(q * (scale * math.sqrt(16)), k, v)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_scale(bad_scale):
    q = torch.randn(1, 1, 2, 16)
    with pytest.raises(ValueError, match="finite"):
        attention(q, q, q, scale=bad_scale, backend="sdpa")


def test_rejects_invalid_head_grouping():
    q = torch.randn(1, 3, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    with pytest.raises(ValueError, match="divisible"):
        attention(q, k, k, backend="sdpa")


def test_blockwise_backward_matches_autograd_mha_and_gqa():
    torch.manual_seed(3)
    for hq, hkv in [(2, 2), (4, 2)]:
        q = torch.randn(1, hq, 5, 16, requires_grad=True)
        k = torch.randn(1, hkv, 7, 16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        scale = 1.0 / math.sqrt(16)
        expanded_k = k.repeat_interleave(hq // hkv, dim=1)
        expanded_v = v.repeat_interleave(hq // hkv, dim=1)
        scores = torch.matmul(q, expanded_k.transpose(-2, -1)) * scale
        lse = torch.logsumexp(scores, dim=-1)
        out = F.scaled_dot_product_attention(q, expanded_k, expanded_v)
        grad_out = torch.randn_like(out)
        ref_grads = torch.autograd.grad(out, (q, k, v), grad_out)
        got_grads = _blockwise_backward(
            q.detach(), k.detach(), v.detach(), out.detach(), lse.detach(), grad_out,
            causal=False, scale=scale, q_block=3, kv_block=2,
        )
        for got, expected in zip(got_grads, ref_grads):
            torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-5)


def test_blockwise_backward_causal_matches_autograd():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 5, 16, requires_grad=True)
    k = torch.randn(1, 2, 7, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    scale = 1.0 / math.sqrt(16)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.arange(7)[None, :] > torch.arange(5)[:, None]
    scores = scores.masked_fill(mask[None, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    grad_out = torch.randn_like(out)
    ref_grads = torch.autograd.grad(out, (q, k, v), grad_out)
    got_grads = _blockwise_backward(
        q.detach(), k.detach(), v.detach(), out.detach(), lse.detach(), grad_out,
        causal=True, scale=scale, q_block=3, kv_block=2,
    )
    for got, expected in zip(got_grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-5)
