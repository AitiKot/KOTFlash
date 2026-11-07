# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests; run these on an actual SM75/T4 runner."""
import pytest
import torch
import torch.nn.functional as F

from kotflash import attention, triton_turing_supported
from kotflash.attention import _t4_profile_backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not triton_turing_supported()
    or torch.cuda.get_device_capability() != (7, 5),
    reason="requires CUDA, upstream Triton 3.2.x and NVIDIA Turing SM75 (e.g. T4)",
)


@pytest.mark.parametrize("head_dim", [32, 64, 72, 96, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_forward_matches_sdpa(head_dim, causal):
    torch.manual_seed(11)
    q = torch.randn(1, 8, 129, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 161, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    actual = attention(q, k, v, causal=causal, backend="triton")
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
        is_causal=causal,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("seq", [128, 256, 512, 1024, 2048])
def test_t4_auto_profile_matches_expanded_sdpa(seq):
    torch.manual_seed(13)
    q = torch.randn(2, 8, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, seq, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    actual = attention(q, k, v, causal=True, backend="auto")
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize(
    ("seq", "expected_backend"),
    [
        (128, "sdpa_expanded"),
        (256, "sdpa_expanded"),
        (512, "sdpa_expanded"),
        (1024, "triton"),
        (2048, "triton"),
    ],
)
def test_t4_auto_profile_selects_measured_backend(seq, expected_backend):
    q = torch.empty(2, 8, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(2, 2, seq, 64, device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)

    selected = _t4_profile_backend(q, k, v, causal=True, scale=0.125)
    assert selected == expected_backend


def test_t4_auto_profile_rejects_unprofiled_cases():
    q = torch.empty(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(2, 2, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)

    assert _t4_profile_backend(q, k, v, causal=False, scale=0.125) is None
    assert _t4_profile_backend(q, k, v, causal=True, scale=0.1) is None
    assert _t4_profile_backend(q[:, :, :-1], k[:, :, :-1], v[:, :, :-1],
                               causal=True, scale=0.125) is None


@pytest.mark.parametrize(
    ("hq", "hkv", "sq", "sk", "head_dim", "causal"),
    [
        (4, 4, 64, 80, 32, False),  # MHA, non-causal
        (8, 2, 64, 80, 64, True),   # GQA, causal
        (8, 1, 80, 64, 32, True),   # MQA, causal, Q longer than K
        (8, 2, 48, 64, 72, False),  # GQA, split head dimension
        (4, 4, 48, 64, 96, True),   # MHA, split head dimension
        (8, 1, 64, 48, 128, False), # MQA, largest supported head dimension
        (8, 2, 1024, 1024, 64, True), # Long causal GQA: split K/V reduction
        (4, 1, 1024, 768, 32, False), # Long unequal-length MQA: split reduction
        (4, 2, 2048, 2048, 32, True), # Long sequence: maximum split count
    ],
)
def test_triton_backward_matches_sdpa(
    hq, hkv, sq, sk, head_dim, causal
):
    torch.manual_seed(12)
    q = torch.randn(
        1, hq, sq, head_dim, device="cuda", dtype=torch.float16,
        requires_grad=True,
    )
    k = torch.randn(
        1, hkv, sk, head_dim, device="cuda", dtype=torch.float16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    grad_seed = torch.randn(
        1, hq, sq, head_dim, device="cuda", dtype=torch.float16,
    )

    out = attention(q, k, v, causal=causal, backend="triton")
    actual_grads = torch.autograd.grad(out, (q, k, v), grad_seed)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    groups = hq // hkv
    out_ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref.repeat_interleave(groups, dim=1),
        v_ref.repeat_interleave(groups, dim=1),
        is_causal=causal,
    )
    expected_grads = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), grad_seed
    )
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-2)
