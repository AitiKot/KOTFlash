# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests; run these on an actual SM75/T4 runner."""
import pytest
import torch
import torch.nn.functional as F

from kotflash import attention, triton_turing_supported


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


def test_triton_backward_matches_sdpa_gqa():
    torch.manual_seed(12)
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 80, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    grad_seed = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)

    out = attention(q, k, v, causal=True, backend="triton")
    actual_grads = torch.autograd.grad(out, (q, k, v), grad_seed)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(
        q_ref, k_ref.repeat_interleave(4, dim=1),
        v_ref.repeat_interleave(4, dim=1), is_causal=True,
    )
    expected_grads = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_seed)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-2)
