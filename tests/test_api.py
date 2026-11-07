import pytest
import torch

from kotflash import attention


def test_sdpa_matches_torch_mha():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 17, 32, dtype=torch.float32)
    k = torch.randn(2, 4, 19, 32, dtype=torch.float32)
    v = torch.randn_like(k)
    got = attention(q, k, v, backend="sdpa")
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(got, expected)


def test_sdpa_gqa_matches_expanded_reference():
    torch.manual_seed(1)
    q = torch.randn(1, 8, 11, 16)
    k = torch.randn(1, 2, 13, 16)
    v = torch.randn_like(k)
    got = attention(q, k, v, backend="sdpa")
    repeat = q.shape[1] // k.shape[1]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    )
    torch.testing.assert_close(got, expected)


def test_rejects_invalid_layout():
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 2, 7, 8)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="divisible"):
        attention(q, k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_forward_and_grad_if_available():
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability()[0:2] != (7, 5):
        pytest.skip("This test is targeted at Turing SM75")
    torch.manual_seed(2)
    q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    got = attention(q, k, v, causal=True, backend="triton")
    exp_k = k.repeat_interleave(2, dim=1)
    exp_v = v.repeat_interleave(2, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(q, exp_k, exp_v, is_causal=True)
    torch.testing.assert_close(got, expected, atol=3e-2, rtol=3e-2)
    got.float().square().mean().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
