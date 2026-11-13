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


@pytest.mark.parametrize(
    ("q_seq", "kv_seq"),
    [(1, 2048), (4, 1024), (16, 4096)],
)
def test_triton_split_kv_decode_matches_sdpa(q_seq, kv_seq):
    """Long-cache decode: all keys are already visible to the short Q chunk."""
    torch.manual_seed(17)
    q = torch.randn(2, 8, q_seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, kv_seq, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    actual = attention(q, k, v, causal=False, backend="triton")
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        is_causal=False,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize(
    ("q_seq", "kv_seq"),
    [(2, 1537), (8, 8193)],
)
def test_triton_split_kv_decode_handles_partial_chunks(q_seq, kv_seq):
    """Exercise non-divisible KV splits and a cache longer than 8k tokens."""
    torch.manual_seed(23)
    q = torch.randn(1, 4, q_seq, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, kv_seq, 32, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    actual = attention(q, k, v, causal=False, backend="triton")
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
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


@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "expected_backend"),
    [
        (1, 2048, "triton"),
        (4, 1024, "sdpa_expanded"),
        (16, 4096, "triton"),
    ],
)
def test_t4_auto_decode_profile_selects_and_matches_sdpa(
    q_seq, kv_seq, expected_backend
):
    torch.manual_seed(19)
    q = torch.randn(2, 8, q_seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, kv_seq, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    selected = _t4_profile_backend(q, k, v, causal=False, scale=0.125)
    assert selected == expected_backend

    actual = attention(q, k, v, causal=False, backend="auto")
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1), is_causal=False,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


def test_t4_auto_profile_rejects_unprofiled_cases():
    q = torch.empty(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(2, 2, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)

    assert _t4_profile_backend(q, k, v, causal=False, scale=0.125) is None

    # Decode profiles are intentionally exact; nearby lengths must fall back.
    q_decode = torch.empty(2, 8, 4, 64, device="cuda", dtype=torch.float16)
    k_decode = torch.empty(2, 2, 1025, 64, device="cuda", dtype=torch.float16)
    assert _t4_profile_backend(
        q_decode, k_decode, torch.empty_like(k_decode),
        causal=False, scale=0.125,
    ) is None
    assert _t4_profile_backend(q, k, v, causal=True, scale=0.1) is None
    assert _t4_profile_backend(q[:, :, :-1], k[:, :, :-1], v[:, :, :-1],
                               causal=True, scale=0.125) is None

    # Non-contiguous tensors were not part of the measured profile.
    q_nc = torch.empty(2, 8, 512, 128, device="cuda", dtype=torch.float16)[..., ::2]
    k_nc = torch.empty(2, 2, 512, 128, device="cuda", dtype=torch.float16)[..., ::2]
    v_nc = torch.empty_like(k_nc)
    assert _t4_profile_backend(q_nc, k_nc, v_nc, causal=True, scale=0.125) is None


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
        (8, 2, 1, 1024, 64, False),  # Single-token decode with long KV cache
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


def test_strided_preallocated_cache_decode_reads_prefix_without_contiguous_copy():
    from kotflash.cache import PreallocatedKVCache

    torch.manual_seed(29)
    cache = PreallocatedKVCache(
        2, 2, 4096, 64, device="cuda", dtype=torch.float16
    )
    past_k = torch.randn(2, 2, 2047, 64, device="cuda", dtype=torch.float16)
    past_v = torch.randn_like(past_k)
    new_k = torch.randn(2, 2, 1, 64, device="cuda", dtype=torch.float16)
    new_v = torch.randn_like(new_k)
    q = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.float16)

    cache.append(past_k, past_v)
    with torch.inference_mode():
        actual = cache.append_and_attend(
            q, new_k, new_v,
            append_backend="torch", attention_backend="triton",
        )
        k_view, v_view = cache.view()
        assert not k_view.is_contiguous() and not v_view.is_contiguous()
        expected = F.scaled_dot_product_attention(
            q,
            k_view.contiguous().repeat_interleave(4, dim=1),
            v_view.contiguous().repeat_interleave(4, dim=1),
            is_causal=False,
        )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


def test_auto_routes_exact_t4_preallocated_cache_decode():
    from kotflash.cache import PreallocatedKVCache

    torch.manual_seed(31)
    cache = PreallocatedKVCache(
        2, 2, 4096, 64, device="cuda", dtype=torch.float16
    )
    k = torch.randn(2, 2, 2048, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    q = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.float16)
    cache.append(k, v)
    with torch.inference_mode():
        actual = cache.attend(q, backend="auto")
        expected = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(4, dim=1),
            v.repeat_interleave(4, dim=1),
            is_causal=False,
        )
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


def test_fused_preallocated_append_and_attention_matches_expanded_sdpa():
    from kotflash.cache import PreallocatedKVCache

    torch.manual_seed(41)
    cache = PreallocatedKVCache(
        2, 2, 4096, 64, device="cuda", dtype=torch.float16
    )
    past_k = torch.randn(2, 2, 2047, 64, device="cuda", dtype=torch.float16)
    past_v = torch.randn_like(past_k)
    new_k = torch.randn(2, 2, 1, 64, device="cuda", dtype=torch.float16)
    new_v = torch.randn_like(new_k)
    q = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.float16)
    cache.append(past_k, past_v, backend="torch")

    with torch.inference_mode():
        actual = cache.append_and_attend(
            q, new_k, new_v,
            append_backend="fused", attention_backend="triton",
        )
        expected = F.scaled_dot_product_attention(
            q,
            torch.cat((past_k, new_k), dim=2).repeat_interleave(4, dim=1),
            torch.cat((past_v, new_v), dim=2).repeat_interleave(4, dim=1),
            is_causal=False,
        )
        torch.cuda.synchronize()
        saved_k, saved_v = cache.view()

    assert cache.length == 2048
    torch.testing.assert_close(saved_k[:, :, -1:], new_k)
    torch.testing.assert_close(saved_v[:, :, -1:], new_v)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


def test_auto_fused_preallocated_append_matches_expanded_sdpa_on_measured_t4():
    from kotflash.cache import PreallocatedKVCache

    torch.manual_seed(43)
    cache = PreallocatedKVCache(
        2, 2, 4096, 64, device="cuda", dtype=torch.float16
    )
    past_k = torch.randn(2, 2, 2047, 64, device="cuda", dtype=torch.float16)
    past_v = torch.randn_like(past_k)
    new_k = torch.randn(2, 2, 1, 64, device="cuda", dtype=torch.float16)
    new_v = torch.randn_like(new_k)
    q = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.float16)
    cache.append(past_k, past_v, backend="torch")

    with torch.inference_mode():
        actual = cache.append_and_attend(q, new_k, new_v, append_backend="auto")
        expected = F.scaled_dot_product_attention(
            q,
            torch.cat((past_k, new_k), dim=2).repeat_interleave(4, dim=1),
            torch.cat((past_v, new_v), dim=2).repeat_interleave(4, dim=1),
            is_causal=False,
        )
    assert cache.length == 2048
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)
