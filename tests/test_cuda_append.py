# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration tests for the optional custom CUDA K/V append operator."""
import os

import pytest
import torch

from kotflash.cache import PreallocatedKVCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("KOTFLASH_TEST_CUDA_APPEND") != "1",
    reason="set KOTFLASH_TEST_CUDA_APPEND=1 on a CUDA toolkit host to compile/test the extension",
)


def test_cuda_append_copies_k_and_v_with_one_extension_call():
    torch.manual_seed(37)
    cache = PreallocatedKVCache(
        2, 2, 64, 32, device="cuda", dtype=torch.float16
    )
    past_k = torch.randn(2, 2, 7, 32, device="cuda", dtype=torch.float16)
    past_v = torch.randn_like(past_k)
    new_k = torch.randn(2, 2, 3, 32, device="cuda", dtype=torch.float16)
    new_v = torch.randn_like(new_k)

    cache.append(past_k, past_v, backend="cuda")
    cache.append(new_k, new_v, backend="cuda")
    actual_k, actual_v = cache.view()

    torch.testing.assert_close(actual_k, torch.cat((past_k, new_k), dim=2))
    torch.testing.assert_close(actual_v, torch.cat((past_v, new_v), dim=2))
    assert cache.length == 10
