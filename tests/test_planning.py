# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for launch planning and measured auto-routing rules."""
import pytest

from kotflash._planning import (
    INT32_MAX, check_int32_indexing, split_kv_plan, t4_profile_for_shapes,
)


def route(q, k, *, causal=True, scale=0.125, contiguous=True, v=None):
    return t4_profile_for_shapes(
        q, k, k if v is None else v, causal=causal,
        scale=scale, contiguous=contiguous,
    )


@pytest.mark.parametrize("seq,expected", [
    (128, "sdpa_expanded"), (256, "sdpa_expanded"), (512, "sdpa_expanded"),
    (1024, "triton"), (2048, "triton"),
])
def test_causal_profile(seq, expected):
    assert route((2, 8, seq, 64), (2, 2, seq, 64)) == expected


@pytest.mark.parametrize("q,k,expected", [
    ((2, 8, 1, 64), (2, 2, 2048, 64), "triton"),
    ((2, 8, 4, 64), (2, 2, 1024, 64), "sdpa_expanded"),
    ((2, 8, 16, 64), (2, 2, 4096, 64), "triton"),
])
def test_decode_profile(q, k, expected):
    assert route(q, k, causal=False) == expected


@pytest.mark.parametrize("q,k,kwargs", [
    ((2, 8, 512, 64), (2, 2, 512, 64), {"causal": False}),
    ((2, 8, 4, 64), (2, 2, 1025, 64), {"causal": False}),
    ((2, 8, 2, 64), (2, 2, 2048, 64), {"causal": False}),
    ((2, 8, 1, 64), (2, 2, 4096, 64), {"causal": False}),
    ((2, 8, 512, 64), (2, 2, 512, 64), {"scale": 0.1}),
    ((2, 8, 512, 64), (2, 2, 512, 64), {"contiguous": False}),
    ((2, 8, 511, 64), (2, 2, 511, 64), {}),
    ((2, 8, 512, 128), (2, 2, 512, 128), {}),
    ((1, 8, 512, 64), (1, 2, 512, 64), {}),
    ((2, 16, 512, 64), (2, 4, 512, 64), {}),
    ((2, 8, 4096, 64), (2, 2, 4096, 64), {}),
    ((2, 8, 512, 64), (2, 2, 256, 64), {}),
    ((8, 512, 64), (2, 512, 64), {}),
])
def test_unprofiled_shapes_fall_back(q, k, kwargs):
    assert route(q, k, **kwargs) is None


def test_mismatched_v_shape_falls_back():
    assert route((2, 8, 512, 64), (2, 2, 512, 64),
                 v=(2, 2, 256, 64)) is None


@pytest.mark.parametrize("sq,sk,d,causal", [
    (1, 1023, 64, False), (17, 4096, 64, False),
    (1, 4096, 72, False), (1, 4096, 64, True),
])
def test_split_kv_envelope(sq, sk, d, causal):
    assert split_kv_plan(sq, sk, d, causal=causal) is None


def test_split_kv_partitions_without_gaps_or_empty_splits():
    lengths = list(range(1024, 3000, 7)) + [4096, 8191, 8192, 16384, 20000]
    for sk in lengths:
        plan = split_kv_plan(1, sk, 64, causal=False)
        assert plan is not None and 1 <= plan.num_splits <= 16
        covered = 0
        for split in range(plan.num_splits):
            start = split * plan.chunk_n
            end = min(start + plan.chunk_n, sk)
            assert start == covered
            assert start < end
            assert start + plan.blocks_per_split * plan.block_n >= end
            covered = end
        assert covered == sk


def test_int32_index_guard():
    check_int32_indexing(q=INT32_MAX)
    with pytest.raises(ValueError, match="32-bit"):
        check_int32_indexing(q=INT32_MAX + 1)
