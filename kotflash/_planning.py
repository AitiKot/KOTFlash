# SPDX-License-Identifier: Apache-2.0
"""Pure-Python launch planning and routing tables (no torch/triton imports)."""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

INT32_MAX = 2**31 - 1


def cdiv(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise ValueError("cdiv expects a >= 0 and b > 0")
    return -(-a // b)


class SplitKVPlan(NamedTuple):
    block_m: int
    block_n: int
    num_splits: int
    chunk_n: int
    blocks_per_split: int


def split_kv_plan(sq: int, sk: int, d: int, *, causal: bool) -> Optional[SplitKVPlan]:
    """Return the conservative short-query/long-cache Split-KV plan, else None."""
    if min(sq, sk, d) <= 0:
        raise ValueError("sq, sk and d must be positive")
    if causal or sq > 16 or sk < 1024 or d > 64:
        return None
    block_n = 64
    num_splits = min(16, cdiv(sk, 128))
    chunk_n = cdiv(sk, num_splits)
    return SplitKVPlan(16, block_n, num_splits, chunk_n, cdiv(chunk_n, block_n))


def check_int32_indexing(**numels: int) -> None:
    """Reject flat tensor offsets that cannot be represented by signed int32."""
    for name, n in numels.items():
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"{name} must be a non-negative integer element count")
        if n > INT32_MAX:
            raise ValueError(
                f"{name} has {n} elements; Triton kernels use 32-bit offsets "
                f"(limit {INT32_MAX}). Reduce batch/sequence size or use backend='sdpa'."
            )


_DECODE_PROFILE = {
    ((2, 8, 1, 64), (2, 2, 2048, 64)): "triton",
    ((2, 8, 4, 64), (2, 2, 1024, 64)): "sdpa_expanded",
    ((2, 8, 16, 64), (2, 2, 4096, 64)): "triton",
}
_CAUSAL_LENGTHS = {128, 256, 512, 1024, 2048}


def t4_profile_for_shapes(
    q_shape: Sequence[int], k_shape: Sequence[int], v_shape: Sequence[int], *,
    causal: bool, scale: float, contiguous: bool,
) -> Optional[str]:
    """Return a measured backend for an exact T4 shape, else None for SDPA."""
    q_shape, k_shape, v_shape = tuple(q_shape), tuple(k_shape), tuple(v_shape)
    if not contiguous or v_shape != k_shape:
        return None
    if not math.isclose(scale, 0.125, rel_tol=0.0, abs_tol=1e-12):
        return None
    if not causal:
        return _DECODE_PROFILE.get((q_shape, k_shape))
    if len(q_shape) != 4 or len(k_shape) != 4:
        return None
    seq = q_shape[2]
    if (q_shape == (2, 8, seq, 64)
            and k_shape == (2, 2, seq, 64)
            and seq in _CAUSAL_LENGTHS):
        return "triton" if seq >= 1024 else "sdpa_expanded"
    return None
