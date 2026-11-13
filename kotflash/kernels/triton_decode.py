# SPDX-License-Identifier: Apache-2.0
"""Stride-aware inference-only Split-KV decode for preallocated KV caches."""
from __future__ import annotations

from .._planning import check_int32_indexing, split_kv_plan

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _strided_decode_split(
        Q, K, V, PART_O, PART_LSE,
        BATCH: tl.constexpr,
        SQ: tl.constexpr, SK: tl.constexpr,
        HQ: tl.constexpr, HKV: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr,
        QSB: tl.constexpr, QSH: tl.constexpr, QSS: tl.constexpr, QSD: tl.constexpr,
        KSB: tl.constexpr, KSH: tl.constexpr, KSS: tl.constexpr, KSD: tl.constexpr,
        VSB: tl.constexpr, VSH: tl.constexpr, VSS: tl.constexpr, VSD: tl.constexpr,
        CHUNK_N: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        """One CTA computes one (split, batch, query-head) partial result."""
        pid_split = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        kv_h = pid_h // (HQ // HKV)

        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, BD)

        q_ptrs = (
            Q + pid_b * QSB + pid_h * QSH
            + offs_m[:, None] * QSS + offs_d[None, :] * QSD
        )
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < SQ) & (offs_d[None, :] < D),
            other=0.0,
        )

        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BD), tl.float32)

        split_start = pid_split * CHUNK_N
        split_end = tl.minimum(split_start + CHUNK_N, SK)

        for block_idx in range(BLOCKS_PER_SPLIT):
            key_pos = split_start + block_idx * BN + offs_n
            k_ptrs = (
                K + pid_b * KSB + kv_h * KSH
                + key_pos[None, :] * KSS + offs_d[:, None] * KSD
            )
            v_ptrs = (
                V + pid_b * VSB + kv_h * VSH
                + key_pos[:, None] * VSS + offs_d[None, :] * VSD
            )
            k = tl.load(
                k_ptrs,
                mask=(key_pos[None, :] < split_end) & (offs_d[:, None] < D),
                other=0.0,
            )
            v = tl.load(
                v_ptrs,
                mask=(key_pos[:, None] < split_end) & (offs_d[None, :] < D),
                other=0.0,
            )

            scores = tl.dot(q, k) * SCALE
            valid = (offs_m[:, None] < SQ) & (key_pos[None, :] < split_end)
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
            p = tl.where(valid, tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        safe_l = tl.maximum(l_i, 1.0)
        partial_o = tl.where(l_i[:, None] > 0.0, acc / safe_l[:, None], 0.0)
        partial_lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))

        partial_base = ((pid_split * BATCH + pid_b) * HQ + pid_h) * SQ * D
        tl.store(
            PART_O + partial_base + offs_m[:, None] * D + offs_d[None, :],
            partial_o,
            mask=(offs_m[:, None] < SQ) & (offs_d[None, :] < D),
        )
        partial_lse_base = ((pid_split * BATCH + pid_b) * HQ + pid_h) * SQ
        tl.store(
            PART_LSE + partial_lse_base + offs_m,
            partial_lse,
            mask=offs_m < SQ,
        )


    @triton.jit
    def _strided_decode_reduce(
        PART_O, PART_LSE, O,
        BATCH: tl.constexpr,
        SQ: tl.constexpr, HQ: tl.constexpr,
        D: tl.constexpr, NUM_SPLITS: tl.constexpr,
        SPLIT_PAD: tl.constexpr,
        BM: tl.constexpr, BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        offs_s = tl.arange(0, SPLIT_PAD)

        lse_ptrs = (
            PART_LSE
            + (offs_s[:, None] * BATCH * HQ + pid_b * HQ + pid_h) * SQ
            + offs_m[None, :]
        )
        lse_all = tl.load(
            lse_ptrs,
            mask=(offs_s[:, None] < NUM_SPLITS) & (offs_m[None, :] < SQ),
            other=float("-inf"),
        )
        row_max = tl.max(lse_all, axis=0)
        weights = tl.exp(lse_all - row_max[None, :])
        weights = tl.where(
            (offs_s[:, None] < NUM_SPLITS) & (offs_m[None, :] < SQ),
            weights,
            0.0,
        )
        denom = tl.sum(weights, axis=0)
        safe_max = tl.where(denom > 0.0, row_max, 0.0)
        safe_denom = tl.maximum(denom, 1.0)

        acc = tl.zeros((BM, BD), tl.float32)
        for split_idx in range(NUM_SPLITS):
            lse_s = tl.load(
                PART_LSE + (split_idx * BATCH * HQ + pid_b * HQ + pid_h) * SQ + offs_m,
                mask=offs_m < SQ,
                other=float("-inf"),
            )
            weight = tl.exp(lse_s - safe_max) / safe_denom
            part_base = ((split_idx * BATCH + pid_b) * HQ + pid_h) * SQ * D
            part = tl.load(
                PART_O + part_base + offs_m[:, None] * D + offs_d[None, :],
                mask=(offs_m[:, None] < SQ) & (offs_d[None, :] < D),
                other=0.0,
            )
            acc += weight[:, None] * part

        tl.store(
            O + ((pid_b * HQ + pid_h) * SQ + offs_m[:, None]) * D + offs_d[None, :],
            acc,
            mask=(offs_m[:, None] < SQ) & (offs_d[None, :] < D),
        )


def strided_decode_supported(q, k, v) -> bool:
    """Check the inner-contiguous padded-prefix layout emitted by PreallocatedKVCache."""
    if triton is None or not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype != __import__("torch").float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        return False
    if q.shape[2] > 16 or k.shape[2] < 1024 or not 16 <= q.shape[-1] <= 64:
        return False
    if q.shape[0] != k.shape[0] or q.shape[1] % k.shape[1]:
        return False
    d, sk, hkv = q.shape[-1], k.shape[2], k.shape[1]
    if not q.is_contiguous():
        return False
    for tensor in (k, v):
        if tensor.stride(3) != 1 or tensor.stride(2) != d:
            return False
        if tensor.stride(1) < sk * d or tensor.stride(0) < hkv * tensor.stride(1):
            return False
    return k.stride() == v.stride()


def attention_forward_strided_decode(q, k, v, *, scale: float):
    """Inference-only Split-KV forward which reads padded cache views in-place."""
    if triton is None:
        raise RuntimeError("Triton is not installed; install a Turing-compatible Triton 3.2.x build.")
    import torch

    if not strided_decode_supported(q, k, v):
        raise ValueError("expected FP16 CUDA tensors with a short query and an inner-contiguous padded KV prefix")
    if torch.cuda.get_device_capability(q.device) != (7, 5):
        raise ValueError("strided KV-cache decode currently requires NVIDIA SM75")
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    plan = split_kv_plan(sq, sk, d, causal=False)
    if plan is None:
        raise ValueError("shape is not eligible for Split-KV decode")

    k_span = (
        (b - 1) * k.stride(0) + (hkv - 1) * k.stride(1)
        + (sk - 1) * k.stride(2) + (d - 1) * k.stride(3) + 1
    )
    v_span = (
        (b - 1) * v.stride(0) + (hkv - 1) * v.stride(1)
        + (sk - 1) * v.stride(2) + (d - 1) * v.stride(3) + 1
    )
    check_int32_indexing(
        q=q.numel(),
        k=k.numel() if k.is_contiguous() else k_span,
        v=v.numel() if v.is_contiguous() else v_span,
    )
    out = q.new_empty((b, hq, sq, d))
    partial_o = torch.empty(
        (plan.num_splits, b, hq, sq, d), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (plan.num_splits, b, hq, sq), device=q.device, dtype=torch.float32
    )

    _strided_decode_split[(plan.num_splits, b, hq)](
        q, k, v, partial_o, partial_lse,
        b, sq, sk, hq, hkv, d, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        plan.chunk_n, plan.blocks_per_split,
        plan.block_m, plan.block_n, triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )
    _strided_decode_reduce[(b, hq)](
        partial_o, partial_lse, out,
        b, sq, hq, d, plan.num_splits,
        triton.next_power_of_2(plan.num_splits),
        plan.block_m, triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )
    return out
