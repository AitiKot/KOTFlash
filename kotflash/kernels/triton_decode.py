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
    def _strided_decode_append_split(
        Q, K_CACHE, V_CACHE, NEW_K, NEW_V, PART_O, PART_LSE,
        BATCH: tl.constexpr,
        SQ: tl.constexpr, SK_PAST: tl.constexpr, SK_TOTAL: tl.constexpr,
        HQ: tl.constexpr, HKV: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr, GROUP: tl.constexpr,
        QSB: tl.constexpr, QSH: tl.constexpr, QSS: tl.constexpr, QSD: tl.constexpr,
        KSB: tl.constexpr, KSH: tl.constexpr, KSS: tl.constexpr, KSD: tl.constexpr,
        VSB: tl.constexpr, VSH: tl.constexpr, VSS: tl.constexpr, VSD: tl.constexpr,
        NKSB: tl.constexpr, NKSH: tl.constexpr, NKSS: tl.constexpr, NKSD: tl.constexpr,
        NVSB: tl.constexpr, NVSH: tl.constexpr, NVSS: tl.constexpr, NVSD: tl.constexpr,
        CHUNK_N: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        """Append one new K/V position and compute its Split-KV partial attention.

        A unique (batch, KV-head) owner writes the new cache slot. Every query
        head reads NEW_K/NEW_V directly for the appended position, so there is
        no grid-wide synchronization or cross-CTA read-after-write dependency.
        """
        pid_split = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        kv_h = pid_h // GROUP

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

        new_k_ptrs = (
            NEW_K + pid_b * NKSB + kv_h * NKSH
            + offs_d[:, None] * NKSD
        )
        new_v_ptrs = (
            NEW_V + pid_b * NVSB + kv_h * NVSH
            + offs_d[None, :] * NVSD
        )
        new_k = tl.load(new_k_ptrs, mask=offs_d[:, None] < D, other=0.0)
        new_v = tl.load(new_v_ptrs, mask=offs_d[None, :] < D, other=0.0)

        # Exactly one CTA per (batch, KV head) performs the persistent append.
        if pid_split == 0 and pid_h % GROUP == 0:
            k_dst = (
                K_CACHE + pid_b * KSB + kv_h * KSH
                + SK_PAST * KSS + offs_d * KSD
            )
            v_dst = (
                V_CACHE + pid_b * VSB + kv_h * VSH
                + SK_PAST * VSS + offs_d * VSD
            )
            tl.store(k_dst, tl.reshape(new_k, (BD,)), mask=offs_d < D)
            tl.store(v_dst, tl.reshape(new_v, (BD,)), mask=offs_d < D)

        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BD), tl.float32)

        split_start = pid_split * CHUNK_N
        split_end = tl.minimum(split_start + CHUNK_N, SK_TOTAL)

        for block_idx in range(BLOCKS_PER_SPLIT):
            key_pos = split_start + block_idx * BN + offs_n
            k_ptrs = (
                K_CACHE + pid_b * KSB + kv_h * KSH
                + key_pos[None, :] * KSS + offs_d[:, None] * KSD
            )
            v_ptrs = (
                V_CACHE + pid_b * VSB + kv_h * VSH
                + key_pos[:, None] * VSS + offs_d[None, :] * VSD
            )
            old_mask_k = (
                (key_pos[None, :] < SK_PAST)
                & (key_pos[None, :] < split_end)
                & (offs_d[:, None] < D)
            )
            old_mask_v = (
                (key_pos[:, None] < SK_PAST)
                & (key_pos[:, None] < split_end)
                & (offs_d[None, :] < D)
            )
            k_old = tl.load(k_ptrs, mask=old_mask_k, other=0.0)
            v_old = tl.load(v_ptrs, mask=old_mask_v, other=0.0)

            is_new_k = key_pos[None, :] == SK_PAST
            is_new_v = key_pos[:, None] == SK_PAST
            k = tl.where(is_new_k, new_k, k_old)
            v = tl.where(is_new_v, new_v, v_old)

            scores = tl.dot(q, k) * SCALE
            valid = (
                (offs_m[:, None] < SQ)
                & (key_pos[None, :] < split_end)
                & (key_pos[None, :] < SK_TOTAL)
            )
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




def attention_forward_strided_append(
    q, k_cache, v_cache, new_k, new_v, *, past_length: int, scale: float
):
    """Fuse a one-token cache append with Split-KV partial attention.

    This is inference-only and uses two launches total: one combined
    append+partial-attention kernel and one partial-result reduction kernel.
    """
    if triton is None:
        raise RuntimeError("Triton is required for fused append/decode")
    import torch

    if not all(t.is_cuda for t in (q, k_cache, v_cache, new_k, new_v)):
        raise ValueError("fused append/decode requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device) != (7, 5):
        raise ValueError("fused append/decode currently requires NVIDIA SM75")
    if any(t.dtype != torch.float16 for t in (q, k_cache, v_cache, new_k, new_v)):
        raise TypeError("fused append/decode currently supports FP16 only")
    if q.ndim != 4 or q.shape[2] != 1 or not q.is_contiguous():
        raise ValueError("q must be contiguous [B,Hq,1,D]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V cache must have matching [B,Hkv,capacity,D] shapes")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("destination K/V cache buffers must be contiguous")
    if new_k.ndim != 4 or new_v.shape != new_k.shape:
        raise ValueError("new K/V must have matching [B,Hkv,1,D] shapes")
    if new_k.shape[2] != 1:
        raise ValueError("fused append/decode accepts exactly one new K/V token")

    b, hq, sq, d = q.shape
    _, hkv, capacity, cache_d = k_cache.shape
    if (sq != 1 or cache_d != d or new_k.shape != (b, hkv, 1, d)
            or new_v.shape != new_k.shape):
        raise ValueError("query, cache and new K/V shapes are incompatible")
    if hq % hkv or not 16 <= d <= 64 or past_length < 1023 or past_length + 1 > capacity:
        raise ValueError("unsupported head grouping, head_dim, cache length or capacity")
    if any(t.device != q.device for t in (k_cache, v_cache, new_k, new_v)):
        raise ValueError("all tensors must be on the same CUDA device")

    plan = split_kv_plan(sq, past_length + 1, d, causal=False)
    if plan is None:
        raise ValueError("shape is not eligible for Split-KV decode")

    def span(t):
        return (
            (t.shape[0] - 1) * t.stride(0)
            + (t.shape[1] - 1) * t.stride(1)
            + (t.shape[2] - 1) * t.stride(2)
            + (t.shape[3] - 1) * t.stride(3) + 1
        )

    check_int32_indexing(
        q=q.numel(), k=k_cache.numel(), v=v_cache.numel(),
        new_k=span(new_k), new_v=span(new_v),
        partial_o=plan.num_splits * b * hq * sq * d,
        partial_lse=plan.num_splits * b * hq * sq,
    )

    out = q.new_empty((b, hq, sq, d))
    partial_o = torch.empty(
        (plan.num_splits, b, hq, sq, d), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (plan.num_splits, b, hq, sq), device=q.device, dtype=torch.float32
    )
    _strided_decode_append_split[(plan.num_splits, b, hq)](
        q, k_cache, v_cache, new_k, new_v, partial_o, partial_lse,
        b, sq, past_length, past_length + 1, hq, hkv, d, scale, hq // hkv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        new_k.stride(0), new_k.stride(1), new_k.stride(2), new_k.stride(3),
        new_v.stride(0), new_v.stride(1), new_v.stride(2), new_v.stride(3),
        plan.chunk_n, plan.blocks_per_split,
        plan.block_m, plan.block_n, triton.next_power_of_2(d),
        num_warps=4, num_stages=1,
    )
    _strided_decode_reduce[(b, hq)](
        partial_o, partial_lse, out,
        b, sq, hq, d, plan.num_splits,
        triton.next_power_of_2(plan.num_splits),
        plan.block_m, triton.next_power_of_2(d),
        num_warps=4, num_stages=1,
    )
    return out


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
