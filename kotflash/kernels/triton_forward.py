# SPDX-License-Identifier: Apache-2.0
"""Experimental Triton online-softmax attention kernel for NVIDIA SM75.

This is a first-party implementation using public Triton APIs. It does not
vendor or copy CUDA/Triton kernels from other attention projects.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _attention_fwd_kernel(
        Q, K, V, O, LSE,
        Sq: tl.constexpr, Sk: tl.constexpr,
        Hq: tl.constexpr, Hkv: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
        DTAIL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        kv_h = pid_h // (Hq // Hkv)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        q_base = ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D

        # Split D=72/96 into 64 + tail instead of padding the full dot to 128.
        if D > 64 and D < 128:
            offs_d0 = tl.arange(0, 64)
            offs_dt = tl.arange(0, DTAIL)
            q0 = tl.load(
                Q + q_base + offs_d0[None, :],
                mask=(offs_m[:, None] < Sq) & (offs_d0[None, :] < D),
                other=0.0,
            )
            qt = tl.load(
                Q + q_base + 64 + offs_dt[None, :],
                mask=(offs_m[:, None] < Sq) & (64 + offs_dt[None, :] < D),
                other=0.0,
            )
            acc0 = tl.zeros((BM, 64), tl.float32)
            acct = tl.zeros((BM, DTAIL), tl.float32)
        else:
            offs_d = tl.arange(0, BD)
            q = tl.load(
                Q + q_base + offs_d[None, :],
                mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
                other=0.0,
            )
            acc = tl.zeros((BM, BD), tl.float32)

        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)

        # Skip full future K/V tiles for causal attention. Clamp to Sq as well
        # as Sk to preserve upper-left causal semantics for unequal lengths.
        if CAUSAL:
            max_key_exclusive = tl.minimum(
                tl.minimum(Sk, Sq),
                (pid_m + 1) * BM,
            )
            end_n = tl.cdiv(max_key_exclusive, BN)
        else:
            end_n = tl.cdiv(Sk, BN)

        for start_n in range(end_n):
            block_n = start_n * BN + offs_n
            k_base = ((pid_b * Hkv + kv_h) * Sk + block_n[None, :]) * D
            v_base = ((pid_b * Hkv + kv_h) * Sk + block_n[:, None]) * D

            if D > 64 and D < 128:
                k0 = tl.load(
                    K + k_base + offs_d0[:, None],
                    mask=(block_n[None, :] < Sk) & (offs_d0[:, None] < D),
                    other=0.0,
                )
                kt = tl.load(
                    K + k_base + 64 + offs_dt[:, None],
                    mask=(block_n[None, :] < Sk) & (64 + offs_dt[:, None] < D),
                    other=0.0,
                )
                v0 = tl.load(
                    V + v_base + offs_d0[None, :],
                    mask=(block_n[:, None] < Sk) & (offs_d0[None, :] < D),
                    other=0.0,
                )
                vt = tl.load(
                    V + v_base + 64 + offs_dt[None, :],
                    mask=(block_n[:, None] < Sk) & (64 + offs_dt[None, :] < D),
                    other=0.0,
                )
                scores = (tl.dot(q0, k0) + tl.dot(qt, kt)) * SCALE
            else:
                k = tl.load(
                    K + k_base + offs_d[:, None],
                    mask=(block_n[None, :] < Sk) & (offs_d[:, None] < D),
                    other=0.0,
                )
                v = tl.load(
                    V + v_base + offs_d[None, :],
                    mask=(block_n[:, None] < Sk) & (offs_d[None, :] < D),
                    other=0.0,
                )
                scores = tl.dot(q, k) * SCALE

            valid = (offs_m[:, None] < Sq) & (block_n[None, :] < Sk)
            if CAUSAL:
                valid = valid & (block_n[None, :] <= offs_m[:, None])
            scores = tl.where(valid, scores, float("-inf"))

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            if D > 64 and D < 128:
                p_cast = p.to(v0.dtype)
                acc0 = acc0 * alpha[:, None] + tl.dot(p_cast, v0)
                acct = acct * alpha[:, None] + tl.dot(p_cast, vt)
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

            m_i = m_new

        if D > 64 and D < 128:
            out0 = acc0 / l_i[:, None]
            outt = acct / l_i[:, None]
            tl.store(
                O + q_base + offs_d0[None, :],
                out0,
                mask=(offs_m[:, None] < Sq) & (offs_d0[None, :] < D),
            )
            tl.store(
                O + q_base + 64 + offs_dt[None, :],
                outt,
                mask=(offs_m[:, None] < Sq) & (64 + offs_dt[None, :] < D),
            )
        else:
            out = acc / l_i[:, None]
            tl.store(
                O + q_base + offs_d[None, :],
                out,
                mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
            )

        lse = m_i + tl.log(l_i)
        tl.store(
            LSE + (pid_b * Hq + pid_h) * Sq + offs_m,
            lse,
            mask=offs_m < Sq,
        )



    @triton.jit
    def _attention_fwd_split_kv(
        Q, K, V, PART_O, PART_LSE,
        BATCH: tl.constexpr,
        Sq: tl.constexpr, Sk: tl.constexpr,
        Hq: tl.constexpr, Hkv: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr,
        CHUNK_N: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        """Parallel split-KV forward for short-query, long-cache decoding."""
        pid_split = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        kv_h = pid_h // (Hq // Hkv)

        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, BD)

        q_base = ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D
        q = tl.load(
            Q + q_base + offs_d[None, :],
            mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
            other=0.0,
        )

        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BD), tl.float32)

        split_start = pid_split * CHUNK_N
        split_end = tl.minimum(split_start + CHUNK_N, Sk)

        for block_idx in range(BLOCKS_PER_SPLIT):
            key_pos = split_start + block_idx * BN + offs_n
            k_ptrs = (
                K + ((pid_b * Hkv + kv_h) * Sk + key_pos[None, :]) * D
                + offs_d[:, None]
            )
            v_ptrs = (
                V + ((pid_b * Hkv + kv_h) * Sk + key_pos[:, None]) * D
                + offs_d[None, :]
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
            valid = (offs_m[:, None] < Sq) & (key_pos[None, :] < split_end)
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(
                m_i == float("-inf"),
                0.0,
                tl.exp(m_i - m_new),
            )
            p = tl.where(valid, tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        safe_l = tl.maximum(l_i, 1.0)
        partial_o = tl.where(l_i[:, None] > 0.0, acc / safe_l[:, None], 0.0)
        partial_lse = tl.where(
            l_i > 0.0,
            m_i + tl.log(l_i),
            float("-inf"),
        )

        partial_base = (
            ((pid_split * BATCH + pid_b) * Hq + pid_h) * Sq * D
        )
        partial_o_ptrs = (
            PART_O + partial_base + offs_m[:, None] * D + offs_d[None, :]
        )
        partial_lse_base = (
            ((pid_split * BATCH + pid_b) * Hq + pid_h) * Sq
        )
        tl.store(
            partial_o_ptrs,
            partial_o,
            mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
        )
        tl.store(
            PART_LSE + partial_lse_base + offs_m,
            partial_lse,
            mask=offs_m < Sq,
        )


    @triton.jit
    def _attention_fwd_split_kv_reduce(
        PART_O, PART_LSE, O, LSE,
        BATCH: tl.constexpr,
        Sq: tl.constexpr, Hq: tl.constexpr,
        D: tl.constexpr, NUM_SPLITS: tl.constexpr,
        SPLIT_PAD: tl.constexpr,
        BM: tl.constexpr, BD: tl.constexpr,
    ):
        """Merge split-KV partial outputs using their log-sum-exp weights."""
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        offs_s = tl.arange(0, SPLIT_PAD)

        lse_ptrs = (
            PART_LSE
            + (offs_s[:, None] * BATCH * Hq + pid_b * Hq + pid_h) * Sq
            + offs_m[None, :]
        )
        lse_all = tl.load(
            lse_ptrs,
            mask=(offs_s[:, None] < NUM_SPLITS) & (offs_m[None, :] < Sq),
            other=float("-inf"),
        )
        row_max = tl.max(lse_all, axis=0)
        exp_weights = tl.exp(lse_all - row_max[None, :])
        exp_weights = tl.where(
            (offs_s[:, None] < NUM_SPLITS) & (offs_m[None, :] < Sq),
            exp_weights,
            0.0,
        )
        denom = tl.sum(exp_weights, axis=0)
        safe_max = tl.where(denom > 0.0, row_max, 0.0)
        safe_denom = tl.maximum(denom, 1.0)

        out_acc = tl.zeros((BM, BD), tl.float32)
        for split_idx in range(NUM_SPLITS):
            lse_s = tl.load(
                PART_LSE
                + (split_idx * BATCH * Hq + pid_b * Hq + pid_h) * Sq
                + offs_m,
                mask=offs_m < Sq,
                other=float("-inf"),
            )
            weight = tl.exp(lse_s - safe_max) / safe_denom
            part_base = (
                ((split_idx * BATCH + pid_b) * Hq + pid_h) * Sq * D
            )
            part_ptrs = (
                PART_O + part_base + offs_m[:, None] * D + offs_d[None, :]
            )
            part = tl.load(
                part_ptrs,
                mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
                other=0.0,
            )
            out_acc += weight[:, None] * part

        out_ptrs = (
            O + ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D
            + offs_d[None, :]
        )
        tl.store(
            out_ptrs,
            out_acc,
            mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
        )
        row_lse = tl.where(
            denom > 0.0,
            row_max + tl.log(denom),
            float("-inf"),
        )
        tl.store(
            LSE + (pid_b * Hq + pid_h) * Sq + offs_m,
            row_lse,
            mask=offs_m < Sq,
        )


def triton_available() -> bool:
    return triton is not None


def triton_turing_supported() -> bool:
    """Return whether installed upstream Triton is the Turing-capable 3.2 line.

    Upstream Triton removed Volta/Turing support in 3.3. Newer versions may
    work only when using a maintained Turing-enabled downstream build; those
    builds are not auto-detected by this conservative check.
    """
    if triton is None:
        return False
    try:
        parts = tuple(int(piece) for piece in triton.__version__.split("+")[0].split(".")[:2])
    except (AttributeError, ValueError):
        return False
    return parts == (3, 2)


def attention_forward(
    q, k, v, *, causal: bool, scale: float,
    block_m: int = 64, block_n: int = 32,
    num_warps: int = 4, num_stages: int = 2,
):
    """Run Triton forward on contiguous [B,H,S,D] FP16 CUDA tensors."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Install a Turing-compatible Triton 3.2.x build.")
    import torch

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Triton backend requires CUDA tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("Triton backend currently supports float16 only")
    if not (16 <= q.shape[-1] <= 128):
        raise ValueError("Triton backend currently supports 16 <= head_dim <= 128")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Triton kernel expects contiguous inputs")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")
    if num_warps <= 0 or num_stages <= 0:
        raise ValueError("num_warps and num_stages must be positive")

    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    out = q.new_empty((b, hq, sq, d))
    lse = q.new_empty((b, hq, sq), dtype=torch.float32)

    # During autoregressive decoding, Q is short while K/V may contain
    # thousands of cached tokens. Splitting K/V exposes more independent
    # CTAs than one CTA per query head and avoids under-utilizing T4.
    if not causal and sq <= 16 and sk >= 1024 and d <= 64:
        block_m_decode = 16
        block_n_decode = 64
        num_splits = min(16, triton.cdiv(sk, 128))
        chunk_n = triton.cdiv(sk, num_splits)
        blocks_per_split = triton.cdiv(chunk_n, block_n_decode)
        partial_o = torch.empty(
            (num_splits, b, hq, sq, d), device=q.device, dtype=torch.float32
        )
        partial_lse = torch.empty(
            (num_splits, b, hq, sq), device=q.device, dtype=torch.float32
        )
        _attention_fwd_split_kv[
            (num_splits, b, hq)
        ](
            q, k, v, partial_o, partial_lse,
            b, sq, sk, hq, hkv, d, scale,
            chunk_n, blocks_per_split,
            block_m_decode, block_n_decode, triton.next_power_of_2(d),
            num_warps=4,
            num_stages=1,
        )
        _attention_fwd_split_kv_reduce[
            (triton.cdiv(sq, block_m_decode), b, hq)
        ](
            partial_o, partial_lse, out, lse,
            b, sq, hq, d, num_splits, triton.next_power_of_2(num_splits),
            block_m_decode, triton.next_power_of_2(d),
            num_warps=4,
            num_stages=1,
        )
        return out, lse

    block_d = triton.next_power_of_2(d)
    tail_d = max(16, triton.next_power_of_2(max(1, d - 64)))
    _attention_fwd_kernel[(triton.cdiv(sq, block_m), b, hq)](
        q, k, v, out, lse,
        sq, sk, hq, hkv, d, scale, causal,
        block_m, block_n, block_d, tail_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse
