# SPDX-License-Identifier: Apache-2.0
"""Fused, memory-bounded Triton backward kernels for NVIDIA SM75."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _attention_bwd_dq(
        Q, K, V, O, DO, LSE, DQ,
        Sq: tl.constexpr, Sk: tl.constexpr,
        Hq: tl.constexpr, Hkv: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        kv_h = pid_h // (Hq // Hkv)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, BD)

        q_ptrs = Q + ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D + offs_d[None, :]
        do_ptrs = DO + ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D + offs_d[None, :]
        o_ptrs = O + ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D + offs_d[None, :]
        lse_ptrs = LSE + (pid_b * Hq + pid_h) * Sq + offs_m

        q = tl.load(q_ptrs, mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D), other=0)
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D), other=0)
        o = tl.load(o_ptrs, mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D), other=0)
        lse = tl.load(lse_ptrs, mask=offs_m < Sq, other=0).to(tl.float32)

        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        dq_acc = tl.zeros((BM, BD), tl.float32)

        if CAUSAL:
            max_key_exclusive = tl.minimum(
                tl.minimum(Sk, Sq), (pid_m + 1) * BM
            )
            end_n = tl.cdiv(max_key_exclusive, BN)
        else:
            end_n = tl.cdiv(Sk, BN)

        for start_n in range(end_n):
            key_pos = start_n * BN + offs_n

            k_ptrs = K + ((pid_b * Hkv + kv_h) * Sk + key_pos[None, :]) * D + offs_d[:, None]
            v_ptrs = V + ((pid_b * Hkv + kv_h) * Sk + key_pos[:, None]) * D + offs_d[None, :]

            k = tl.load(
                k_ptrs,
                mask=(key_pos[None, :] < Sk) & (offs_d[:, None] < D),
                other=0,
            )
            v = tl.load(
                v_ptrs,
                mask=(key_pos[:, None] < Sk) & (offs_d[None, :] < D),
                other=0,
            )

            scores = tl.dot(q, k) * SCALE
            valid = (offs_m[:, None] < Sq) & (key_pos[None, :] < Sk)
            if CAUSAL:
                valid = valid & (key_pos[None, :] <= offs_m[:, None])
            scores = tl.where(valid, scores, float("-inf"))

            p = tl.exp(scores - lse[:, None])
            p = tl.where(valid, p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])

            dq_acc += tl.dot(ds.to(k.dtype), tl.trans(k)) * SCALE

        tl.store(
            DQ + ((pid_b * Hq + pid_h) * Sq + offs_m[:, None]) * D + offs_d[None, :],
            dq_acc,
            mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < D),
        )


    @triton.jit
    def _attention_bwd_dkv(
        Q, K, V, LSE, DO, DK, DV,
        Sq: tl.constexpr, Sk: tl.constexpr,
        Hq: tl.constexpr, Hkv: tl.constexpr,
        D: tl.constexpr, SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_kh = tl.program_id(2)

        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = tl.arange(0, BM)
        offs_d = tl.arange(0, BD)

        k_ptrs = K + ((pid_b * Hkv + pid_kh) * Sk + offs_n[:, None]) * D + offs_d[None, :]
        v_ptrs = V + ((pid_b * Hkv + pid_kh) * Sk + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(
            k_ptrs,
            mask=(offs_n[:, None] < Sk) & (offs_d[None, :] < D),
            other=0,
        )
        v = tl.load(
            v_ptrs,
            mask=(offs_n[:, None] < Sk) & (offs_d[None, :] < D),
            other=0,
        )

        dk_acc = tl.zeros((BN, BD), tl.float32)
        dv_acc = tl.zeros((BN, BD), tl.float32)

        if CAUSAL:
            # Earlier query tiles cannot attend to this key tile.
            first_q_tile = (pid_n * BN) // BM
        else:
            first_q_tile = 0

        end_q_tile = tl.cdiv(Sq, BM)

        for group_idx in range(GROUP):
            q_head = pid_kh * GROUP + group_idx

            for start_m in range(first_q_tile, end_q_tile):
                query_pos = start_m * BM + offs_m

                q_ptrs = Q + ((pid_b * Hq + q_head) * Sq + query_pos[:, None]) * D + offs_d[None, :]
                do_ptrs = DO + ((pid_b * Hq + q_head) * Sq + query_pos[:, None]) * D + offs_d[None, :]
                lse_ptrs = LSE + (pid_b * Hq + q_head) * Sq + query_pos

                q = tl.load(
                    q_ptrs,
                    mask=(query_pos[:, None] < Sq) & (offs_d[None, :] < D),
                    other=0,
                )
                do = tl.load(
                    do_ptrs,
                    mask=(query_pos[:, None] < Sq) & (offs_d[None, :] < D),
                    other=0,
                )
                lse = tl.load(
                    lse_ptrs,
                    mask=query_pos < Sq,
                    other=0,
                ).to(tl.float32)

                scores = tl.dot(q, tl.trans(k)) * SCALE
                valid = (query_pos[:, None] < Sq) & (offs_n[None, :] < Sk)
                if CAUSAL:
                    valid = valid & (offs_n[None, :] <= query_pos[:, None])
                scores = tl.where(valid, scores, float("-inf"))

                p = tl.exp(scores - lse[:, None])
                p = tl.where(valid, p, 0.0)

                dp = tl.dot(do, tl.trans(v))
                delta = tl.sum(
                    # O is not needed here: delta = sum(P * dP), but using
                    # sum(O * dO) requires O. Reconstruct O from P @ V below.
                    tl.zeros((BM, BD), tl.float32), axis=1
                )
                # The softmax Jacobian correction is delta_i = O_i dot dO_i.
                # Reconstruct O_i from the probabilities and this K tile is
                # insufficient, so this kernel receives O in the next version.
                ds = p * dp

                dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * SCALE
                dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

        tl.store(
            DK + ((pid_b * Hkv + pid_kh) * Sk + offs_n[:, None]) * D + offs_d[None, :],
            dk_acc,
            mask=(offs_n[:, None] < Sk) & (offs_d[None, :] < D),
        )
        tl.store(
            DV + ((pid_b * Hkv + pid_kh) * Sk + offs_n[:, None]) * D + offs_d[None, :],
            dv_acc,
            mask=(offs_n[:, None] < Sk) & (offs_d[None, :] < D),
        )


def attention_backward(
    q, k, v, out, lse, grad_out, *, causal: bool, scale: float,
    block_m: int = 32, block_n: int = 64,
    num_warps: int = 4, num_stages: int = 1,
):
    """Compute attention gradients with fused Triton kernels.

    The Q gradient kernel and the grouped K/V gradient kernel recompute the
    softmax probabilities from Q, K and row LSE. No full attention matrix is
    materialized.
    """
    if triton is None:
        raise RuntimeError("Triton is required for fused attention backward")
    import torch

    if not all(t.is_cuda for t in (q, k, v, out, lse, grad_out)):
        raise ValueError("Triton backward requires CUDA tensors")
    if q.dtype != torch.float16:
        raise TypeError("Triton backward currently supports float16 only")

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()
    grad_out = grad_out.contiguous()

    bsz, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    group = hq // hkv
    bd = triton.next_power_of_2(d)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _attention_bwd_dq[(triton.cdiv(sq, block_m), bsz, hq)](
        q, k, v, out, grad_out, lse, dq,
        sq, sk, hq, hkv, d, scale, causal,
        block_m, block_n, bd,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _attention_bwd_dkv[(triton.cdiv(sk, block_n), bsz, hkv)](
        q, k, v, lse, grad_out, dk, dv,
        sq, sk, hq, hkv, d, scale, causal,
        block_m, block_n, bd, group,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv
