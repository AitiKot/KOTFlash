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
