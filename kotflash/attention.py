# SPDX-License-Identifier: Apache-2.0
"""Public KOTFlash attention API."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .kernels.triton_forward import attention_forward, triton_turing_supported


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("q, k and v must be torch.Tensor instances")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q/k/v in [batch, heads, sequence, head_dim] layout")
    if q.shape[0] == 0:
        raise ValueError("Batch dimension must be non-zero")
    if q.shape[1] == 0 or k.shape[1] == 0:
        raise ValueError("Head dimensions must be non-zero")
    if q.shape[-1] == 0:
        raise ValueError("Head dimension must be non-zero")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Batch dimensions must match")
    if k.shape != v.shape:
        raise ValueError("K and V shapes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads (GQA/MQA)")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if q.shape[2] == 0 or k.shape[2] == 0:
        raise ValueError("Sequence dimensions must be non-zero")


def _sdpa(q, k, v, *, causal: bool, scale: Optional[float]):
    kwargs = {"attn_mask": None, "dropout_p": 0.0, "is_causal": causal}
    if scale is not None:
        kwargs["scale"] = scale

    if q.shape[1] != k.shape[1]:
        try:
            # GQA is implemented natively by modern PyTorch SDPA.
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
        except TypeError as exc:
            # Older PyTorch releases do not expose enable_gqa. Only fall back
            # for that specific API mismatch, not for arbitrary TypeErrors.
            if "enable_gqa" not in str(exc):
                raise
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _blockwise_backward(
    q, k, v, out, lse, grad_out, *, causal: bool, scale: float,
    q_block: int = 32, kv_block: int = 64,
):
    """Memory-bounded reference backward; recomputes score tiles instead of storing SxS.

    This is intentionally a correctness-oriented PyTorch implementation, not a
    fused high-performance backward kernel. Its Python loops make it slow for
    long sequences; use it for validation until a native SM75 backward lands.
    """
    bsz, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    group = hq // hkv
    qf, kf, vf = q.float(), k.float(), v.float()
    of, dof = out.float(), grad_out.float()
    lsef = lse.float()
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)

    for b in range(bsz):
        for h in range(hq):
            kh = h // group
            for qs in range(0, sq, q_block):
                qe = min(qs + q_block, sq)
                qb = qf[b, h, qs:qe]
                dob = dof[b, h, qs:qe]
                lseb = lsef[b, h, qs:qe]
                delta = (of[b, h, qs:qe] * dob).sum(dim=-1)
                dq_block = torch.zeros_like(qb)
                qpos = torch.arange(qs, qe, device=q.device)
                for ks in range(0, sk, kv_block):
                    ke = min(ks + kv_block, sk)
                    kb = kf[b, kh, ks:ke]
                    vb = vf[b, kh, ks:ke]
                    scores = torch.matmul(qb, kb.transpose(0, 1)) * scale
                    kpos = torch.arange(ks, ke, device=q.device)
                    if causal:
                        scores = scores.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
                    p = torch.exp(scores - lseb[:, None])
                    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                    dp = torch.matmul(dob, vb.transpose(0, 1))
                    ds = p * (dp - delta[:, None])
                    dq_block.add_(torch.matmul(ds, kb) * scale)
                    dk[b, kh, ks:ke].add_(torch.matmul(ds.transpose(0, 1), qb) * scale)
                    dv[b, kh, ks:ke].add_(torch.matmul(p.transpose(0, 1), dob))
                dq[b, h, qs:qe] = dq_block
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class _TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool, scale: float):
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = attention_forward(qc, kc, vc, causal=causal, scale=scale)
        ctx.save_for_backward(qc, kc, vc, out, lse)
        ctx.causal = causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        grads = _blockwise_backward(
            q, k, v, out, lse, grad_out.contiguous(),
            causal=ctx.causal, scale=ctx.scale,
        )
        return grads[0], grads[1], grads[2], None, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Scaled dot-product attention with MHA, GQA and MQA support.

    Tensors use [batch, heads, sequence, head_dim] layout. For GQA/MQA,
    Hq must be divisible by Hkv. The causal mask is upper-left aligned
    (key position <= query position), matching PyTorch SDPA.

    backend is one of auto, triton or sdpa. The Triton forward requires
    CUDA SM75, FP16, Triton 3.2.x and head_dim in [16, 128]. Its backward
    is a slow, blockwise PyTorch reference implementation.
    """
    _validate(q, k, v)
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    else:
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise TypeError("scale must be a finite real number or None") from exc
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
    if backend not in {"auto", "triton", "sdpa"}:
        raise ValueError("backend must be one of: auto, triton, sdpa")

    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    can_triton = (
        q.is_cuda
        and q.dtype == torch.float16
        and triton_turing_supported()
        and 16 <= q.shape[-1] <= 128
        and capability == (7, 5)
    )
    if backend == "triton" and not can_triton:
        raise RuntimeError(
            "Triton backend requires Triton 3.2.x (or a compatible Turing-enabled build), "
            "CUDA SM75, FP16 and 16 <= head_dim <= 128"
        )
    if backend == "triton" or (backend == "auto" and can_triton):
        return _TritonAttention.apply(q, k, v, causal, float(scale))
    return _sdpa(q, k, v, causal=causal, scale=scale)
