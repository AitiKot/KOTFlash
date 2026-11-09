#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal caller-managed GQA KV-cache example for KOTFlash.

This is an attention integration example, not a complete language model:
Q/K/V projections and token sampling are intentionally left to the caller.
"""
from __future__ import annotations

import argparse
from typing import Tuple

import torch

from kotflash import attention, triton_turing_supported


def append_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append new [B, Hkv, Snew, D] entries without mutating the old cache."""
    tensors = (k_cache, v_cache, new_k, new_v)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("K/V cache and new K/V values must be torch.Tensor instances")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("All K/V tensors must use [B, Hkv, S, D] layout")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    if new_k.shape != new_v.shape:
        raise ValueError("New K and V shapes must match")
    if (
        k_cache.shape[0] != new_k.shape[0]
        or k_cache.shape[1] != new_k.shape[1]
        or k_cache.shape[3] != new_k.shape[3]
    ):
        raise ValueError("New K/V batch, head count and head dimension must match the cache")
    if k_cache.device != v_cache.device or k_cache.device != new_k.device or new_k.device != new_v.device:
        raise ValueError("All K/V tensors must be on the same device")
    if k_cache.dtype != v_cache.dtype or k_cache.dtype != new_k.dtype or new_k.dtype != new_v.dtype:
        raise ValueError("All K/V tensors must have the same dtype")
    if k_cache.shape[2] == 0 or new_k.shape[2] == 0:
        raise ValueError("Cache and appended K/V sequence lengths must be non-zero")

    return (
        torch.cat((k_cache, new_k), dim=2),
        torch.cat((v_cache, new_v), dim=2),
    )


def decode_step(
    q_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one token and attend to the now fully-visible cache.

    For a single newest token, every entry in the updated cache is visible.
    Therefore this call correctly uses causal=False. Do not pass future K/V
    entries to this function.
    """
    if not isinstance(q_new, torch.Tensor) or q_new.ndim != 4:
        raise ValueError("q_new must use [B, Hq, 1, D] layout")
    if q_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new query token")
    if k_new.ndim != 4 or k_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new K token")
    if v_new.ndim != 4 or v_new.shape[2] != 1:
        raise ValueError("decode_step accepts exactly one new V token")
    if (
        q_new.shape[0] != k_cache.shape[0]
        or q_new.shape[-1] != k_cache.shape[-1]
    ):
        raise ValueError("Query batch and head dimension must match the KV cache")

    updated_k, updated_v = append_kv_cache(k_cache, v_cache, k_new, v_new)
    output = attention(
        q_new,
        updated_k,
        updated_v,
        causal=False,
        backend=backend,
    )
    return output, updated_k, updated_v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdpa", "triton", "auto"), default="triton")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=2047,
        help="Prefill length; one token is appended before the first decode step.",
    )
    parser.add_argument("--decode-steps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This example uses CUDA; run it on an NVIDIA GPU.")
    if min(args.batch, args.q_heads, args.kv_heads, args.head_dim, args.prefill_tokens, args.decode_steps) <= 0:
        parser.error("All dimensions and step counts must be positive.")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads.")
    if not 16 <= args.head_dim <= 128:
        parser.error("--head-dim must be in [16, 128] for the Triton backend.")
    if args.backend == "triton":
        if torch.cuda.get_device_capability() != (7, 5) or not triton_turing_supported():
            parser.error("backend=triton requires an SM75 GPU and Turing-compatible Triton 3.2.x.")

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(2026)

    with torch.inference_mode():
        q_prefill = torch.randn(
            args.batch, args.q_heads, args.prefill_tokens, args.head_dim,
            device=device, dtype=dtype,
        )
        k_cache = torch.randn(
            args.batch, args.kv_heads, args.prefill_tokens, args.head_dim,
            device=device, dtype=dtype,
        )
        v_cache = torch.randn_like(k_cache)

        prefill_output = attention(
            q_prefill, k_cache, v_cache, causal=True, backend=args.backend
        )
        del q_prefill
        print(
            "prefill:",
            tuple(prefill_output.shape),
            "cache:",
            tuple(k_cache.shape),
            "backend:",
            args.backend,
        )
        del prefill_output

        for step in range(args.decode_steps):
            q_new = torch.randn(
                args.batch, args.q_heads, 1, args.head_dim,
                device=device, dtype=dtype,
            )
            k_new = torch.randn(
                args.batch, args.kv_heads, 1, args.head_dim,
                device=device, dtype=dtype,
            )
            v_new = torch.randn_like(k_new)
            output, k_cache, v_cache = decode_step(
                q_new, k_cache, v_cache, k_new, v_new, backend=args.backend
            )
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite attention output during decode")
            print(
                "decode step:",
                step + 1,
                "output:",
                tuple(output.shape),
                "cache length:",
                k_cache.shape[2],
            )

    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("Triton-compatible Turing build:", triton_turing_supported())


if __name__ == "__main__":
    main()
