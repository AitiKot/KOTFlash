#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal GQA prefill + fixed-capacity KV-cache decode example.

This is an attention integration example, not a complete language model:
Q/K/V projections and token sampling are intentionally left to the caller.
"""
from __future__ import annotations

import argparse
import torch

from kotflash import attention, triton_turing_supported
from kotflash.cache import PreallocatedKVCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdpa", "triton", "auto"), default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=2047)
    parser.add_argument("--decode-steps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This example uses CUDA; run it on an NVIDIA GPU.")
    if min(args.batch, args.q_heads, args.kv_heads, args.head_dim,
           args.prefill_tokens, args.decode_steps) <= 0:
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
    max_len = args.prefill_tokens + args.decode_steps

    with torch.inference_mode():
        q_prefill = torch.randn(
            args.batch, args.q_heads, args.prefill_tokens, args.head_dim,
            device=device, dtype=dtype,
        )
        k_prefill = torch.randn(
            args.batch, args.kv_heads, args.prefill_tokens, args.head_dim,
            device=device, dtype=dtype,
        )
        v_prefill = torch.randn_like(k_prefill)
        cache = PreallocatedKVCache(
            args.batch, args.kv_heads, max_len, args.head_dim,
            device=device, dtype=dtype,
        )
        cache.append(k_prefill, v_prefill)
        prefill_k, prefill_v = cache.view()
        prefill_output = attention(
            q_prefill, prefill_k, prefill_v, causal=True, backend=args.backend
        )
        print(
            "prefill:", tuple(prefill_output.shape),
            "cache length:", cache.length, "capacity:", cache.max_len,
        )
        del q_prefill, k_prefill, v_prefill, prefill_k, prefill_v, prefill_output

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
            cache.append(k_new, v_new)
            output = cache.attend(q_new, backend=args.backend)
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite attention output during decode")
            print("decode step:", step + 1, "output:", tuple(output.shape),
                  "cache length:", cache.length)

    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("Turing Triton supported:", triton_turing_supported())


if __name__ == "__main__":
    main()
