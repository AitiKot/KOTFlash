#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep T4 Triton attention tile and launch parameters for one shape."""
from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from kotflash.kernels.triton_forward import attention_forward


def median_cuda_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def sdpa(q, k, v, causal: bool):
    try:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=(q.shape[1] != k.shape[1])
        )
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise
        groups = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(
            q, k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1), is_causal=causal
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=72)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 5):
        parser.error("This tuner is intended for NVIDIA T4 / SM75")
    if args.batch <= 0 or args.q_heads <= 0 or args.kv_heads <= 0:
        parser.error("batch and head counts must be positive")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.seq <= 0 or args.head_dim < 16 or args.head_dim > 128:
        parser.error("seq must be positive and head-dim must be in [16, 128]")
    if args.warmup < 1 or args.iterations < 5:
        parser.error("use at least one warmup and five timed iterations")

    device = "cuda"
    dtype = torch.float16
    q = torch.randn(
        args.batch, args.q_heads, args.seq, args.head_dim,
        device=device, dtype=dtype
    ).contiguous()
    k = torch.randn(
        args.batch, args.kv_heads, args.seq, args.head_dim,
        device=device, dtype=dtype
    ).contiguous()
    v = torch.randn_like(k)

    reference_fn = lambda: sdpa(q, k, v, args.causal)
    reference = reference_fn()
    sdpa_ms = median_cuda_ms(reference_fn, args.warmup, args.iterations)

    scale = 1.0 / math.sqrt(args.head_dim)
    results = []
    # These dimensions are intentionally bounded: all candidates are tested
    # on the real SM75 device, and resource failures are recorded, not fatal.
    for block_m in (32, 64):
        for block_n in (32, 64, 128):
            for num_warps in (4, 8):
                for num_stages in (1, 2):
                    label = (
                        f"BM={block_m:2d} BN={block_n:3d} "
                        f"warps={num_warps} stages={num_stages}"
                    )
                    try:
                        def fn():
                            return attention_forward(
                                q, k, v, causal=args.causal, scale=scale,
                                block_m=block_m, block_n=block_n,
                                num_warps=num_warps, num_stages=num_stages,
                            )[0]

                        output = fn()
                        torch.cuda.synchronize()
                        error = (output.float() - reference.float()).abs().max().item()
                        elapsed = median_cuda_ms(fn, args.warmup, args.iterations)
                        row = {
                            "BM": block_m,
                            "BN": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "median_ms": elapsed,
                            "speedup_vs_sdpa": sdpa_ms / elapsed,
                            "max_abs_error": error,
                        }
                        results.append(row)
                        print(
                            f"{label} | {elapsed:.4f} ms | "
                            f"{sdpa_ms / elapsed:.3f}x | error={error:.6g}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"{label} FAILED: {type(exc).__name__}: {exc}",
                            flush=True,
                        )

    results.sort(key=lambda row: row["median_ms"])
    report = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "shape": {
            "B": args.batch, "H_Q": args.q_heads, "H_KV": args.kv_heads,
            "SEQ": args.seq, "D": args.head_dim,
        },
        "causal": args.causal,
        "sdpa_median_ms": sdpa_ms,
        "best_configs": results[:8],
        "successful_config_count": len(results),
        "all_configs": results,
    }
    print("\n" + json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
