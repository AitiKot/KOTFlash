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
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton

from kotflash.kernels.triton_forward import (
    attention_forward,
    triton_turing_supported,
)


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


def sdpa_native_gqa(q, k, v, causal: bool):
    """Use PyTorch's native enable_gqa path (may fall back to math on SM75)."""
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


def sdpa_expanded(q, k_expanded, v_expanded, causal: bool):
    """SDPA with KV heads expanded before the call, allowing fused backends."""
    return F.scaled_dot_product_attention(
        q, k_expanded, v_expanded, is_causal=causal
    )


def sdpa_forced_efficient(q, k_expanded, v_expanded, causal: bool):
    """Force PyTorch's memory-efficient backend; raises if unsupported."""
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return sdpa_expanded(q, k_expanded, v_expanded, causal)


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
    parser.add_argument(
        "--allow-unsupported-triton",
        action="store_true",
        help="run anyway for diagnostics; results are not valid T4 performance claims",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 5):
        parser.error("This tuner is intended for NVIDIA T4 / SM75")
    if not triton_turing_supported():
        message = (
            f"T4 tuning requires upstream Triton 3.2.x; detected {triton.__version__}. "
            "Upstream Triton 3.3+ dropped Turing (SM75) support. "
            "Use a compatible T4 environment, or pass --allow-unsupported-triton "
            "only for diagnostics (not valid performance measurements)."
        )
        if not args.allow_unsupported_triton:
            parser.error(message)
        print("WARNING: " + message, flush=True)
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

    # Native enable_gqa is retained as a diagnostic baseline. In PyTorch
    # versions where GQA is restricted to math/flash, T4 (SM75) can be forced
    # onto the math path because flash is unavailable. Expand KV heads once
    # to let SDPA independently select an SM75-compatible fused backend.
    native_gqa_fn = lambda: sdpa_native_gqa(q, k, v, args.causal)
    reference = native_gqa_fn()

    groups = args.q_heads // args.kv_heads
    k_expanded = k.repeat_interleave(groups, dim=1).contiguous()
    v_expanded = v.repeat_interleave(groups, dim=1).contiguous()
    expanded_auto_fn = lambda: sdpa_expanded(q, k_expanded, v_expanded, args.causal)
    expanded_efficient_fn = lambda: sdpa_forced_efficient(
        q, k_expanded, v_expanded, args.causal
    )
    expanded_inclusive_fn = lambda: sdpa_expanded(
        q,
        k.repeat_interleave(groups, dim=1).contiguous(),
        v.repeat_interleave(groups, dim=1).contiguous(),
        args.causal,
    )

    baselines = {}

    def profile_baseline(name, fn):
        try:
            output = fn()
            torch.cuda.synchronize()
            error = (output.float() - reference.float()).abs().max().item()
            elapsed = median_cuda_ms(fn, args.warmup, args.iterations)
            baselines[name] = {"median_ms": elapsed, "max_abs_error": error}
            print(
                f"SDPA {name}: {elapsed:.4f} ms | "
                f"error={error:.6g}",
                flush=True,
            )
        except Exception as exc:
            baselines[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"SDPA {name} UNAVAILABLE: {type(exc).__name__}: {exc}", flush=True)

    profile_baseline("native_gqa_auto", native_gqa_fn)
    profile_baseline("expanded_kv_auto_precomputed", expanded_auto_fn)
    profile_baseline("expanded_kv_efficient_precomputed", expanded_efficient_fn)
    profile_baseline("expanded_kv_auto_inclusive", expanded_inclusive_fn)

    native_gqa_ms = baselines.get("native_gqa_auto", {}).get("median_ms")
    expanded_auto_ms = baselines.get("expanded_kv_auto_precomputed", {}).get("median_ms")
    efficient_ms = baselines.get("expanded_kv_efficient_precomputed", {}).get("median_ms")
    expanded_inclusive_ms = baselines.get("expanded_kv_auto_inclusive", {}).get("median_ms")

    scale = 1.0 / math.sqrt(args.head_dim)
    results = []
    failed_configs = []
    attempted_configs = 0
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
                    attempted_configs += 1
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
                            "speedup_vs_sdpa_native_gqa": (
                                native_gqa_ms / elapsed if native_gqa_ms else None
                            ),
                            "speedup_vs_sdpa_expanded_auto": (
                                expanded_auto_ms / elapsed if expanded_auto_ms else None
                            ),
                            "speedup_vs_sdpa_efficient": (
                                efficient_ms / elapsed if efficient_ms else None
                            ),
                            "speedup_vs_sdpa_expanded_inclusive": (
                                expanded_inclusive_ms / elapsed if expanded_inclusive_ms else None
                            ),
                            "max_abs_error": error,
                        }
                        results.append(row)
                        ratios = []
                        if native_gqa_ms:
                            ratios.append(f"native={native_gqa_ms / elapsed:.2f}x")
                        if expanded_auto_ms:
                            ratios.append(f"auto-expanded={expanded_auto_ms / elapsed:.2f}x")
                        if efficient_ms:
                            ratios.append(f"efficient={efficient_ms / elapsed:.2f}x")
                        if expanded_inclusive_ms:
                            ratios.append(f"incl-expand={expanded_inclusive_ms / elapsed:.2f}x")
                        print(
                            f"{label} | {elapsed:.4f} ms | "
                            f"{' '.join(ratios)} | error={error:.6g}",
                            flush=True,
                        )
                    except Exception as exc:
                        failure = {
                            "BM": block_m,
                            "BN": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failed_configs.append(failure)
                        print(
                            f"{label} FAILED: {type(exc).__name__}: {exc}",
                            flush=True,
                        )

    results.sort(key=lambda row: row["median_ms"])
    report = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "t4_compatibility_override": args.allow_unsupported_triton,
        "sdpa_baselines": baselines,
        "shape": {
            "B": args.batch, "H_Q": args.q_heads, "H_KV": args.kv_heads,
            "SEQ": args.seq, "D": args.head_dim,
        },
        "causal": args.causal,
        "sdpa_native_gqa_median_ms": native_gqa_ms,
        "sdpa_expanded_auto_precomputed_median_ms": expanded_auto_ms,
        "sdpa_expanded_efficient_precomputed_median_ms": efficient_ms,
        "sdpa_expanded_auto_inclusive_median_ms": expanded_inclusive_ms,
        "best_configs": results[:8],
        "attempted_config_count": attempted_configs,
        "successful_config_count": len(results),
        "failed_config_count": len(failed_configs),
        "failed_configs": failed_configs,
        "all_configs": results,
    }
    print("\n" + json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
