#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible KOTFlash vs PyTorch SDPA benchmark on CUDA or CPU."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from kotflash import attention


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, warmup: int, iterations: int, device):
    for _ in range(warmup):
        fn()
    sync(device)

    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        events = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        sync(device)
        elapsed_ms = [
            start.elapsed_time(end) for start, end in events
        ]
        peak = max(
            0, torch.cuda.max_memory_allocated(device) - baseline
        )
    else:
        elapsed_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        peak = None

    return statistics.median(elapsed_ms), peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--q-heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--kv-seq", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()

    if args.batch <= 0 or args.q_heads <= 0 or args.kv_heads <= 0:
        p.error("batch and head counts must be positive")
    if args.q_heads % args.kv_heads:
        p.error("--q-heads must be divisible by --kv-heads")
    if args.seq <= 0 or (args.kv_seq is not None and args.kv_seq <= 0):
        p.error("sequence lengths must be positive")
    if args.head_dim <= 0 or args.warmup < 0 or args.iterations <= 0:
        p.error(
            "head dimension must be positive, warmup non-negative "
            "and iterations positive"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but torch.cuda.is_available() is False")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    sk = args.kv_seq or args.seq
    q = torch.randn(
        args.batch, args.q_heads, args.seq, args.head_dim,
        device=device, dtype=dtype, requires_grad=True,
    )
    k = torch.randn(
        args.batch, args.kv_heads, sk, args.head_dim,
        device=device, dtype=dtype, requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)

    groups = args.q_heads // args.kv_heads

    def sdpa_native_gqa():
        return attention(
            q, k, v, causal=args.causal, backend="sdpa"
        )

    def sdpa_expanded_kv():
        expanded_k = (
            k if groups == 1
            else k.repeat_interleave(groups, dim=1)
        )
        expanded_v = (
            v if groups == 1
            else v.repeat_interleave(groups, dim=1)
        )
        return F.scaled_dot_product_attention(
            q, expanded_k, expanded_v, is_causal=args.causal
        )

    def triton_fn():
        return attention(
            q, k, v, causal=args.causal, backend="triton"
        )

    def auto_fn():
        return attention(
            q, k, v, causal=args.causal, backend="auto"
        )

    backends = {
        "pytorch_sdpa_native_gqa": sdpa_native_gqa,
        "pytorch_sdpa_expanded_kv": sdpa_expanded_kv,
        "triton": triton_fn,
        "auto": auto_fn,
    }

    expanded_reference = sdpa_expanded_kv()
    sync(device)

    report = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU"
        ),
        "capability": (
            torch.cuda.get_device_capability(device)
            if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "shape": {
            "B": args.batch,
            "Hq": args.q_heads,
            "Hkv": args.kv_heads,
            "Sq": args.seq,
            "Sk": sk,
            "D": args.head_dim,
        },
        "dtype": str(dtype),
        "causal": args.causal,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "comparison_note": (
            "expanded_kv includes repeat_interleave cost; both PyTorch "
            "baselines use the same Q/K/V dtype and causal setting"
        ),
    }

    results = {}
    for name, fn in backends.items():
        try:
            output = fn()
            sync(device)
            output_error = (
                output.float() - expanded_reference.float()
            ).abs().max().item()

            def train_step():
                q.grad = None
                k.grad = None
                v.grad = None
                y = fn()
                y.float().square().mean().backward()

            forward_ms, forward_mem = timed(
                fn, args.warmup, args.iterations, device
            )
            train_ms, train_mem = timed(
                train_step,
                max(2, args.warmup // 4),
                max(5, args.iterations // 4),
                device,
            )
            results[name] = {
                "forward_ms": forward_ms,
                "forward_incremental_peak_allocated_bytes": forward_mem,
                "forward_backward_ms": train_ms,
                "forward_backward_incremental_peak_allocated_bytes": train_mem,
                "max_abs_error_vs_expanded_sdpa": output_error,
            }
        except Exception as exc:
            results[name] = {
                "unavailable_or_failed": f"{type(exc).__name__}: {exc}"
            }

    report.update(results)

    for backend in ("triton", "auto"):
        candidate = report.get(backend, {})
        for baseline in (
            "pytorch_sdpa_native_gqa",
            "pytorch_sdpa_expanded_kv",
        ):
            reference = report.get(baseline, {})
            if (
                "forward_ms" in candidate
                and "forward_ms" in reference
            ):
                report[f"{backend}_forward_speedup_vs_{baseline}"] = (
                    reference["forward_ms"] / candidate["forward_ms"]
                )
            if (
                "forward_backward_ms" in candidate
                and "forward_backward_ms" in reference
            ):
                report[
                    f"{backend}_forward_backward_speedup_vs_{baseline}"
                ] = (
                    reference["forward_backward_ms"]
                    / candidate["forward_backward_ms"]
                )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
