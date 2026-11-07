#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible KOTFlash vs PyTorch SDPA benchmark."""
from __future__ import annotations

import argparse
import json
import time

import torch

from kotflash import attention


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, warmup: int, iterations: int, device):
    for _ in range(warmup):
        fn()
    sync(device)
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync(device)
    elapsed = (time.perf_counter() - start) * 1000.0 / iterations
    peak = None
    if device.type == "cuda":
        peak = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    return elapsed, peak


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
        p.error("head dimension must be positive, warmup non-negative and iterations positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but torch.cuda.is_available() is False")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    sk = args.kv_seq or args.seq
    q = torch.randn(args.batch, args.q_heads, args.seq, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.kv_heads, sk, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)

    reference = lambda: attention(q, k, v, causal=args.causal, backend="sdpa")
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "shape": {"B": args.batch, "Hq": args.q_heads, "Hkv": args.kv_heads, "Sq": args.seq, "Sk": sk, "D": args.head_dim},
        "dtype": str(dtype),
        "causal": args.causal,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    ref_out = reference()
    ref_ms, ref_mem = timed(reference, args.warmup, args.iterations, device)

    def reference_train_step():
        q.grad = None
        k.grad = None
        v.grad = None
        y = reference()
        y.float().square().mean().backward()

    ref_train_ms, ref_train_mem = timed(
        reference_train_step, max(2, args.warmup // 4), max(5, args.iterations // 4), device
    )
    report["pytorch_sdpa"] = {
        "forward_ms": ref_ms,
        "forward_incremental_peak_allocated_bytes": ref_mem,
        "forward_backward_ms": ref_train_ms,
        "forward_backward_incremental_peak_allocated_bytes": ref_train_mem,
    }

    for backend in ("triton", "auto"):
        try:
            fn = lambda: attention(q, k, v, causal=args.causal, backend=backend)
            out = fn()
            sync(device)
            diff = (out.float() - ref_out.float()).abs().max().item()

            def train_step():
                q.grad = None
                k.grad = None
                v.grad = None
                y = fn()
                y.float().square().mean().backward()

            fwd_ms, fwd_mem = timed(fn, args.warmup, args.iterations, device)
            train_ms, train_mem = timed(
                train_step, max(2, args.warmup // 4), max(5, args.iterations // 4), device
            )
            report[backend] = {
                "forward_ms": fwd_ms,
                "forward_incremental_peak_allocated_bytes": fwd_mem,
                "forward_backward_ms": train_ms,
                "forward_backward_incremental_peak_allocated_bytes": train_mem,
                "max_abs_error_vs_sdpa": diff,
            }
        except Exception as exc:
            report[backend] = {"unavailable_or_failed": repr(exc)}

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
