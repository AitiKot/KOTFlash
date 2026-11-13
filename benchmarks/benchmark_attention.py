#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved KOTFlash vs PyTorch SDPA benchmark on CUDA or CPU."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

# Make direct execution work from a fresh repository checkout:
# `python benchmarks/benchmark_attention.py` sets sys.path[0] to benchmarks/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from kotflash import attention


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, iterations, device):
    """Return median latency and incremental peak allocated memory."""
    if device.type == "cuda":
        sync(device)
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
        elapsed = [start.elapsed_time(end) for start, end in events]
        peak = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    else:
        elapsed = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed.append((time.perf_counter() - start) * 1000.0)
        peak = None
    return statistics.median(elapsed), peak


def summary(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "per_round": values,
    }


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
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if min(args.batch, args.q_heads, args.kv_heads, args.seq, args.head_dim) <= 0:
        p.error("batch, head counts, sequence length and head dimension must be positive")
    if args.q_heads % args.kv_heads:
        p.error("--q-heads must be divisible by --kv-heads")
    if args.kv_seq is not None and args.kv_seq <= 0:
        p.error("--kv-seq must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats < 3:
        p.error("warmup must be non-negative, iterations positive and repeats at least 3")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but torch.cuda.is_available() is False")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    sk = args.kv_seq or args.seq
    q = torch.randn(args.batch, args.q_heads, args.seq, args.head_dim,
                    device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.kv_heads, sk, args.head_dim,
                    device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    groups = args.q_heads // args.kv_heads

    def native_gqa():
        return attention(q, k, v, causal=args.causal, backend="sdpa")

    def expanded_sdpa():
        ek = k if groups == 1 else k.repeat_interleave(groups, dim=1)
        ev = v if groups == 1 else v.repeat_interleave(groups, dim=1)
        return F.scaled_dot_product_attention(q, ek, ev, is_causal=args.causal)

    def triton_fn():
        return attention(q, k, v, causal=args.causal, backend="triton")

    def auto_fn():
        return attention(q, k, v, causal=args.causal, backend="auto")

    backends = {
        "pytorch_sdpa_native_gqa": native_gqa,
        "pytorch_sdpa_expanded_kv": expanded_sdpa,
        "triton": triton_fn,
        "auto": auto_fn,
    }

    def make_forward(fn):
        def run():
            with torch.no_grad():
                fn()
        return run

    def make_backward(fn):
        def run():
            q.grad = None
            k.grad = None
            v.grad = None
            fn().float().square().mean().backward()
        return run

    train_iterations = max(5, args.iterations // 4)
    train_warmup = max(2, args.warmup // 4)
    prepared, failures = {}, {}

    # Warm all paths (including backward compilation) before measuring any.
    for name, fn in backends.items():
        try:
            fwd, bwd = make_forward(fn), make_backward(fn)
            for _ in range(args.warmup):
                fwd()
            for _ in range(train_warmup):
                bwd()
            sync(device)
            prepared[name] = (fwd, bwd)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    with torch.no_grad():
        reference = expanded_sdpa()
    sync(device)

    samples = {
        name: {"forward": [], "forward_backward": [], "forward_peak": [], "train_peak": []}
        for name in prepared
    }
    rng = random.Random(args.seed)
    orders = []
    for _ in range(args.repeats):
        order = list(prepared)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            fwd, bwd = prepared[name]
            try:
                with torch.no_grad():
                    output = backends[name]()
                sync(device)
                error = (output.float() - reference.float()).abs().max().item()
                del output
                f_ms, f_mem = timed(fwd, args.iterations, device)
                # Do not count stale warm-up gradients in the backward
                # incremental-memory baseline.
                q.grad = None
                k.grad = None
                v.grad = None
                b_ms, b_mem = timed(bwd, train_iterations, device)
                sample = samples[name]
                sample["forward"].append(f_ms)
                sample["forward_backward"].append(b_ms)
                if f_mem is not None:
                    sample["forward_peak"].append(f_mem)
                    sample["train_peak"].append(b_mem)
                sample["error"] = error
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                samples.pop(name, None)
                prepared.pop(name, None)

    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "shape": {"B": args.batch, "Hq": args.q_heads, "Hkv": args.kv_heads,
                  "Sq": args.seq, "Sk": sk, "D": args.head_dim},
        "dtype": str(dtype),
        "causal": args.causal,
        "warmup": args.warmup,
        "iterations_per_round": args.iterations,
        "backward_iterations_per_round": train_iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_order_by_round": orders,
        "comparison_note": (
            "All backends are warmed before timing; order is shuffled each round "
            "with a fixed seed. Latencies are medians of per-round medians. "
            "Expanded-KV includes repeat_interleave cost."
        ),
    }

    for name, sample in samples.items():
        report[name] = {
            "forward_ms": summary(sample["forward"]),
            "forward_backward_ms": summary(sample["forward_backward"]),
            "forward_incremental_peak_allocated_bytes": (
                int(statistics.median(sample["forward_peak"])) if sample["forward_peak"] else None
            ),
            "forward_backward_incremental_peak_allocated_bytes": (
                int(statistics.median(sample["train_peak"])) if sample["train_peak"] else None
            ),
            "max_abs_error_vs_expanded_sdpa": sample.get("error"),
        }
    if failures:
        report["backend_failures"] = failures

    for backend in ("triton", "auto"):
        candidate = report.get(backend, {})
        for baseline in ("pytorch_sdpa_native_gqa", "pytorch_sdpa_expanded_kv"):
            reference_result = report.get(baseline, {})
            for metric, key in (("forward", "forward_ms"),
                                ("forward_backward", "forward_backward_ms")):
                c, r = candidate.get(key), reference_result.get(key)
                if c and r and c["median"] > 0:
                    report[f"{backend}_{metric}_speedup_vs_{baseline}"] = (
                        r["median"] / c["median"]
                    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
