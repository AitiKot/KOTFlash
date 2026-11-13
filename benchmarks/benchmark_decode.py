#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one-token GQA decode with caller-managed KV-cache updates.

This measures attention-only and append+attention separately. It is a kernel /
reference-attention benchmark, not an end-to-end language-model benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from kotflash.cache import append_kv_cache, PreallocatedKVCache
from kotflash import attention, triton_turing_supported


def synchronize() -> None:
    torch.cuda.synchronize()


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn()
        synchronize()
        del result

        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gpu_samples = []
        wall_samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                result = fn()
            end.record()
            synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000.0 / iterations)
            gpu_samples.append(start.elapsed_time(end) / iterations)
            del result

        peak_bytes = max(
            0, torch.cuda.max_memory_allocated() - baseline_bytes
        )
    return {
        "gpu_median_ms": statistics.median(gpu_samples),
        "gpu_min_ms": min(gpu_samples),
        "gpu_max_ms": max(gpu_samples),
        "gpu_per_repeat_ms": gpu_samples,
        "wall_median_ms": statistics.median(wall_samples),
        "wall_min_ms": min(wall_samples),
        "wall_max_ms": max(wall_samples),
        "wall_per_repeat_ms": wall_samples,
        "incremental_peak_allocated_bytes": int(peak_bytes),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-seq", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--append-backend", choices=("auto", "torch", "cuda"), default="auto",
        help="append implementation for the preallocated-cache benchmark",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark.")
    if min(args.batch, args.q_heads, args.kv_heads, args.head_dim, args.kv_seq) <= 0:
        parser.error("All tensor dimensions must be positive.")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads.")
    if not 16 <= args.head_dim <= 64:
        parser.error("Split-KV decode benchmark requires 16 <= --head-dim <= 64.")
    if args.kv_seq < 1024:
        parser.error("Split-KV decode benchmark requires --kv-seq >= 1024.")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        parser.error("Use warmup >= 1, iterations >= 1 and repeats >= 3.")
    if torch.cuda.get_device_capability() != (7, 5) or not triton_turing_supported():
        parser.error("This comparison requires an SM75 GPU and Turing-compatible Triton 3.2.x.")

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(1337)
    q = torch.randn(
        args.batch, args.q_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    past_k = torch.randn(
        args.batch, args.kv_heads, args.kv_seq - 1, args.head_dim,
        device=device, dtype=dtype,
    )
    past_v = torch.randn_like(past_k)
    new_k = torch.randn(
        args.batch, args.kv_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    new_v = torch.randn_like(new_k)
    visible_k, visible_v = append_kv_cache(past_k, past_v, new_k, new_v)
    groups = args.q_heads // args.kv_heads

    # Keep unused capacity after the visible prefix so the views are genuinely
    # strided. This is the path that previously incurred full-prefix contiguous
    # copies inside the Triton autograd wrapper.
    cache_capacity = max(args.kv_seq * 2, args.kv_seq + 16)
    prefilled_cache = PreallocatedKVCache(
        args.batch, args.kv_heads, cache_capacity, args.head_dim,
        device=device, dtype=dtype,
    )
    prefilled_cache.append(visible_k, visible_v, backend="torch")

    decode_cache = PreallocatedKVCache(
        args.batch, args.kv_heads, cache_capacity, args.head_dim,
        device=device, dtype=dtype,
    )
    decode_cache.append(past_k, past_v, backend="torch")

    def expanded_sdpa():
        return F.scaled_dot_product_attention(
            q,
            visible_k.repeat_interleave(groups, dim=1),
            visible_v.repeat_interleave(groups, dim=1),
            is_causal=False,
        )

    def triton_attention():
        return attention(q, visible_k, visible_v, causal=False, backend="triton")

    def auto_attention():
        return attention(q, visible_k, visible_v, causal=False, backend="auto")

    def append_only():
        updated_k, updated_v = append_kv_cache(past_k, past_v, new_k, new_v)
        return updated_k, updated_v

    def decode_expanded():
        k, v = append_kv_cache(past_k, past_v, new_k, new_v)
        return F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1),
            is_causal=False,
        )

    def decode_triton():
        k, v = append_kv_cache(past_k, past_v, new_k, new_v)
        return attention(q, k, v, causal=False, backend="triton")

    def decode_auto():
        k, v = append_kv_cache(past_k, past_v, new_k, new_v)
        return attention(q, k, v, causal=False, backend="auto")

    def preallocated_attention(backend):
        return prefilled_cache.attend(q, backend=backend)

    def preallocated_decode(backend):
        # Reset only the logical prefix; the last slot is overwritten on every
        # iteration. No allocation or copy of the existing KV history occurs.
        decode_cache.length = args.kv_seq - 1
        return decode_cache.append_and_attend(
            q, new_k, new_v,
            append_backend=args.append_backend,
            attention_backend=backend,
        )

    with torch.inference_mode():
        reference = expanded_sdpa()
        candidates = {
            "triton": (triton_attention, decode_triton),
            "auto": (auto_attention, decode_auto),
            "pytorch_expanded_sdpa": (expanded_sdpa, decode_expanded),
            "preallocated_triton": (
                lambda: preallocated_attention("triton"),
                lambda: preallocated_decode("triton"),
            ),
            "preallocated_auto": (
                lambda: preallocated_attention("auto"),
                lambda: preallocated_decode("auto"),
            ),
        }
        correctness = {}
        for name, (attention_fn, _) in candidates.items():
            output = attention_fn()
            torch.cuda.synchronize()
            max_error = float(
                (output.float() - reference.float()).abs().max().item()
            )
            finite = bool(torch.isfinite(output).all().item())
            close = bool(torch.allclose(
                output.float(), reference.float(), rtol=8e-3, atol=8e-3
            ))
            correctness[name] = {
                "max_abs_error_vs_expanded_sdpa": max_error,
                "finite": finite,
                "within_tolerance": close,
            }
            if not finite or not close:
                raise RuntimeError(
                    f"{name} failed the expanded-SDPA correctness check: "
                    f"finite={finite}, max_abs_error={max_error}, within_tolerance={close}"
                )
            del output

    report = {
        "benchmark": "one-token GQA decode; attention-only and append+attention",
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "query_length": 1,
            "visible_kv_length": args.kv_seq,
            "head_dim": args.head_dim,
        },
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "append_backend": args.append_backend,
        "cuda_append_extension_requested": (
            args.append_backend == "cuda"
            or os.environ.get("KOTFLASH_ENABLE_CUDA_APPEND") == "1"
        ),
        "preallocated_cache_capacity": cache_capacity,
        "preallocated_cache_layout": "strided prefix view; unused capacity follows the visible prefix",
        "cache_append_only": measure(
            append_only,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        ),
        "correctness": correctness,
        "results": {},
        "interpretation": (
            "tokens_per_second is batch throughput for one decode position per batch "
            "and excludes model projections, MLP, sampling and host-side scheduling. "
            "legacy append_plus_attention variants include torch.cat; preallocated "
            "variants append into fixed storage. GPU event time and synchronized "
            "wall-clock time are reported separately."
        ),
    }

    for name, (attention_fn, decode_fn) in candidates.items():
        attention_result = measure(
            attention_fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        decode_result = measure(
            decode_fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        decode_result["batch_tokens_per_second"] = (
            args.batch * 1000.0 / decode_result["wall_median_ms"]
        )
        report["results"][name] = {
            "attention_only": attention_result,
            "append_plus_attention": decode_result,
        }

    baseline = report["results"]["pytorch_expanded_sdpa"]
    for name, values in report["results"].items():
        for metric in ("attention_only", "append_plus_attention"):
            values[metric]["gpu_speedup_vs_expanded_sdpa"] = (
                baseline[metric]["gpu_median_ms"] / values[metric]["gpu_median_ms"]
            )
            values[metric]["wall_speedup_vs_expanded_sdpa"] = (
                baseline[metric]["wall_median_ms"] / values[metric]["wall_median_ms"]
            )

    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
