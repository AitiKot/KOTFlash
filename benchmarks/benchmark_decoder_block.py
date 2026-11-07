#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare SDPA and KOTFlash inside a complete decoder-block training step.

Run after installing KOTFlash in a CUDA environment:
    python benchmarks/benchmark_decoder_block.py --batch 2 --seq 1024 --q-heads 16 --kv-heads 4 --head-dim 64
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch
from torch import nn
import torch.nn.functional as F

from kotflash import attention


class TinyDecoderBlock(nn.Module):
    """Small pre-norm decoder block with GQA/MQA-compatible projections."""

    def __init__(self, q_heads: int, kv_heads: int, head_dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = q_heads * head_dim
        kv_width = kv_heads * head_dim
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        self.norm1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        self.norm2 = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, hidden * mlp_ratio, bias=False)
        self.up_proj = nn.Linear(hidden, hidden * mlp_ratio, bias=False)
        self.down_proj = nn.Linear(hidden * mlp_ratio, hidden, bias=False)

    def forward(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        batch, seq, hidden = x.shape
        normalized = self.norm1(x)

        q = self.q_proj(normalized).view(
            batch, seq, self.q_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(normalized).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normalized).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        attended = attention(q, k, v, causal=True, backend=backend)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, hidden)
        x = x + self.o_proj(attended)

        normalized = self.norm2(x)
        mlp = F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return x + self.down_proj(mlp)


def train_step(model, optimizer, x, target, backend: str):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x, backend)
    loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()
    optimizer.step()
    return loss.detach()


def benchmark_backend(
    backend: str,
    initial_state,
    x: torch.Tensor,
    target: torch.Tensor,
    args,
):
    model = TinyDecoderBlock(
        args.q_heads, args.kv_heads, args.head_dim
    ).to(device=x.device, dtype=x.dtype)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warm up kernels and lazy optimizer state before measuring memory/time.
    for _ in range(args.warmup):
        train_step(model, optimizer, x, target, backend)
    torch.cuda.synchronize()

    baseline_memory = torch.cuda.memory_allocated(x.device)
    torch.cuda.reset_peak_memory_stats(x.device)
    timings = []
    last_loss = None
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_loss = train_step(model, optimizer, x, target, backend)
        end.record()
        timings.append((start, end))

    torch.cuda.synchronize()
    elapsed_ms = [start.elapsed_time(end) for start, end in timings]
    median_ms = statistics.median(elapsed_ms)
    peak_extra = max(
        0, torch.cuda.max_memory_allocated(x.device) - baseline_memory
    )
    tokens_per_step = x.shape[0] * x.shape[1]
    return {
        "median_step_ms": median_ms,
        "tokens_per_second": tokens_per_step * 1000.0 / median_ms,
        "incremental_peak_allocated_bytes": peak_extra,
        "last_loss": float(last_loss.item()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a complete decoder-block training step on CUDA."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if min(args.batch, args.seq, args.q_heads, args.kv_heads, args.head_dim) <= 0:
        parser.error("batch, sequence length, head counts and head dimension must be positive")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.warmup < 1 or args.iterations < 3:
        parser.error("use at least one warmup and three measured iterations")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden = args.q_heads * args.head_dim
    x = torch.randn(args.batch, args.seq, hidden, device=device, dtype=dtype)
    target = torch.randn_like(x)

    template = TinyDecoderBlock(
        args.q_heads, args.kv_heads, args.head_dim
    )
    initial_state = {
        key: value.detach().clone()
        for key, value in template.state_dict().items()
    }

    # Compare the two implementations before any optimizer updates.
    sdpa_model = TinyDecoderBlock(
        args.q_heads, args.kv_heads, args.head_dim
    ).to(device=device, dtype=dtype)
    sdpa_model.load_state_dict(initial_state)
    triton_model = TinyDecoderBlock(
        args.q_heads, args.kv_heads, args.head_dim
    ).to(device=device, dtype=dtype)
    triton_model.load_state_dict(initial_state)

    report = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "shape": {
            "batch": args.batch,
            "sequence": args.seq,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "hidden_size": hidden,
        },
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    triton_available = True
    try:
        with torch.no_grad():
            reference = sdpa_model(x, "sdpa")
            candidate = triton_model(x, "triton")
            torch.cuda.synchronize()
            report["max_abs_output_error_vs_sdpa"] = (
                candidate.float() - reference.float()
            ).abs().max().item()
    except Exception as exc:
        triton_available = False
        report["triton"] = {
            "unavailable_or_failed": f"{type(exc).__name__}: {exc}"
        }

    del sdpa_model, triton_model, template
    torch.cuda.empty_cache()

    backends = ("sdpa", "triton") if triton_available else ("sdpa",)
    for backend in backends:
        try:
            report[backend] = benchmark_backend(
                backend, initial_state, x, target, args
            )
        except Exception as exc:
            report[backend] = {
                "unavailable_or_failed": f"{type(exc).__name__}: {exc}"
            }
        torch.cuda.empty_cache()

    sdpa_ms = report.get("sdpa", {}).get("median_step_ms")
    triton_ms = report.get("triton", {}).get("median_step_ms")
    if sdpa_ms and triton_ms:
        report["training_step_speedup_vs_sdpa"] = sdpa_ms / triton_ms

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
