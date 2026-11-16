#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark: 100 TinyStories language-model training steps with SDPA vs KOTFlash.

The dataset/tokenizer pipeline runs before timing. Both backends use identical
initial weights and paired token batches; backend order is shuffled per step.
This is a compact 2-layer GPT-style LM proxy, not KOT-V2-2B.

Example:
  python benchmarks/benchmark_tinystories_training.py --seq 4096 --batch 1 --steps 100
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from kotflash import attention

DATASET_NAME = "skeskinen/TinyStories-hf"
DEFAULT_TOKENIZER = "gpt2"


class DecoderLayer(nn.Module):
    """Pre-norm causal GQA decoder layer with a SwiGLU MLP."""

    def __init__(self, hidden: int, q_heads: int, kv_heads: int,
                 head_dim: int, mlp_ratio: int = 4):
        super().__init__()
        kv_width = kv_heads * head_dim
        self.q_heads, self.kv_heads, self.head_dim = q_heads, kv_heads, head_dim
        self.norm1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, hidden * mlp_ratio, bias=False)
        self.up_proj = nn.Linear(hidden, hidden * mlp_ratio, bias=False)
        self.down_proj = nn.Linear(hidden * mlp_ratio, hidden, bias=False)

    def forward(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        batch, seq, _ = x.shape
        residual = x
        normalized = self.norm1(x)
        q = self.q_proj(normalized).view(
            batch, seq, self.q_heads, self.head_dim
        ).transpose(1, 2).to(torch.float16)
        k = self.k_proj(normalized).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2).to(torch.float16)
        v = self.v_proj(normalized).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2).to(torch.float16)

        if backend == "triton":
            attended = attention(q, k, v, causal=True, backend="triton")
        elif backend == "sdpa_expanded":
            groups = self.q_heads // self.kv_heads
            attended = F.scaled_dot_product_attention(
                q, k.repeat_interleave(groups, dim=1),
                v.repeat_interleave(groups, dim=1), is_causal=True,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        attended = attended.transpose(1, 2).contiguous().view(batch, seq, -1)
        x = residual + self.o_proj(attended.to(dtype=normalized.dtype))
        residual = x
        normalized = self.norm2(x)
        mlp = F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return residual + self.down_proj(mlp)


class TinyStoriesLM(nn.Module):
    """Small GPT-style next-token model with tied token embedding / LM head."""

    def __init__(self, vocab_size: int, max_seq: int, layers: int = 2,
                 q_heads: int = 8, kv_heads: int = 2, head_dim: int = 64,
                 mlp_ratio: int = 4):
        super().__init__()
        if q_heads % kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        hidden = q_heads * head_dim
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Embedding(max_seq, hidden)
        self.blocks = nn.ModuleList([
            DecoderLayer(hidden, q_heads, kv_heads, head_dim, mlp_ratio)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)
        self.max_seq = max_seq
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, backend: str) -> torch.Tensor:
        _, seq = input_ids.shape
        if seq > self.max_seq:
            raise ValueError(f"sequence {seq} exceeds max_seq {self.max_seq}")
        positions = torch.arange(seq, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x, backend)
        return F.linear(self.final_norm(x), self.token_embedding.weight)


def load_tinystories_tokens(tokenizer_name: str, train_tokens_needed: int,
                            validation_tokens_needed: int, max_examples: int):
    """Load story-aligned train and validation examples from TinyStories-hf."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install data dependencies: pip install 'datasets>=3,<5' 'transformers>=4.45,<5'"
        ) from exc

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name!r} has no EOS token")

    def collect(split: str, required: int, limit: int):
        stream = load_dataset(DATASET_NAME, split=split, streaming=True)
        ids: list[int] = []
        examples = 0
        for example in stream:
            text = example.get("text")
            if isinstance(text, str) and text.strip():
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if encoded:
                    ids.extend(encoded)
                    ids.append(eos_id)
            examples += 1
            if len(ids) >= required or examples >= limit:
                break
        if len(ids) < required:
            raise RuntimeError(
                f"TinyStories {split}: got {len(ids):,} tokens, need {required:,}; "
                "increase --max-examples"
            )
        return torch.tensor(ids[:required], dtype=torch.long), examples, len(ids)

    train, train_examples, train_loaded = collect(
        "train", train_tokens_needed, max_examples
    )
    valid, valid_examples, valid_loaded = collect(
        "validation", validation_tokens_needed, max(1000, max_examples // 10)
    )
    return train, valid, {
        "dataset": DATASET_NAME,
        "train_split": "train",
        "validation_split": "validation",
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "train_examples_read": train_examples,
        "validation_examples_read": valid_examples,
        "train_tokens_loaded": train_loaded,
        "validation_tokens_loaded": valid_loaded,
        "preprocessing_seconds": time.perf_counter() - started,
    }


def build_batches(tokens: torch.Tensor, batch: int, seq: int, count: int,
                  device: torch.device, seed: int):
    stride = batch * seq
    windows = tokens[:count * stride + 1].unfold(0, stride + 1, stride)
    if windows.shape[0] != count:
        raise RuntimeError("Could not build the requested token windows")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    windows = windows[torch.randperm(count, generator=generator)]
    x = windows[:, :-1].reshape(count, batch, seq).to(device)
    y = windows[:, 1:].reshape(count, batch, seq).to(device)
    return x, y


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def evaluate_loss(model, x, y, backend):
    model.eval()
    with torch.no_grad(), amp_context():
        logits = model(x, backend)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        value = float(loss.float().item())
    del logits, loss
    return value


def train_step(model, optimizer, scaler, x, y, backend):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context():
        logits = model(x, backend)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    del logits
    return loss.detach()


def finite_after_training(model, loss) -> bool:
    if not bool(torch.isfinite(loss).all().item()):
        return False
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


def benchmark(args) -> dict[str, Any]:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose a Tesla T4 runtime")
    gpu = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if args.require_t4 and (capability != (7, 5) or "T4" not in gpu):
        raise RuntimeError(f"Expected Tesla T4 / SM75, got {gpu} / {capability}")

    train_count = args.warmup + args.steps
    train_needed = train_count * args.batch * args.seq + 1
    valid_needed = args.batch * args.seq + 1
    train_tokens, valid_tokens, data = load_tinystories_tokens(
        args.tokenizer, train_needed, valid_needed, args.max_examples
    )
    train_x, train_y = build_batches(
        train_tokens, args.batch, args.seq, train_count, device, args.seed
    )
    valid_window = valid_tokens[:valid_needed]
    eval_x = valid_window[:-1].reshape(args.batch, args.seq).to(device)
    eval_y = valid_window[1:].reshape(args.batch, args.seq).to(device)

    torch.manual_seed(args.seed)
    template = TinyStoriesLM(
        data["vocab_size"], args.seq, args.layers, args.q_heads,
        args.kv_heads, args.head_dim, args.mlp_ratio,
    )
    initial_state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    parameter_count = sum(p.numel() for p in template.parameters())
    del template

    models, optimizers, scalers = {}, {}, {}
    for backend in ("sdpa_expanded", "triton"):
        model = TinyStoriesLM(
            data["vocab_size"], args.seq, args.layers, args.q_heads,
            args.kv_heads, args.head_dim, args.mlp_ratio,
        ).to(device=device, dtype=torch.float32)
        model.load_state_dict(initial_state)
        models[backend] = model
        optimizers[backend] = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scalers[backend] = make_scaler()

    initial_losses = {
        backend: evaluate_loss(models[backend], eval_x, eval_y, backend)
        for backend in ("sdpa_expanded", "triton")
    }
    loss_delta = abs(initial_losses["sdpa_expanded"] - initial_losses["triton"])
    torch.cuda.synchronize(device)

    rng = random.Random(args.seed + 1)
    for step in range(args.warmup):
        order = ["sdpa_expanded", "triton"]
        rng.shuffle(order)
        for backend in order:
            train_step(
                models[backend], optimizers[backend], scalers[backend],
                train_x[step], train_y[step], backend,
            )
    torch.cuda.synchronize(device)

    events = {"sdpa_expanded": [], "triton": []}
    last_losses, orders = {}, []
    wall_started = time.perf_counter()
    for step in range(args.steps):
        order = ["sdpa_expanded", "triton"]
        rng.shuffle(order)
        orders.append(order)
        x, y = train_x[args.warmup + step], train_y[args.warmup + step]
        for backend in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            last_losses[backend] = train_step(
                models[backend], optimizers[backend], scalers[backend], x, y, backend
            )
            end.record()
            events[backend].append((start, end))
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_started) * 1000.0

    tokens_per_backend = args.steps * args.batch * args.seq
    results = {}
    for backend in ("sdpa_expanded", "triton"):
        times = [start.elapsed_time(end) for start, end in events[backend]]
        total_ms = sum(times)
        results[backend] = {
            "median_step_ms": statistics.median(times),
            "mean_step_ms": statistics.mean(times),
            "p10_step_ms": sorted(times)[max(0, int(0.10 * (len(times) - 1)))],
            "p90_step_ms": sorted(times)[min(len(times) - 1, int(0.90 * (len(times) - 1)))],
            "sum_gpu_step_ms": total_ms,
            "training_tokens_per_second": tokens_per_backend * 1000.0 / total_ms,
            "initial_eval_loss": initial_losses[backend],
            "final_eval_loss": evaluate_loss(models[backend], eval_x, eval_y, backend),
            "last_train_loss": float(last_losses[backend].float().item()),
            "finite_after_training": finite_after_training(models[backend], last_losses[backend]),
            "step_times_ms": times,
        }

    baseline_ms = results["sdpa_expanded"]["sum_gpu_step_ms"]
    triton_ms = results["triton"]["sum_gpu_step_ms"]
    return {
        "benchmark": "TinyStories next-token LM training A/B",
        "gpu": gpu,
        "compute_capability": capability,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "shape": {
            "batch": args.batch, "sequence": args.seq, "layers": args.layers,
            "q_heads": args.q_heads, "kv_heads": args.kv_heads,
            "head_dim": args.head_dim, "hidden_size": args.q_heads * args.head_dim,
            "vocab_size": data["vocab_size"], "parameters": parameter_count,
        },
        "precision": "FP32 master weights + CUDA autocast FP16 + GradScaler; FP16 attention",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps_per_backend": args.warmup,
        "measured_steps_per_backend": args.steps,
        "training_tokens_per_backend": tokens_per_backend,
        "paired_batches_and_interleaved_order": True,
        "measurement_order_by_step": orders,
        "dataset": data,
        "initial_eval_loss_abs_delta": loss_delta,
        "initial_eval_loss_parity_passed": loss_delta <= args.max_initial_loss_delta,
        "max_initial_loss_delta": args.max_initial_loss_delta,
        "paired_wall_time_ms": wall_ms,
        "results": results,
        "speedup_triton_vs_expanded_sdpa": baseline_ms / triton_ms if triton_ms else None,
        "interpretation": (
            "CUDA-event times cover full LM forward, cross-entropy, backward and AdamW. "
            "Dataset streaming/tokenization and kernel warmup are excluded. Both backends "
            "use identical initial weights and paired TinyStories batches; evaluation uses "
            "the separate validation split. This is a 2-layer GPT-style proxy, not KOT-V2-2B."
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-initial-loss-delta", type=float, default=0.02)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-t4", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if min(args.seq, args.batch, args.layers, args.q_heads, args.kv_heads,
           args.head_dim, args.mlp_ratio, args.steps, args.max_examples) <= 0:
        parser.error("all sizes and step counts must be positive")
    if args.warmup < 1 or args.q_heads % args.kv_heads:
        parser.error("warmup must be positive and q-heads divisible by kv-heads")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_initial_loss_delta < 0:
        parser.error("invalid optimizer/parity settings")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = benchmark(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["initial_eval_loss_parity_passed"]:
        raise SystemExit("Initial validation loss mismatch; do not interpret speed.")
    if not all(item["finite_after_training"] for item in report["results"].values()):
        raise SystemExit("Non-finite loss or parameters detected; result is invalid.")


if __name__ == "__main__":
    main()
