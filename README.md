# KOTFlash

**Experimental memory-efficient attention for NVIDIA Turing (SM75), exposed through a general PyTorch API.**

KOTFlash is an independent, research-stage implementation with a PyTorch SDPA path and experimental fused Triton online-softmax forward/backward kernels for NVIDIA Turing (SM75). The custom path has passed user-run correctness tests on a Tesla T4 and has shown gains for selected attention and synthetic decoder-block shapes. It remains experimental: performance varies by sequence length, and full-model training speedup has not been established.

> **Status: alpha / user-reported Tesla T4 validation: 42/42 tests passed on PyTorch 2.6.0+cu124 and Triton 3.2.0, including exact `auto` routing and long-sequence split-backward cases.** The physical GPU suite is user-run; run it again on your own hardware before using KOTFlash for training.

## What works today

| Component | Status |
|---|---|
| PyTorch SDPA backend | Implemented; CPU and CUDA where PyTorch supports it |
| MHA / GQA / MQA API | Implemented |
| Triton online-softmax forward | Experimental; FP16, SM75, Triton 3.2.x, head_dim 16–128 |
| Causal / non-causal | Implemented in both paths; upper-left causal alignment |
| Head dimensions 72 and 96 | Split into 64 + tail tiles to reduce padding versus a full 128-wide dot; still requires T4 profiling |
| Backward | Fused Triton dQ/dK/dV; short and long split-query paths passed reported T4 correctness tests |
| Physical T4 validation | User-reported Tesla T4 run: 42/42 tests passed, including five `auto` profile lengths and long split-query gradients |
| Fused SM75 backward | Both short-sequence and long split-query dQ/dK/dV paths passed T4 correctness tests; speed remains shape-dependent |
| Packed variable-length sequences | Not implemented |
| Dropout, local window, KV cache | Not implemented |

The custom forward does not materialize the full attention matrix. Its online softmax maintains a running maximum, denominator and output accumulator while visiting K/V tiles. It stores the output and row log-sum-exp for the current reference backward.

## Installation

Install a CUDA-enabled PyTorch build that matches your driver/runtime first. KOTFlash does not install, replace or downgrade PyTorch automatically.

```bash
git clone https://github.com/AitiKot/KOTFlash.git
cd KOTFlash
python -m pip install -e .
```

The SDPA backend does not require Triton. For the upstream Turing-compatible Triton line, install the optional dependency separately:

```bash
python -m pip install 'triton>=3.2,<3.3'
```

**Important compatibility note:** upstream Triton 3.3 and later dropped Volta/Turing support. The upstream 3.2 line is therefore the conservative target for SM75. PyTorch versions that bundle a newer Triton may need a separate compatible environment; do not blindly downgrade Triton in an existing environment without checking the PyTorch/Triton pairing. A maintained Turing-enabled downstream build may also work, but is not automatically detected by KOTFlash.

Install test dependencies:

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Quick start

```python
import torch
from kotflash import attention

# [batch, heads, sequence, head_dim]
q = torch.randn(2, 16, 1024, 72, device='cuda', dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 4, 1024, 72, device='cuda', dtype=torch.float16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)

# Conservative default: use the installed PyTorch SDPA backend.
y = attention(q, k, v, causal=True, backend='auto')

# Explicitly opt in to the experimental SM75 Triton kernel.
y_triton = attention(q, k, v, causal=True, backend='triton')
y.float().square().mean().backward()
```

Select a backend explicitly:

```python
y = attention(q, k, v, backend='sdpa')   # portable PyTorch reference/fallback
y = attention(q, k, v, backend='triton') # fail loudly if prerequisites are absent
```

## API

```python
attention(q, k, v, *, causal=False, scale=None, backend='auto')
```

- `q`: `[B, Hq, Sq, D]`
- `k`, `v`: `[B, Hkv, Sk, D]`
- `Hq % Hkv == 0` for GQA/MQA
- `causal=True` applies the upper-left mask `key_index <= query_index`, matching PyTorch SDPA semantics for unequal sequence lengths
- `scale=None` uses `1 / sqrt(D)`
- `backend`: `auto`, `triton` or `sdpa`

The experimental Triton path currently requires CUDA SM75, FP16, upstream Triton 3.2.x, and `16 <= D <= 128`. Non-contiguous inputs are copied to contiguous buffers before the custom kernel. `auto` uses a narrow T4 profile only on the measured PyTorch 2.6.0+cu124 / Triton 3.2.x stack and exact profiled shapes; every other shape or software stack falls back to PyTorch SDPA.

## Tests and benchmark

Run CPU/API tests:

```bash
pytest -q
```

The T4 integration tests are skipped unless CUDA, Triton 3.2.x and SM75 are present. On a physical T4, run:

```bash
pytest -q tests/test_triton.py
python benchmarks/benchmark_attention.py --device cuda --batch 2 --q-heads 16 --kv-heads 4 --seq 1024 --head-dim 72 --causal
```

The benchmark compares four paths on the same input: native PyTorch GQA SDPA, PyTorch SDPA with K/V heads expanded, Triton and auto. The expanded-KV baseline includes the cost of repeat_interleave, so it is a fair end-to-end alternative when native GQA selects a slower backend. All backends are warmed before timing; their order is shuffled reproducibly between rounds to reduce first/last-position clock bias. The report shows per-round values, median/min/max latency, forward and forward+backward time, incremental peak allocated memory, maximum absolute error and speedup ratios against both PyTorch baselines. The numerical reference is kept outside the measured autograd path. Use at least 5 repeats (the default) and record GPU model, driver, CUDA runtime, PyTorch and Triton versions with published results. Benchmark results from other GPUs are not T4 results.
For the exact automatic T4 profile, run the following sequence sweep (the benchmark defaults to 5 interleaved rounds):

```bash
for seq in 128 256 512 1024 2048; do
  python benchmarks/benchmark_attention.py --device cuda --batch 2 --q-heads 8 --kv-heads 2 --seq "$seq" --head-dim 64 --causal --warmup 20 --iterations 50 --repeats 5
done
```


To measure whether KOTFlash helps an actual training step (rather than attention in isolation), run the decoder-block benchmark:

```bash
python benchmarks/benchmark_decoder_block.py --batch 2 --seq 512 --q-heads 8 --kv-heads 2 --head-dim 64
```

It includes Q/K/V/output projections, attention, an MLP, forward, backward and an AdamW update. Model parameters and surrounding block operations remain FP32; Q/K/V are cast to FP16 for both attention implementations, avoiding direct FP16 AdamW master weights. The report compares KOTFlash with both native PyTorch GQA SDPA and SDPA with KV heads expanded (which can select a faster T4 backend). It includes median step time, tokens/second, incremental peak memory, output error and finite-loss/parameter checks. This is a small decoder-block test, not a full language-model training result; use your own model and sequence lengths for the final decision.


For a T4-specific forward tile sweep, run:

```bash
python benchmarks/tune_triton_tiles.py --seq 1024 --head-dim 72 --causal
```

The tuner measures combinations of `BM={32,64}`, `BN={32,64,128}`, `num_warps={4,8}` and `num_stages={1,2}`, compares them with SDPA, and prints numerical error. Its JSON report now records every attempted, successful and failed configuration, including the compiler/resource error for failed tiles. On the first user-reported T4 sweep, `BM=64, BN=32` reached 1.118× SDPA for B=2/Hq=16/Hkv=4/SEQ=1024/D=72 causal, and 1.075× for SEQ=512/D=64 non-causal. These are individual measurements, not a general guarantee.


## User-reported Tesla T4 benchmark matrix (2026-11-08)
Environment: Tesla T4 / SM75, PyTorch 2.6.0+cu124, Triton 3.2.0, B=2, Hq=8, Hkv=2, D=64, causal attention.

| Sequence | Full decoder SDPA best path | KOTFlash decoder step | KOTFlash vs best SDPA |
|---:|---:|---:|---:|
| 256 | 5.468 ms (native GQA) | 4.631 ms | 1.18x faster |
| 512 | 9.345 ms (expanded KV) | 8.759 ms | 1.07x faster |
| 1024 | 16.683 ms (expanded KV) | 16.763 ms | 0.995x (effectively tied) |

Standalone attention forward+backward versus expanded-KV SDPA measured 0.95x, 0.86x, 0.89x, 0.56x and 0.55x at sequence lengths 128, 256, 512, 1024 and 2048 respectively. This identifies long-sequence dK/dV backward parallelism as the next optimization target. Forward-only Triton became faster than expanded-KV SDPA at 1024 (1.34x) and 2048 (2.04x), with lower peak allocations in all tested cases. These are single-run, user-reported measurements of the previous serial dK/dV implementation; the new split-query reduction has not yet been measured on T4. The decoder benchmark is a synthetic single-block training step, not a full-model result.

## Latest physical T4 validation (2026-11-08)
A user-run Colab validation on Tesla T4 / SM75, PyTorch 2.6.0+cu124 and Triton 3.2.0 completed:
- Full suite: **42 passed, 0 failed, 0 skipped**.
- The five auto-profile lengths (128, 256, 512, 1024 and 2048) passed numerical comparison against expanded-KV SDPA.
- Explicit routing assertions confirmed expanded-SDPA dispatch for lengths 128–512 and Triton dispatch for 1024–2048.
- Long split-query backward cases passed for 1024-token causal GQA, unequal-length MQA and 2048-token causal GQA.

The benchmark has since been changed to warm all backends before timing and shuffle backend order reproducibly across five rounds. The earlier sequential timings showed a discrepancy between direct Triton and auto even though both selected the same Triton path; those numbers are not considered a stable performance result. Re-run the interleaved benchmark on a physical T4 before publishing fresh speedup claims. This validation establishes correctness and dispatch, not full-model training speedup.

## Physical T4 validation of split-query backward (2026-11-08)
User-run on Tesla T4 / SM75, PyTorch 2.6.0+cu124, Triton 3.2.0, B=2, Hq=8, Hkv=2, D=64, causal:
- Full GPU suite: **37 passed, 0 failed, 0 skipped**. The three added long-sequence gradient cases (1024-token GQA, unequal-length MQA, and 2048-token causal attention) passed against PyTorch.
- Standalone attention forward+backward vs expanded-KV SDPA: SEQ=512 1.002 vs 0.894 ms (Triton 0.89x); SEQ=1024 1.029 vs 1.386 ms (**1.35x**); SEQ=2048 3.588 vs 4.007 ms (**1.12x**).
- Standalone forward at SEQ=1024 and 2048: **1.70x** and **2.11x** faster than expanded-KV SDPA.
- Synthetic decoder-block training step: SEQ=512 8.63 vs 12.40 ms (**1.44x**, but the SDPA baseline varied noticeably from the prior run); SEQ=1024 16.03 vs 16.05 ms (effectively tied); SEQ=2048 31.91 vs 31.17 ms (KOTFlash about 2.4% slower).
- Maximum output error in the decoder-block comparison stayed below 6.1e-5. Losses and model parameters remained finite.
- Peak incremental allocation at SEQ=2048: 42.1 MB for Triton forward+backward vs 48.4 MB for expanded-KV SDPA.
These are user-reported single-run results for a synthetic one-block training benchmark, not a full-model guarantee. The long-sequence split path is now correctness-tested on T4; further tuning is needed to turn its attention-level gain into a consistent whole-block training speedup.

## Automatic T4 backend profile
`backend="auto"` uses a deliberately narrow profile measured on **Tesla T4 / SM75**, PyTorch **2.6.0+cu124** and Triton **3.2.x**. The profile only matches contiguous FP16 inputs with B=2, Hq=8, Hkv=2, D=64, causal self-attention, default scale 0.125 and Sq=Sk in {128, 256, 512, 1024, 2048}. For 128–512 tokens it expands K/V and uses PyTorch SDPA; for 1024–2048 it selects fused Triton. GPU tests assert both the selected route and numerical output. Other shapes, devices, PyTorch versions, CUDA versions and non-contiguous inputs retain the conservative PyTorch SDPA fallback. This is not a universal autotuner; explicit `backend="triton"` and `backend="sdpa"` remain available. Re-measure the profile when changing the software stack.

## Architecture notes

- Forward uses online softmax and a tiled Q × Kᵀ / P × V loop. The full `[Sq, Sk]` attention matrix is never stored.
- Causal forward skips complete future K/V tiles instead of loading and masking them.
- Head dimensions 65–127 are split into a 64-wide dot plus a padded tail dot, avoiding a single full 128-wide dot for D=72/96.
- Accumulators and log-sum-exp are FP32; inputs and output are FP16 in the custom path.
- GQA maps each query head to its corresponding KV head group.
- The experimental fused backward uses Triton kernels for dQ and dK/dV. The original short-sequence dK/dV path assigns one CTA to each KV tile and walks query tiles serially. A new long-sequence path (Sq >= 1024) splits query tiles into independent chunks, writes FP32 partial dK/dV tensors and reduces them in a second kernel. This avoids atomics and increases parallel work on Turing. The split path passed the expanded physical-T4 correctness suite (37/37 tests, including 1024- and 2048-token GQA/MQA cases). Its measured T4 speed and limitations are recorded below. The earlier batched PyTorch implementation remains available internally as a correctness reference.
- The custom kernel currently targets the upstream Triton 3.2 Turing backend. The current experimental launch defaults are BM=64, BN=32, 4 warps and 2 stages, based on the first T4 tile sweep; keep benchmarking new shapes before treating them as universal defaults.

## Roadmap

- [x] Public PyTorch API and SDPA fallback
- [x] Experimental tiled online-softmax forward
- [x] MHA/GQA/MQA indexing
- [x] Batched blockwise recomputation backward for validation
- [x] CPU/API tests and SM75 GPU test suite
- [x] Run the Triton integration suite on a physical T4 (11/11 user-reported tests passed)
- [x] Initial fused SM75 dQ/dK/dV backward implementation; earlier revision passed T4 tests
- [x] Validate split-query dK/dV reduction correctness on a physical T4 (user-reported 37/37 suite)
- [ ] Tune long-sequence split count and partial-gradient reduction overhead on T4
- [ ] Complete tile/warp/stage sweeps across representative T4 workloads
- [x] Add causal future-tile skipping
- [x] Add split-dot path for head dimensions 72/96
- [ ] Profile and tune the split-dot path on T4
- [ ] Variable-length packed sequences
- [ ] Dropout, local-window attention and KV-cache variants
- [ ] More PyTorch / Triton compatibility checks

## Related work and dependencies

KOTFlash is an independent implementation and does not vendor source from these projects. They are acknowledged as related work and compatibility context:

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — IO-aware exact attention algorithms.
- [FlashAttention Turing](https://github.com/ssiu/flash-attention-turing) — Turing-oriented implementation with forward/backward, GQA and additional features.
- [flash-attn-turing](https://github.com/BhoumikPatidar/flash_attn_turing) — Triton-based Turing forward implementation and fused RoPE experiments.
- [Triton](https://github.com/triton-lang/triton) — optional separately installed compiler/runtime.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency and attribution notes.

## License and patent note

KOTFlash is distributed under the **Apache License 2.0**. It permits use, modification, redistribution and commercial use subject to its terms, and includes a limited patent license from contributors for patent claims necessarily infringed by their contributions. See [LICENSE](LICENSE).

This project has not undergone a formal patent search, freedom-to-operate analysis, or legal review. The license does not guarantee that every possible use is free of third-party patent claims. No third-party attention-kernel source is intentionally included in this repository; external projects are references only. Contributions are accepted under Apache-2.0; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Correctness comes before speed. Include tests for numerical equivalence and gradients, and never publish performance claims without reproducible device/software details. See [CONTRIBUTING.md](CONTRIBUTING.md).
