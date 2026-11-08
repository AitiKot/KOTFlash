# KOTFlash

**Experimental memory-efficient attention for NVIDIA Turing (SM75), exposed through a general PyTorch API.**

KOTFlash is an independent, research-stage implementation with a PyTorch SDPA path and experimental fused Triton online-softmax forward/backward kernels for NVIDIA Turing (SM75). It has measured modest gains for selected long self-attention workloads and now includes an experimental split-KV path for autoregressive decode. Full decoder-block and full-model training speedups have not been established.

> **Status: alpha / split-KV kernel correctness: 52/52 tests passed on physical Tesla T4 / PyTorch 2.6.0+cu124 / Triton 3.2.0.** Three additional tests for the newly added automatic decode routing still need a physical-T4 rerun. The decode benchmark shows shape-dependent results; see the measured table below. Full-model generation speedup has not been measured.

## What works today

| Component | Status |
|---|---|
| PyTorch SDPA backend | Implemented; CPU and CUDA where PyTorch supports it |
| MHA / GQA / MQA API | Implemented |
| Triton online-softmax forward | Experimental; FP16, SM75, Triton 3.2.x, head_dim 16–128 |
| Causal / non-causal | Implemented in both paths; upper-left causal alignment |
| Head dimensions 72 and 96 | Split into 64 + tail tiles to reduce padding versus a full 128-wide dot; still requires T4 profiling |
| Backward | Fused Triton dQ/dK/dV; short and long split-query paths passed reported T4 correctness tests |
| Physical T4 validation | Split-KV implementation: user-reported Tesla T4 run **52/52 passed**; new auto-routing tests await rerun |
| Fused SM75 backward | Both short-sequence and long split-query dQ/dK/dV paths passed T4 correctness tests; speed remains shape-dependent |
| Packed variable-length sequences | Not implemented |
| Long-context decode | Experimental split-KV attention for Q lengths 1–16 over caller-supplied visible K/V contexts ≥1024 tokens (D≤64); no cache manager or paging |

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

## Long-context autoregressive decode (experimental)

KOTFlash includes a split-KV Triton forward path for decoding when the query is short (1–16 positions) and the supplied K/V context is long (at least 1024 positions, head dimension up to 64). It splits the visible K/V cache into independent chunks, computes partial attention results in parallel, and merges them with log-sum-exp weights. This is designed to improve GPU occupancy for single-token or short-chunk generation on Turing GPUs.

For one-token generation, pass only the prefix and current token's K/V entries that the new query is allowed to attend to. Since all supplied keys are already visible to that query, use `causal=False`:

```python
import torch
from kotflash import attention

# q: [B, Hq, 1, D] for the next token
# k_cache, v_cache: [B, Hkv, past_tokens + 1, D]
# The cache contains only the visible prefix/current-token positions.
next_token_context = attention(
    q, k_cache, v_cache,
    causal=False,
    backend="triton",
)
```

The split-KV route is selected inside the explicit Triton backend when `Sq <= 16`, `Sk >= 1024`, `D <= 64`, and causal masking is disabled. It returns the same output and log-sum-exp representation used by the fused backward. The library does not allocate, append to, or manage a model's KV cache. Do not use this non-causal mode when the supplied K/V contains keys that should be masked from any query; use the ordinary causal path instead.

On the measured Tesla T4 / PyTorch 2.6.0+cu124 / Triton 3.2.0 stack, the short-query decode benchmark produced these forward medians (B=2, Hq=8, Hkv=2, D=64, FP16; five interleaved rounds):

| Q / KV | Native GQA SDPA | Expanded-KV SDPA | Triton Split-KV | Triton vs expanded |
|---:|---:|---:|---:|---:|
| 1 / 2048 | 0.358 ms | 0.264 ms | 0.240 ms | **1.10x** |
| 4 / 1024 | 0.374 ms | 0.174 ms | 0.240 ms | 0.73x |
| 16 / 4096 | 1.035 ms | 0.447 ms | 0.235 ms | **1.90x** |

The native GQA path was slower than expanded-KV SDPA in these runs, so expanded-KV SDPA is the stronger comparison. Split-KV wins at Q=1 and Q=16 but loses at Q=4. Accordingly, the measured `auto` profile routes only the exact tested shapes: (Q,KV)=(1,2048) to Triton, (4,1024) to expanded-KV SDPA, and (16,4096) to Triton. Other decode shapes and software stacks retain the conservative fallback. These are attention-only timings and must not be presented as end-to-end model tokens/second.

Benchmark this decode case on a physical T4:

```bash
python benchmarks/benchmark_attention.py --device cuda --batch 2 --q-heads 8 --kv-heads 2 --seq 1 --kv-seq 2048 --head-dim 64 --warmup 20 --iterations 50 --repeats 5
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

The experimental Triton path currently requires CUDA SM75, FP16, upstream Triton 3.2.x, and `16 <= D <= 128`. Non-contiguous inputs are copied to contiguous buffers before the custom kernel. `auto` uses narrow T4 profiles only on the measured PyTorch 2.6.0+cu124 / Triton 3.2.x stack and exact profiled shapes: causal training attention at Sq=Sk 128..2048, plus three non-causal decode cases described above. Every other shape or software stack falls back to PyTorch SDPA.

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


## Current physical T4 validation and benchmark (2026-11-08)
User-run Colab validation on Tesla T4 / SM75, PyTorch 2.6.0+cu124, CUDA 12.4 and Triton 3.2.0. Workload: B=2, Hq=8, Hkv=2, D=64, FP16 causal self-attention.

- Full current-branch suite: **48 passed, 0 failed, 0 skipped** in 3.85 seconds.
- The five profiled sequence lengths passed numerical comparison against expanded-KV SDPA.
- Route tests confirmed expanded-KV SDPA at sequence lengths 128/256/512 and fused Triton at 1024/2048.
- Long split-query backward cases passed: 1024-token causal GQA, unequal-length MQA (Sk=768), and 2048-token causal GQA.

The updated benchmark warmed all paths first, randomized backend order reproducibly over five rounds, and reported medians. Each round used 50 forward timing iterations and 12 forward+backward iterations.

| Sequence | Expanded SDPA F+B | Triton F+B | Auto F+B | Auto vs expanded SDPA |
|---:|---:|---:|---:|---:|
| 128 | 0.865 ms | 1.014 ms | 0.914 ms | 0.947x |
| 256 | 0.973 ms | 0.978 ms | 0.911 ms | 1.067x |
| 512 | 0.884 ms | 0.959 ms | 0.899 ms | 0.983x |
| 1024 | 1.314 ms | 1.113 ms | 1.212 ms | **1.084x** |
| 2048 | 4.056 ms | 3.886 ms | 3.842 ms | **1.056x** |

Interpretation:
- At 128–512, auto selects expanded-KV SDPA. Results are close to the baseline (within about 7% in this run); treat these as parity, not a meaningful performance gain.
- At 1024 and 2048, auto selects Triton and measured 8.4% and 5.6% lower forward+backward attention latency than expanded-KV SDPA.
- Direct Triton measured 1.18x and 1.04x versus expanded SDPA at 1024 and 2048, respectively. The 1024 direct-Triton and auto results differ by about 0.10 ms despite using the same kernel path; this highlights remaining timing/dispatch variability. The results are medians from one five-round sweep, not a broad statistical study.

These are **standalone attention** measurements, not a full decoder block or full language-model training result. Earlier synthetic decoder-block measurements were near parity at 1024 and about 2.4% slower at 2048. Do not extrapolate the attention-only gains to end-to-end model tokens/second.

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
`backend="auto"` uses deliberately narrow profiles measured on **Tesla T4 / SM75**, PyTorch **2.6.0+cu124** and Triton **3.2.x**. For contiguous FP16 causal self-attention with B=2, Hq=8, Hkv=2, D=64, default scale 0.125 and Sq=Sk in {128, 256, 512, 1024, 2048}, it uses expanded-KV SDPA at 128–512 and fused Triton at 1024–2048. For non-causal decode on the same stack and batch/head/dimension profile, it has three exact measured cases: (Sq,Sk)=(1,2048) selects Triton, (4,1024) selects expanded-KV SDPA, and (16,4096) selects Triton. GPU tests assert both the selected route and numerical output. Other shapes, devices, PyTorch versions, CUDA versions and non-contiguous inputs retain the conservative PyTorch SDPA fallback. This is not a universal autotuner; explicit `backend="triton"` and `backend="sdpa"` remain available. Re-measure the profile when changing the software stack.

## Architecture notes

- Forward uses online softmax and a tiled Q × Kᵀ / P × V loop. The full `[Sq, Sk]` attention matrix is never stored.
- Causal forward skips complete future K/V tiles instead of loading and masking them.
- Head dimensions 65–127 are split into a 64-wide dot plus a padded tail dot, avoiding a single full 128-wide dot for D=72/96.
- Accumulators and log-sum-exp are FP32; inputs and output are FP16 in the custom path.
- GQA maps each query head to its corresponding KV head group.
- The experimental fused backward uses Triton kernels for dQ and dK/dV. The original short-sequence dK/dV path assigns one CTA to each KV tile and walks query tiles serially. A new long-sequence path (Sq >= 1024) splits query tiles into independent chunks, writes FP32 partial dK/dV tensors and reduces them in a second kernel. This avoids atomics and increases parallel work on Turing. The split path passed the expanded physical-T4 correctness suite; the current complete branch suite is 48/48 tests, including 1024- and 2048-token GQA/MQA cases. Its measured T4 speed and limitations are recorded above. The earlier batched PyTorch implementation remains available internally as a correctness reference.
- The custom kernel currently targets the upstream Triton 3.2 Turing backend. The current experimental launch defaults are BM=64, BN=32, 4 warps and 2 stages, based on the first T4 tile sweep; keep benchmarking new shapes before treating them as universal defaults.

## Roadmap

- [x] Public PyTorch API and SDPA fallback
- [x] Experimental tiled online-softmax forward
- [x] MHA/GQA/MQA indexing
- [x] Batched blockwise recomputation backward for validation
- [x] CPU/API tests and SM75 GPU test suite
- [x] Run the Triton integration suite on a physical T4 (11/11 user-reported tests passed)
- [x] Initial fused SM75 dQ/dK/dV backward implementation; earlier revision passed T4 tests
- [x] Validate split-query dK/dV reduction correctness on a physical T4 (included in the user-reported 48/48 current suite)
- [ ] Tune long-sequence split count and partial-gradient reduction overhead on T4
- [ ] Complete tile/warp/stage sweeps across representative T4 workloads
- [x] Add causal future-tile skipping
- [x] Add split-dot path for head dimensions 72/96
- [ ] Profile and tune the split-dot path on T4
- [ ] Variable-length packed sequences
- [x] Experimental split-KV decode for short queries with long caller-managed K/V context
- [ ] Integrated cache management, paged KV cache and local-window/dropout variants
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
