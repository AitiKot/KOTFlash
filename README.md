# KOTFlash

**Experimental memory-efficient attention for NVIDIA Turing (SM75), exposed through a general PyTorch API.**

KOTFlash is an independent research project with a PyTorch SDPA path and experimental Triton online-softmax forward/backward kernels. Tesla T4 is the primary optimization target.

> **Status: alpha.** The Split-KV implementation passed a user-reported 52/52 physical-T4 test run on PyTorch 2.6.0+cu124 / Triton 3.2.0. Subsequent changes to automatic decode routing need a fresh full-suite T4 run. Full-model training and generation speedups have not been established.

## Features

| Component | Status |
|---|---|
| PyTorch SDPA fallback | Implemented |
| MHA / GQA / MQA | Implemented |
| Triton online-softmax forward | Experimental; FP16, SM75, Triton 3.2.x |
| Causal future-tile skipping | Implemented |
| Head dimensions 72 and 96 | Split-dot forward path implemented; broader T4 tuning needed |
| Fused Triton backward | dQ and dK/dV kernels implemented |
| Long-sequence backward | Split-query partial dK/dV plus reduction |
| Split-KV decode | Experimental short-query / long-visible-cache path |
| Automatic backend | Narrow, measured T4 profiles; otherwise SDPA fallback |
| Paged KV cache / cache manager | Not implemented |
| Packed variable-length sequences, dropout, local window | Not implemented |

## Installation

Install a CUDA-enabled PyTorch build compatible with your driver first. The portable SDPA path does not require Triton:

~~~bash
git clone https://github.com/AitiKot/KOTFlash.git
cd KOTFlash
python -m pip install -e .
~~~

For the experimental Turing path, install in an environment with a compatible PyTorch/Triton pairing. Upstream Triton 3.2.x is the conservative Turing-compatible line:

~~~bash
python -m pip install -e '.[t4,test]'
~~~

Do not blindly downgrade Triton in an existing PyTorch environment. PyTorch may rely on its bundled Triton version. See [docs/INSTALL.md](docs/INSTALL.md).

## Quick start

~~~python
import torch
from kotflash import attention

q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 2, 512, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)

# Conservative automatic selection; unprofiled inputs use PyTorch SDPA.
out = attention(q, k, v, causal=True, backend="auto")
out.float().square().mean().backward()
~~~

Select a backend explicitly:

~~~python
out = attention(q, k, v, backend="sdpa")    # portable PyTorch implementation
out = attention(q, k, v, backend="triton")  # experimental SM75 Triton kernels
out = attention(q, k, v, backend="auto")    # narrow measured profiles + fallback
~~~

Public tensors use [B, H, S, D] layout. For GQA/MQA, the query-head count must be divisible by the KV-head count.

## KV-cache decode

KOTFlash accepts caller-supplied K/V tensors; it does not own or page the cache. For one-token decode, append the new K/V entries and pass the complete visible prefix with `causal=False`:

~~~python
from kotflash import attention

# q_new: [B, Hq, 1, D]
# k_cache, v_cache: [B, Hkv, visible_tokens, D]
out = attention(q_new, k_cache, v_cache, causal=False, backend="triton")
~~~

The package also exports simple convenience helpers, `append_kv_cache` and `decode_step`, for caller-managed single-token decoding. The `torch.cat` helper copies the existing cache for clarity; production model code should normally write into preallocated KV storage. For fixed-capacity inference, `kotflash.cache.PreallocatedKVCache` provides validated append-only storage and exposes only the populated prefix. Its `contiguous_view()` still copies that prefix because the current Triton kernels require contiguous K/V.

For a runnable GQA example that performs prefill, appends a token and runs incremental decode:

~~~bash
python examples/gqa_kv_cache_decode.py --backend triton
~~~

For a single new query, all entries in the supplied cache must be visible to that query. Do not use this non-causal mode when the passed K/V includes future positions.

The Split-KV forward path is selected by the explicit Triton backend for non-causal Sq <= 16, Sk >= 1024 and D <= 64. It splits the visible K/V context into independent chunks and merges partial results using log-sum-exp weights.

## Benchmarks

Attention forward/backward benchmark:

~~~bash
python benchmarks/benchmark_attention.py \
  --device cuda --batch 2 --q-heads 8 --kv-heads 2 \
  --seq 1024 --head-dim 64 --causal \
  --warmup 20 --iterations 50 --repeats 5
~~~

Caller-managed KV append plus one-token decode benchmark:

~~~bash
python benchmarks/benchmark_decode.py \
  --batch 2 --q-heads 8 --kv-heads 2 --head-dim 64 \
  --kv-seq 2048 --warmup 20 --iterations 50 --repeats 5
~~~

It reports GPU-event and synchronized wall-clock latency for attention-only and append+attention. Throughput uses wall-clock time for one decode position per batch. The example uses torch.cat for clarity, which copies the cache; production models should normally write into preallocated KV storage. Model projections, MLP and sampling are excluded.

Synthetic decoder-block training benchmark:

~~~bash
python benchmarks/benchmark_decoder_block.py \
  --batch 2 --seq 512 --q-heads 8 --kv-heads 2 --head-dim 64
~~~

Tile sweep:

~~~bash
python benchmarks/tune_triton_tiles.py --seq 1024 --head-dim 72 --causal
~~~

For a reproducible T4 validation run, use:

~~~bash
bash benchmarks/run_t4_validation.sh
~~~

## Reported T4 measurements

All numbers below are user-run measurements on Tesla T4 / SM75, PyTorch 2.6.0+cu124, CUDA 12.4 and Triton 3.2.0. They are measurements of specific shapes, not general performance guarantees.

### Causal attention forward + backward

B=2, Hq=8, Hkv=2, D=64, FP16:

| Sequence | Expanded-KV SDPA | KOTFlash Auto | Auto vs expanded |
|---:|---:|---:|---:|
| 128 | 0.865 ms | 0.914 ms | 0.95x |
| 256 | 0.973 ms | 0.911 ms | 1.07x |
| 512 | 0.884 ms | 0.899 ms | 0.98x |
| 1024 | 1.314 ms | 1.212 ms | 1.08x |
| 2048 | 4.056 ms | 3.842 ms | 1.06x |

The measured attention-only gain at the two longest tested lengths is modest. An earlier synthetic decoder-block benchmark was effectively tied at sequence 1024 and about 2.4% slower with KOTFlash at 2048. Do not extrapolate these standalone attention results to full-model training.

### Non-causal decode-shaped attention forward

B=2, Hq=8, Hkv=2, D=64, FP16:

| Q / KV | Expanded-KV SDPA | KOTFlash Auto | Auto vs expanded |
|---:|---:|---:|---:|
| 1 / 2048 | 0.33549 ms | 0.20326 ms | 1.65x |
| 4 / 1024 | 0.15445 ms | 0.15480 ms | 1.00x |
| 16 / 4096 | 0.43869 ms | 0.28550 ms | 1.54x |

These are attention-only forward timings from one five-round measurement. The Q=4 result is parity. Explicit Triton and Auto selected the same Triton path for Q=1 and Q=16, but their medians differed during the recorded run. Treat the ratios as indicative, not guaranteed.

## Compatibility and limitations

The experimental Triton backend currently requires:
- NVIDIA Turing SM75;
- FP16 Q/K/V;
- head dimension 16 through 128;
- upstream Triton 3.2.x or a compatible Turing-enabled build.

The narrow automatic profile was measured only on Tesla T4 / PyTorch 2.6.0+cu124 / CUDA 12.4 / Triton 3.2.x for exact documented shapes. Other shapes and stacks retain the portable SDPA fallback.

KOTFlash is an attention library, not a full inference engine. It offers a simple cache-append convenience helper but no preallocated cache manager, paging or eviction. It does not include model projections, MLPs, token sampling or a serving runtime.

## Architecture and development

- [docs/DESIGN.md](docs/DESIGN.md) — forward, backward and Split-KV design.
- [docs/INSTALL.md](docs/INSTALL.md) — environment and installation details.
- [docs/benchmarking.md](docs/benchmarking.md) — benchmark methodology.
- [docs/supported_hardware.md](docs/supported_hardware.md) — tested hardware and constraints.
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution process.

## License and patent note

KOTFlash is distributed under the **Apache License 2.0**. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This project has not undergone a formal patent search, freedom-to-operate analysis or legal review. The license does not guarantee that every possible use is free of third-party patent claims. No third-party attention-kernel source is intentionally included in this repository; external projects are references only.
