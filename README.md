# KOTFlash

**Experimental memory-efficient attention for NVIDIA Turing (SM75), exposed through a general PyTorch API.**

KOTFlash is an independent, research-stage implementation. The first public milestone contains a PyTorch SDPA fallback and an experimental Triton online-softmax forward kernel. Backward currently recomputes blocks in ordinary PyTorch operations: it is intended for correctness experiments, not production-speed training.

> **Status: alpha / initial physical Tesla T4 forward benchmark completed; full GPU test-suite validation is still pending. No general speedup is claimed.** Run the correctness suite and benchmarks on your hardware before using it for training.

## What works today

| Component | Status |
|---|---|
| PyTorch SDPA backend | Implemented; CPU and CUDA where PyTorch supports it |
| MHA / GQA / MQA API | Implemented |
| Triton online-softmax forward | Experimental; FP16, SM75, Triton 3.2.x, head_dim 16–128 |
| Causal / non-causal | Implemented in both paths; upper-left causal alignment |
| Head dimensions 72 and 96 | Split into 64 + tail tiles to reduce padding versus a full 128-wide dot; still requires T4 profiling |
| Backward | Blockwise PyTorch recomputation; correctness-first and slow |
| Physical T4 validation | Initial user-reported forward benchmark; full pytest integration run pending |
| Fused SM75 backward | Not implemented |
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

The experimental Triton path currently requires CUDA SM75, FP16, upstream Triton 3.2.x, and `16 <= D <= 128`. Non-contiguous inputs are copied to contiguous buffers before the custom kernel. `auto` currently uses SDPA conservatively because the initial T4 measurements do not establish a consistent Triton win; select `backend='triton'` to benchmark the custom kernel.

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

The benchmark compares SDPA, Triton and auto on the same input shape. It reports forward latency, forward+backward latency, incremental peak allocated memory, and maximum absolute error. Record the GPU model, driver, CUDA runtime, PyTorch and Triton versions with any published result. Benchmark results from other GPUs are not T4 results.

## Architecture notes

- Forward uses online softmax and a tiled Q × Kᵀ / P × V loop. The full `[Sq, Sk]` attention matrix is never stored.
- Causal forward skips complete future K/V tiles instead of loading and masking them.
- Head dimensions 65–127 are split into a 64-wide dot plus a padded tail dot, avoiding a single full 128-wide dot for D=72/96.
- Accumulators and log-sum-exp are FP32; inputs and output are FP16 in the custom path.
- GQA maps each query head to its corresponding KV head group.
- The reference backward recomputes score/probability tiles and serially accumulates shared dK/dV in Python. This avoids a full attention matrix, but is expected to be slow. It is not a fused CUDA/Triton backward kernel.
- The custom kernel currently targets the upstream Triton 3.2 Turing backend. Tile choices are initial correctness-oriented defaults, not a tuning result.

## Roadmap

- [x] Public PyTorch API and SDPA fallback
- [x] Experimental tiled online-softmax forward
- [x] MHA/GQA/MQA indexing
- [x] Blockwise recomputation backward for validation
- [x] CPU/API tests and SM75 GPU test suite
- [ ] Run the full test suite on a physical T4 and publish raw logs
- [ ] Fused SM75 backward kernels
- [ ] Tune tile sizes, warps and stages on T4
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
