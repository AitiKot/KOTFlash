# KOTFlash

**Experimental memory-efficient attention for NVIDIA Turing (SM75), with a general PyTorch API.**

KOTFlash is an independent research implementation. Its first release provides a PyTorch SDPA fallback and an experimental Triton online-softmax forward kernel designed to compile for Turing GPUs. The Triton path has a blockwise recomputation backward implementation intended for correctness experiments, not yet a tuned production backward kernel.

> **Status: experimental / alpha.** Do not assume a speedup. Run the included correctness tests and benchmarks on your own GPU. The project has not yet been validated on a physical T4 in this development environment.

## Goals

- Work on NVIDIA T4 and compatible NVIDIA Turing GPUs (compute capability 7.5).
- Offer a common API for MHA, GQA and MQA experiments.
- Avoid materializing the full attention matrix in the Triton forward path.
- Keep a portable PyTorch SDPA fallback for unsupported hardware/shapes.
- Publish reproducible correctness and performance tests before claiming improvements.

## Current support (0.1.0)

| Feature | Status |
|---|---|
| PyTorch SDPA fallback | Available |
| Triton online-softmax forward | Experimental; FP16 CUDA, head dimension <= 128 |
| Triton path on SM75 | Intended; requires validation on T4 |
| MHA / GQA / MQA | Implemented in API and Triton indexing; needs GPU test coverage |
| Causal / non-causal | Implemented in Triton forward and blockwise backward |
| Backward | Blockwise PyTorch recomputation; correctness-first and potentially slow |
| Variable-length packed batches | Not implemented |
| Dropout / local-window / KV cache | Not implemented |
| Native CUDA/C++ kernels | Not in this release |

The Triton kernel pads `head_dim` internally to the next power of two. `head_dim=72` is therefore accepted by the experimental path, but may perform extra arithmetic and is not a tuned specialization. The public API uses `[B, H, S, D]` layout, matching PyTorch SDPA.

## Install

Install a CUDA-enabled PyTorch build appropriate for your environment first. KOTFlash deliberately does not force-install or replace PyTorch.

```bash
pip install -U pip
pip install "triton>=3.2,<3.3"
pip install git+https://github.com/AitiKot/KOTFlash.git
```

The `triton` dependency is optional at runtime: without it, `backend="auto"` uses PyTorch SDPA. For upstream Triton on Turing, use the 3.2.x line: support for Turing/Volta was removed from upstream Triton 3.3. This usually pairs naturally with a PyTorch 2.6 environment; newer PyTorch releases bundle newer Triton versions and may require a separate compatible environment or a maintained Turing-enabled Triton build. You can install the test tools with `pip install pytest`.

For a source checkout:

```bash
git clone https://github.com/AitiKot/KOTFlash.git
cd KOTFlash
pip install -e .
```

## Quick start

```python
import torch
from kotflash import attention

# PyTorch layout: [batch, heads, sequence, head_dim]
q = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 4, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)

# auto: experimental Triton path on supported SM75 FP16 inputs,
# otherwise PyTorch SDPA.
y = attention(q, k, v, causal=True, backend="auto")
y.float().square().mean().backward()
```

Force a backend:

```python
y = attention(q, k, v, causal=False, backend="triton")
y = attention(q, k, v, causal=False, backend="sdpa")
```

`backend="triton"` raises an explicit error when the requested path is unavailable. `backend="auto"` falls back to SDPA rather than silently changing tensor dtype or device.

## Benchmark

```bash
python benchmarks/benchmark_attention.py --device cuda --batch 2 --q-heads 16 --kv-heads 4 --seq 1024 --head-dim 64 --causal
```

The benchmark reports forward latency, forward+backward latency, peak allocated memory, and maximum absolute difference from PyTorch SDPA. It warms up, synchronizes CUDA, and reports the selected backend. Compare results on the same device, software stack, shapes, and clock/power conditions.

## API

```python
attention(q, k, v, *, causal=False, scale=None, backend="auto")
```

- `q`: `[B, Hq, Sq, D]`
- `k`, `v`: `[B, Hkv, Sk, D]`
- `Hq % Hkv == 0` for GQA/MQA
- `causal=True` uses an upper-left causal mask (`key_index <= query_index`)
- `scale=None` means `1 / sqrt(D)`
- `backend`: `auto`, `triton`, or `sdpa`

## Important implementation notes

1. The first Triton kernel implements online softmax and stores output plus log-sum-exp, not the full `[Sq, Sk]` attention matrix.
2. The Triton backward currently recomputes score tiles in PyTorch. This limits temporary memory but launches many operations and is not expected to be the fastest path. A fused SM75 backward is a future milestone.
3. The implementation currently assumes contiguous inputs for the custom Triton path; the wrapper makes contiguous copies where necessary.
4. FP16 inputs are required for the Triton backend. SDPA fallback supports the dtypes/hardware supported by the installed PyTorch build.
5. GQA backward accumulates KV gradients across query-head groups. Validate numerical tolerance on the exact PyTorch/CUDA stack before training a model.

## Development

```bash
pip install -e .
pip install "triton>=3.2,<3.3" pytest
pytest -q
```

Please include GPU model, compute capability, driver, CUDA runtime, PyTorch, Triton versions, command line, and full benchmark output in bug reports.

## Roadmap

- [x] Public PyTorch API and SDPA fallback
- [x] Experimental Triton online-softmax forward
- [x] Blockwise recomputation backward
- [ ] GPU correctness validation on physical T4
- [ ] Fused SM75 backward kernels
- [ ] Tuned tile configurations for common head dimensions
- [ ] Packed variable-length sequences
- [ ] Dropout, local-window attention, and KV-cache paths
- [ ] CI matrix and published T4 benchmarks

## License

KOTFlash is distributed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are accepted under the same license; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Third-party acknowledgements

KOTFlash is inspired by the general research direction of IO-aware, tiled attention and uses Triton as an optional separately installed compiler/runtime. No code from the repositories listed below is copied or bundled here:

- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention Turing](https://github.com/ssiu/flash-attention-turing)
- [Triton](https://github.com/triton-lang/triton)

These links identify related work and dependencies only; their licenses remain separate from this project.
