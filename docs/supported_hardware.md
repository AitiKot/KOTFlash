# Hardware and software support

## Custom Triton kernels

| Hardware | Compute capability | Custom Triton backend | Evidence / status |
|---|---:|---|---|
| NVIDIA Tesla T4 | 7.5 (SM75) | Intended target | Physical T4 correctness and performance runs have been reported by the maintainer; latest auto-routing additions still require a final full-suite T4 rerun |
| Other NVIDIA Turing GPUs | 7.5 (SM75) | Intended | Not independently validated on every device |
| Ampere and newer | 8.0+ | Not enabled by this release | Use PyTorch SDPA |
| CPU | N/A | Not available | Use PyTorch SDPA |

## Software requirements

The conservative custom-kernel target is:
- CUDA-enabled PyTorch compatible with the host driver;
- upstream Triton 3.2.x, or a maintained compatible Turing-enabled build;
- FP16 Q/K/V tensors;
- head dimension between 16 and 128.

Upstream Triton 3.3 and later dropped Volta/Turing support. Newer PyTorch environments may bundle a Triton build that cannot compile for SM75. Use a separate compatible environment rather than blindly downgrading the Triton installation inside an existing PyTorch setup.

## Measured automatic profile

The current `backend="auto"` performance profile is deliberately narrow. It was measured on:

- NVIDIA Tesla T4 / SM75;
- PyTorch 2.6.0+cu124;
- CUDA 12.4;
- Triton 3.2.x;
- contiguous FP16 Q/K/V;
- B=2, Hq=8, Hkv=2, D=64, default scale 0.125.

The profile contains five causal self-attention sequence lengths (128, 256, 512, 1024 and 2048) and three non-causal decode shapes: (Sq, Sk) = (1, 2048), (4, 1024) and (16, 4096).

No other shape or software stack should be assumed to use the same winning backend. Unprofiled inputs retain the PyTorch SDPA fallback.

## Known limitations

- The Triton kernels currently support FP16 only.
- BF16 is not supported by the current Turing custom path.
- Packed variable-length sequences are not implemented.
- KOTFlash does not manage or page KV caches.
- Dropout and local-window attention are not implemented.
- Performance varies with shapes, clocks, thermal/power state, driver and software versions.
- No full-model generation or full-model training speedup has been established.

Always run correctness tests and benchmarks on the exact target GPU before relying on a performance profile.
