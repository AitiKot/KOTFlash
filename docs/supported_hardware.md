# Hardware and software support

## Current custom-kernel target

| Hardware | Compute capability | Custom Triton forward | Status |
|---|---:|---|---|
| NVIDIA Tesla T4 | 7.5 (SM75) | Intended | Target hardware; physical validation still required |
| Other NVIDIA Turing GPUs | 7.5 (SM75) | Intended | Not yet validated individually |
| Ampere and newer | 8.0+ | No custom dispatch in this release | Uses PyTorch SDPA fallback |
| CPU | N/A | No | Uses PyTorch SDPA |

## Software requirements for the experimental path

- CUDA-enabled PyTorch installation compatible with the installed NVIDIA driver.
- Upstream Triton 3.2.x, the last upstream release line with SM75 support.
- FP16 Q/K/V.
- Head dimension between 16 and 128.

Upstream Triton 3.3 and later dropped Volta and Turing support. Newer PyTorch installations may bundle a Triton version without SM75 support. Use a separate environment with a compatible PyTorch/Triton pairing or a maintained downstream build; KOTFlash does not rewrite the installed compiler stack.

## Known limitations

- The current kernel uses simple fixed tile defaults (BLOCK_M=32, BLOCK_N=64, 4 warps, 2 stages). They are not claimed to be optimal.
- Head dimension 72 is padded to 128. It is functionally accepted but may waste work.
- Custom forward is FP16 only. SDPA fallback supports the dtypes/hardware supported by the installed PyTorch build.
- Backward is a Python/PyTorch reference implementation and will be much slower than a fused GPU kernel.
- No dropout, local-window, paged KV-cache or variable-length packed interface yet.

A backend is not considered validated until the test suite runs on the stated physical GPU and the logs are published with the software versions.
