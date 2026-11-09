# Installation

KOTFlash is a PyTorch package. Install a CUDA-enabled PyTorch build compatible with the NVIDIA driver before installing the project. KOTFlash does not replace or downgrade PyTorch, CUDA or the GPU driver.

## Install the portable SDPA package

The PyTorch SDPA backend does not require Triton:

~~~bash
git clone https://github.com/AitiKot/KOTFlash.git
cd KOTFlash
python -m pip install -e .
~~~

The package can also be used on CPU through PyTorch SDPA.

## Install the experimental Turing backend

The custom Triton kernels target NVIDIA Turing SM75. Upstream Triton 3.2.x is the conservative supported line for Turing:

~~~bash
python -m pip install -e '.[t4]'
~~~

Install the optional dependencies in an environment with a compatible PyTorch/Triton pairing. Do not blindly downgrade Triton inside an existing PyTorch environment: PyTorch may depend on a particular bundled Triton build.

For the reproducible Tesla T4 profile currently documented by the project, the measured software stack is:

- Tesla T4, compute capability 7.5;
- PyTorch 2.6.0 + CUDA 12.4;
- upstream Triton 3.2.0.

Other combinations may work for the explicit Triton backend, but have not been established as equivalent performance profiles.

## Verify the environment

~~~bash
python - <<'PY'
import torch
import triton
import kotflash

print("KOTFlash:", kotflash.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Triton:", triton.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
    print("Compute capability:", torch.cuda.get_device_capability())
print("Turing Triton supported:", kotflash.triton_turing_supported())
PY
~~~

## Install test dependencies

~~~bash
python -m pip install -e '.[test]'
python -m compileall -q kotflash benchmarks tests examples
python -m pytest -q
~~~

On a physical T4, run:

~~~bash
python -m pytest -q tests/test_triton.py
bash benchmarks/run_t4_validation.sh
~~~

The validation script is intentionally strict: it refuses to label runs from a different GPU or software stack as the measured T4 profile.

## Current limitations

- Triton kernels require CUDA, SM75-compatible Triton and FP16 tensors.
- The explicit custom backend supports head dimensions from 16 through 128.
- The narrow `auto` profile is not a general autotuner.
- CPU and unsupported GPU configurations use PyTorch SDPA when `backend="auto"` or `backend="sdpa"` is selected.
