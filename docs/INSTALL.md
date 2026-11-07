# Installation notes

KOTFlash does not ship a custom CUDA extension in the 0.1.x series. The experimental kernel is compiled by Triton at runtime.

1. Install a CUDA-enabled PyTorch build compatible with the host driver.
2. Install KOTFlash:

       git clone https://github.com/AitiKot/KOTFlash.git
       cd KOTFlash
       python -m pip install -e .

3. The SDPA fallback does not need Triton. For the experimental Turing path, install upstream Triton 3.2.x:

       python -m pip install 'triton>=3.2,<3.3'

   Upstream Triton dropped Volta/Turing support beginning with 3.3. Newer PyTorch environments may bundle a newer Triton; use a separate environment with a compatible PyTorch/Triton pairing or a maintained Turing-enabled downstream build. Do not blindly downgrade Triton in an existing environment.
4. Install test dependencies and run API tests:

       python -m pip install -e '.[test]'
       pytest -q

5. On a physical T4, verify that torch.cuda.get_device_capability() returns (7, 5), then run:

       pytest -q tests/test_triton.py
       python benchmarks/benchmark_attention.py --device cuda

The custom Turing path currently requires FP16 and head_dim between 16 and 128. Head dimension 72 is padded to 128. BF16 is not supported by Turing Tensor Cores.
