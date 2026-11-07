# Installation notes

KOTFlash does not ship a custom CUDA extension in v0.1.0. This avoids tying the first release to one CUDA toolkit/compiler combination. The experimental kernel is compiled by Triton at runtime for the installed GPU.

1. Install the CUDA-enabled PyTorch build recommended for the host (Colab, Kaggle, or local machine).
2. Install a Triton version compatible with that PyTorch build. On many PyTorch installations Triton is already present; check its version with python -c import-triton.
3. For upstream Turing support, use Triton 3.2.x (pip install triton greater-than-or-equal-to 3.2 and less-than 3.3). Upstream Triton dropped Volta/Turing support beginning with 3.3, so the newest Triton release is not appropriate for the T4 path. PyTorch 2.6 normally pairs with Triton 3.2; PyTorch 2.7+ environments bundle newer Triton and may need a separate environment or a maintained Turing-enabled build.
4. Install KOTFlash from GitHub.
5. Run python -m pytest -q and the benchmark script.

For T4, verify torch.cuda.get_device_capability() returns (7, 5). The T4 path currently requires FP16 tensors. BF16 Tensor Core execution is not supported on SM75.
