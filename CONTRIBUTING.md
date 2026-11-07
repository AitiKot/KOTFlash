# Contributing to KOTFlash

Thanks for helping improve KOTFlash.

## Ground rules

- Keep changes focused and explain the performance or correctness hypothesis.
- Do not copy code from another project unless its license is compatible and the source, license, and required notices are recorded in THIRD_PARTY_NOTICES.md.
- Do not submit benchmark claims without the exact GPU, driver, CUDA, PyTorch, Triton versions, shapes, warm-up, iteration count, and baseline.
- Correctness takes priority over speed. Add tests for forward, gradients, masks, and grouped-query layouts.
- Avoid silent dtype/device conversions and silent backend changes.

## Development setup

    pip install -e .
    pip install "triton>=3.2,<3.3" pytest
    pytest -q

## Pull requests

Describe the problem, design, alternatives considered, tests run, and known limitations. For GPU work, attach reproducible benchmark output from a physical device. Contributors submit their contributions under Apache-2.0 unless a separate written agreement says otherwise.
