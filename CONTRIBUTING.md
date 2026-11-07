# Contributing to KOTFlash

Thanks for helping improve KOTFlash.

## Ground rules

- Correctness comes before speed. Include tests for forward values, gradients, masks and grouped-query layouts.
- Keep changes focused and state the correctness/performance hypothesis.
- Do not copy code from another project unless its license has been reviewed and all required source, license and attribution notices are included. Record incorporated third-party code in THIRD_PARTY_NOTICES.md.
- Do not submit benchmark claims without the exact GPU, compute capability, driver, CUDA, PyTorch and Triton versions, shapes, warm-up, iteration count and baseline.
- Avoid silent dtype/device conversions and silent backend changes.
- Do not claim T4 validation unless the tests were actually run on a physical NVIDIA T4.

## Development setup

```bash
python -m pip install -e '.[test]'
python -m pip install 'triton>=3.2,<3.3'  # only for SM75 kernel work
pytest -q
```

## Pull requests

Describe the problem, design, alternatives considered, tests run and known limitations. For GPU work, attach reproducible benchmark output from a physical device.

Unless a separate written agreement says otherwise, contributions submitted for inclusion are licensed under Apache-2.0, consistent with the project license.
