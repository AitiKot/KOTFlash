# Third-party notices

KOTFlash does not intentionally bundle source code, assets or build scripts from third-party attention-kernel repositories.

## Runtime dependencies

Optional/runtime dependencies are installed separately and remain under their own licenses:

- PyTorch — BSD-style license; https://github.com/pytorch/pytorch
- Triton — see the license distributed with the installed Triton version; https://github.com/triton-lang/triton

## Related research implementations

These projects were inspected as related research/compatibility references. Their implementation source is not copied or bundled in KOTFlash:

- Dao-AILab/flash-attention — BSD 3-Clause; https://github.com/Dao-AILab/flash-attention
- BhoumikPatidar/flash_attn_turing — MIT; https://github.com/BhoumikPatidar/flash_attn_turing
- ssiu/flash-attention-turing — https://github.com/ssiu/flash-attention-turing

The inspected default-branch tree of ssiu/flash-attention-turing contains no top-level LICENSE/COPYING/NOTICE file. KOTFlash therefore treats it as an unlicensed reference for high-level research context only and does not copy, adapt or redistribute its source. Do not reuse that repository's code unless its authors provide explicit permission or a valid license.

This file is informational and does not relicense any external project. Any future vendored or adapted code must be reviewed before inclusion and recorded here with its exact origin, applicable license, copyright notices, required attributions and modifications.
