# SPDX-License-Identifier: Apache-2.0
"""Static checks for the opt-in TinyStories training benchmark and Colab notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tinystories_benchmark_is_valid_python_and_uses_real_splits():
    path = ROOT / "benchmarks" / "benchmark_tinystories_training.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    assert 'DATASET_NAME = "skeskinen/TinyStories-hf"' in source
    assert '"train", train_needed' in source
    assert '"validation", valid_needed' in source
    assert 'backend == "triton"' in source
    assert 'backend == "sdpa_expanded"' in source
    assert '"--seq", type=int, default=4096' in source
    assert '"--steps", type=int, default=100' in source
    assert '"--warmup", type=int, default=5' in source
    assert '"speedup_triton_vs_expanded_sdpa"' in source
    assert '"finite_after_training"' in source
    assert '"initial_eval_loss_parity_passed"' in source


def test_tinystories_benchmark_has_no_notebook_shell_magics():
    path = ROOT / "benchmarks" / "benchmark_tinystories_training.py"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith("!")
        assert not stripped.startswith("%")


def test_tinystories_colab_notebook_code_cells_parse_as_python():
    path = ROOT / "notebooks" / "KOTFlash_TinyStories_4096_100step.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 4
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{path}:cell-{index}")
        for line in source.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith("!")
            assert not stripped.startswith("%")
    assert 'BRANCH = "main"' in combined
    assert '"--seq", "4096"' in combined
    assert '"--steps", "100"' in combined
    assert '"benchmark_tinystories_training.py"' in combined
