# SPDX-License-Identifier: Apache-2.0
"""Static checks for the runnable Google Colab validation notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "KOTFlash_T4_Validation.ipynb"


def test_colab_notebook_is_valid_and_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert len(code_cells) >= 4

    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        assert not any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ), f"Cell {index} must use plain Python, not notebook shell magics"
        ast.parse(source, filename=f"<colab-cell-{index}>")


def test_colab_environment_installs_pip_and_tracks_main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert '"--seed"' in source
    assert 'PY, "-m", "pip", "--version"' in source
    assert 'BRANCH = "main"' in source
    assert "feature/t4-performance-rebased" not in source
