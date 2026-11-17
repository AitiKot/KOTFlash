# SPDX-License-Identifier: Apache-2.0
"""Static regression checks for the interactive KOT-V2-Mini Colab notebook."""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "KOT_V2_Mini_Russian_TinyStories_Weight_Lab.ipynb"
)


def load_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    return notebook


def code_sources(notebook):
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_all_code_cells_are_python_syntax_without_shell_magics():
    sources = code_sources(load_notebook())
    assert len(sources) >= 5

    for index, source in enumerate(sources):
        for line in source.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith(("!", "%")), (
                f"Cell {index} contains a notebook shell/line magic"
            )

        # Colab/IPython accepts top-level await. Wrapping the cell in an async
        # function lets CPython's AST parser validate that syntax in CI.
        wrapped = "async def _notebook_cell():\n" + textwrap.indent(
            source or "pass", "    "
        )
        ast.parse(wrapped, filename=f"<kot-v2-mini-cell-{index}>")


def test_live_weight_control_and_checkpoint_are_wired():
    notebook = load_notebook()
    sources = code_sources(notebook)
    combined = "\n".join(sources)

    assert "weight_slider.observe(slider_changed" in combined
    assert "await asyncio.sleep(0.01)" in combined
    assert "current_multiplier=float(weight_slider.value)" in combined
    assert "change_expert_weights(alpha)" in combined
    assert "flat_out.index_add_" in combined
    assert "route_positions = torch.nonzero(" in combined
    assert "final_state" in combined and "best_state" in combined
    assert "trained_state" not in combined

    # The save/download action must be an executable code cell, not Markdown.
    assert notebook["cells"][-1]["cell_type"] == "code"
    assert "files.download(checkpoint_path)" in sources[-1]
