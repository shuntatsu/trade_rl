from __future__ import annotations

import ast
from pathlib import Path


def test_single_split_calls_authoritative_candidate_api() -> None:
    source = Path("mars_lite/pipeline/residual_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_baseline_residual"
    )
    called_names = {
        node.func.id
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "train_select_residual_candidates" in called_names
    assert "_train_residual_ensemble" not in called_names
