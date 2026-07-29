from __future__ import annotations

import ast
from pathlib import Path


def test_dataset_validation_precedes_publication() -> None:
    source = (Path(__file__).resolve().parents[2] / "examples/binance-multitimeframe/full_research_pipeline.py").read_text()
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_build_dataset")
    ordered = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                ordered.append(node.func.id)
    assert ordered.index("validate_maintained_dataset_preset") < ordered.index("publish_market_dataset_artifact")
