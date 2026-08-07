from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def test_dataset_validation_precedes_publication() -> None:
    source = (EXAMPLE_ROOT / "full_research_pipeline_legacy.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_dataset"
    )
    ordered: list[str] = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                ordered.append(node.func.id)
    assert ordered.index("validate_maintained_dataset_preset") < ordered.index(
        "publish_market_dataset_artifact"
    )


def test_maintained_facade_uses_isolated_validated_dataset_builder() -> None:
    if str(EXAMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT))
    maintained = importlib.import_module("full_research_pipeline")
    legacy = importlib.import_module("full_research_pipeline_legacy")

    assert maintained._legacy is not legacy
    assert maintained._legacy.__name__ == (
        "_trade_rl_single_symbol_full_research_pipeline_runtime"
    )
    assert maintained._legacy._SYMBOLS == ("BTCUSDT",)
    assert maintained.build_dataset is maintained._legacy._build_dataset
    assert maintained._build_dataset is maintained._legacy._build_dataset
    assert legacy._SYMBOLS == ("BTCUSDT", "ETHUSDT", "BNBUSDT")
