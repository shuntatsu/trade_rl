from __future__ import annotations

import ast
from pathlib import Path


def _tree(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _imports(tree: ast.AST) -> dict[str, str]:
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return imported


def _called_names(tree: ast.AST) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_serving_and_walk_forward_share_ensemble_aggregation() -> None:
    serving = _tree("trade_rl/integrations/sb3_serving.py")
    walk_forward = _tree("trade_rl/workflows/_market_walk_forward_core.py")

    expected = "trade_rl.integrations.sb3_ensemble.predict_deterministic_mean_action"
    assert _imports(serving)["predict_deterministic_mean_action"] == expected
    assert _imports(walk_forward)["predict_deterministic_mean_action"] == expected
    assert "predict_deterministic_mean_action" in _called_names(serving)
    assert "predict_deterministic_mean_action" in _called_names(walk_forward)


def test_wrappers_do_not_reimplement_numpy_mean_aggregation() -> None:
    for path in (
        "trade_rl/integrations/sb3_serving.py",
        "trade_rl/workflows/_market_walk_forward_core.py",
    ):
        tree = _tree(path)
        numpy_aggregation_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in {"mean", "stack"}
        }
        assert numpy_aggregation_calls == set()


def test_ensemble_helper_does_not_depend_on_serving_or_workflows() -> None:
    tree = _tree("trade_rl/integrations/sb3_ensemble.py")
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("trade_rl.serving") for module in modules)
    assert not any(module.startswith("trade_rl.workflows") for module in modules)
