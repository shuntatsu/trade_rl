from __future__ import annotations

import ast

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT


def _tree(path: str) -> ast.Module:
    return ast.parse((PYTHON_SOURCE_ROOT / path).read_text(encoding="utf-8"))


def _imports(tree: ast.AST) -> dict[str, str]:
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return imported


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _called_names(tree: ast.AST) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _numpy_aggregation_calls(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in {"mean", "stack"}
    }


def test_serving_and_walk_forward_share_ensemble_aggregation() -> None:
    serving = _tree("integrations/sb3_serving.py")
    walk_forward = _tree("workflows/_market_walk_forward_core.py")

    expected = "trade_rl.integrations.sb3_ensemble.predict_deterministic_mean_action"
    assert _imports(serving)["predict_deterministic_mean_action"] == expected
    assert _imports(walk_forward)["predict_deterministic_mean_action"] == expected

    wrappers = (
        _class_method(serving, "_SB3EnsemblePolicy", "predict"),
        _class_method(serving, "_SB3StructuredSequenceEnsemblePolicy", "predict"),
        _class_method(walk_forward, "_DeterministicMeanPolicy", "predict"),
    )
    for wrapper in wrappers:
        assert "predict_deterministic_mean_action" in _called_names(wrapper)


def test_wrapper_predict_methods_do_not_reimplement_numpy_mean_aggregation() -> None:
    serving = _tree("integrations/sb3_serving.py")
    walk_forward = _tree("workflows/_market_walk_forward_core.py")
    wrappers = (
        _class_method(serving, "_SB3EnsemblePolicy", "predict"),
        _class_method(serving, "_SB3StructuredSequenceEnsemblePolicy", "predict"),
        _class_method(walk_forward, "_DeterministicMeanPolicy", "predict"),
    )

    for wrapper in wrappers:
        assert _numpy_aggregation_calls(wrapper) == set()


def test_ensemble_helper_does_not_depend_on_serving_or_workflows() -> None:
    tree = _tree("integrations/sb3_ensemble.py")
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("trade_rl.serving") for module in modules)
    assert not any(module.startswith("trade_rl.workflows") for module in modules)
