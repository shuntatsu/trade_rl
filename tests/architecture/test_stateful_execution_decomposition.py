from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "trade_rl" / "simulation" / "stateful_execution.py"
REQUIRED_MODULES = {
    "stateful_runtime.py": "StatefulExecutionRuntime",
    "stateful_bar_lifecycle.py": "StatefulBarLifecycle",
    "stateful_order_transitions.py": "StatefulOrderTransitionProcessor",
    "stateful_symbol_fills.py": "StatefulSymbolFillProcessor",
}


def _execute_node() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_stateful_orders"
    )


def test_stateful_execution_phase_modules_exist() -> None:
    directory = SOURCE.parent
    for filename, class_name in REQUIRED_MODULES.items():
        path = directory / filename
        assert path.is_file(), filename
        assert class_name in path.read_text(encoding="utf-8")


def test_execute_stateful_orders_is_bounded_orchestration() -> None:
    node = _execute_node()
    assert node.end_lineno is not None
    assert node.end_lineno - node.lineno + 1 <= 180
    source = ast.unparse(node)
    for class_name in REQUIRED_MODULES.values():
        assert class_name in source
    for low_level in (
        "OrderAdmissionPolicy",
        "select_bar_path",
        "evaluate_trigger",
        "allocate_symbol_capacity",
        "apply_dividend",
        "apply_cash_interest",
    ):
        assert low_level not in source


def test_stateful_execution_result_remains_public_in_orchestration_module() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    result_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "StatefulExecutionResult" in result_classes
