"""Closing composes native execution; it cannot become a training/final gateway."""

import ast
from pathlib import Path


def test_terminal_owner_has_no_training_study_or_second_ledger_dependency():
    root = Path(__file__).resolve().parents[2]
    path = root / "trade_rl/evaluation/allocation_terminal_execution.py"
    assert path.exists(), "native terminal closing owner is missing"
    tree = ast.parse(path.read_text("utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = (
        "stable_baselines3",
        "torch",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.simulation.accounting",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )
    calls = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "run_declared_global_allocation_execution" in calls
    assert "_execute_allocation_target" in calls
    assert "native_allocation_execution_facts" in calls
    assert not calls & {
        "BookState",
        "MarketExecutor",
        "env.step",
        "env.reset",
        "book.clone",
    }
    assigned = {
        ast.unparse(target)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
    }
    assert not assigned & {
        "env.index",
        "env.stop_index",
        "env._terminated",
        "env.book",
        "env.order_book",
    }


def test_terminal_capability_and_limits_have_current_documentation():
    root = Path(__file__).resolve().parents[2]
    for name in (
        "architecture/lean-core.md",
        "architecture/package-boundaries.md",
        "architecture/research-assurance.md",
        "research/current-status.md",
    ):
        text = (root / "docs" / name).read_text("utf-8")
        assert "allocation_terminal_execution" in text
