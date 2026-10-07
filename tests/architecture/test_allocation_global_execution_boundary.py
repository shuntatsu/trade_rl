"""The global observer adds bounded consistency, never study/model authority."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_global_context_and_facade_owners_keep_upper_provenance_separate():
    lower = ROOT / "trade_rl/evaluation/rl_allocation/global_execution_context.py"
    upper = ROOT / "trade_rl/evaluation/allocation_global_execution.py"
    assert lower.exists() and upper.exists(), "global receipt owners are missing"
    for path in (lower, upper):
        tree = ast.parse(path.read_text("utf-8"))
        imports = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module
        }
        forbidden = (
            "torch",
            "stable_baselines3",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.final_test",
            "trade_rl.evaluation.rl_allocation.training",
            "trade_rl.evaluation.rl_allocation.scheduled_training",
        )
        assert not any(
            name == p or name.startswith(p + ".") for name in imports for p in forbidden
        )
        if path == lower:
            assert not any(
                name.startswith("trade_rl.evaluation.runs") for name in imports
            )
        else:
            assert "trade_rl.evaluation.runs.provenance" in imports


def test_lower_collector_is_concrete_and_not_a_second_callback_protocol():
    lower = ROOT / "trade_rl/evaluation/rl_allocation/global_execution_context.py"
    tree = ast.parse(lower.read_text("utf-8"))
    collector = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "GlobalAllocationExecutionCollector"
    )
    assert not collector.bases and not collector.decorator_list
    assert "Protocol" not in lower.read_text(
        "utf-8"
    ) and "runtime_checkable" not in lower.read_text("utf-8")
    imports = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "trade_rl.evaluation.allocation_global_execution" not in imports
    assert "trade_rl.evaluation.rl_allocation.global_walk_forward" not in imports
    validate = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_global_collector"
    )
    assert "type(collector) is not GlobalAllocationExecutionCollector" in ast.unparse(
        validate
    )
