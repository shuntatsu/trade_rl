import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl/evaluation/allocation_nonrl_walk_forward.py"


def test_fixed_nonrl_consumer_reuses_execution_without_learning_or_selection():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "trade_rl.evaluation.rl_allocation.continuous_walk_forward" in imports
    assert "trade_rl.evaluation.allocation_comparison_evidence" in imports
    forbidden = (
        "torch",
        "stable_baselines3",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.allocation_selection",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
        "trade_rl.strategies.forecasts.simple_prequential",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )
