import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl" / "evaluation" / "allocation_cash_control.py"


def imports():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"), filename=str(OWNER))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_cash_control_does_not_train_select_or_open_final_data():
    actual = imports()
    forbidden = (
        "stable_baselines3",
        "torch",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.gates",
        "trade_rl.evaluation.final_test",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )
