import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl" / "evaluation" / "allocation_scenario_identity.py"


def imports():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"), filename=str(OWNER))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_scenario_identity_is_pure_and_does_not_execute_or_load_models():
    actual = imports()
    forbidden = (
        "numpy",
        "stable_baselines3",
        "torch",
        "trade_rl.simulation",
        "trade_rl.risk",
        "trade_rl.strategies",
        "trade_rl.evaluation.rl_allocation",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )
