from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.strategies as strategies
import trade_rl.strategies.rl as rl

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "trade_rl" / "strategies" / "rl" / "portfolio_runtime_scenario.py"
FORBIDDEN_IMPORT_PREFIXES = (
    "gymnasium",
    "stable_baselines3",
    "torch",
    "trade_rl.evaluation",
    "trade_rl.risk",
    "trade_rl.simulation.replay",
    "trade_rl.strategies.rl.portfolio_env",
    "trade_rl.strategies.rl.portfolio_inference_bundle",
    "trade_rl.strategies.rl.portfolio_training",
)


def _imports() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_runtime_scenario_stays_below_runtime_and_evaluation_owners() -> None:
    imports = _imports()
    assert not {
        module
        for module in imports
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    }
    assert "trade_rl.simulation.execution" in imports


def test_runtime_scenario_does_not_expand_strategy_facades() -> None:
    assert "PortfolioRuntimeExecutionScenarioV2" not in strategies.__all__
    assert "PortfolioRuntimeExecutionScenarioV2" not in rl.__all__
