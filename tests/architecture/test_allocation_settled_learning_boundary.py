"""Settled timing declarations belong below evaluation and the optional learner."""

import ast
from importlib import import_module
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_settled_clock_has_no_runtime_or_optional_learner_dependency():
    path = PACKAGE / "strategies/rl/allocation_settled_clock.py"
    assert path.is_file(), "the pure settled allocation clock owner is missing"
    imports = ImportCollector(PACKAGE).collect(path)
    assert all(
        not name.startswith("trade_rl.") or name.startswith("trade_rl.artifacts.")
        for name in imports
    )
    assert not any(
        name.split(".")[0] in {"numpy", "gymnasium", "torch", "stable_baselines3"}
        for name in imports
    )
    module = import_module("trade_rl.strategies.rl.allocation_settled_clock")
    assert module.__all__ == ["SettledAllocationClockContract"]


def test_settled_composition_uses_canonical_execution_and_shared_reserve():
    path = PACKAGE / "evaluation/rl_allocation/settled_env.py"
    tree = ast.parse(path.read_text("utf8"))
    imports = ImportCollector(PACKAGE).collect(path)
    assert not any(
        name.startswith(prefix)
        for name in imports
        for prefix in (
            "stable_baselines3",
            "torch",
            "trade_rl.evaluation.rl_allocation.env",
            "trade_rl.evaluation.rl_allocation.training",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.final_test",
            "trade_rl.evaluation.allocation_terminal_execution",
        )
    )
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert [ast.unparse(base) for base in owner.bases] == ["gym.Env"]
    step = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    calls = {
        ast.unparse(node.func) for node in ast.walk(step) if isinstance(node, ast.Call)
    }
    assert {"execute_allocation_action", "_iter_reserved_allocation_closing"} <= calls
    assert not calls & {
        "BookState",
        "MarketExecutor",
        "PreTradeRisk",
        "self.reset",
        "self.executor.reset_random_state",
    }


def test_settled_recipe_stays_lower_and_source_has_no_learner_or_legacy_owner():
    recipe = PACKAGE / "strategies/rl/allocation_recipe_v4.py"
    imports = ImportCollector(PACKAGE).collect(recipe)
    assert not any(
        name.startswith(prefix)
        for name in imports
        for prefix in (
            "trade_rl.evaluation",
            "trade_rl.simulation",
            "trade_rl.risk",
            "torch",
            "stable_baselines3",
            "gymnasium",
        )
    )
    source = PACKAGE / "evaluation/rl_allocation/settled_training_source.py"
    imports = ImportCollector(PACKAGE).collect(source)
    assert not any(
        name.startswith(prefix)
        for name in imports
        for prefix in (
            "trade_rl.evaluation.rl_allocation.env",
            "trade_rl.evaluation.allocation_terminal_execution",
            "torch",
            "stable_baselines3",
            "gymnasium",
        )
    )
