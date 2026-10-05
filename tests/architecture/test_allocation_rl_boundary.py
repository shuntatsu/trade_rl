"""Allocation learning composes canonical transitions from evaluation only."""

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_pure_allocation_rl_owners_do_not_reverse_the_package_dependency():
    paths = [
        PACKAGE / "strategies/allocation_action.py",
        *sorted((PACKAGE / "strategies/rl").glob("allocation_*.py")),
    ]
    forbidden = ("trade_rl.evaluation", "trade_rl.simulation", "trade_rl.risk")
    for path in paths:
        imports = ImportCollector(PACKAGE).collect(path)
        assert not any(
            within_module(name, prefix) for name in imports for prefix in forbidden
        ), path


def test_learning_environment_uses_common_action_execution_without_a_second_ledger():
    path = PACKAGE / "evaluation/rl_allocation/env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "execute_allocation_action" in calls
    assert calls.isdisjoint(
        {"apply_fill", "apply_funding", "apply_borrow", "execute_interval"}
    )
    assert "test_allocation_ppo_runtime.py" in (
        ROOT / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
