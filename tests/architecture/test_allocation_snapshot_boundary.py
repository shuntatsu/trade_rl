"""Snapshot evidence stays below orchestration; its producer owns no ledger."""

from __future__ import annotations

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

PACKAGE = Path(__file__).resolve().parents[2] / "trade_rl"


def test_snapshot_dto_has_no_evaluation_risk_simulation_or_learning_dependency():
    path = PACKAGE / "strategies" / "allocation_snapshot.py"
    imports = ImportCollector(PACKAGE).collect(path)
    assert not any(
        within_module(name, prefix)
        for name in imports
        for prefix in (
            "trade_rl.evaluation",
            "trade_rl.simulation",
            "trade_rl.risk",
            "trade_rl.strategies.rl",
            "gymnasium",
            "stable_baselines3",
            "torch",
        )
    )


def test_snapshot_producer_has_no_account_order_fit_or_execution_mutation():
    path = PACKAGE / "evaluation" / "allocation_snapshot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(
        {
            "execute_interval",
            "apply_fill",
            "execute",
            "charge_borrow",
            "apply_funding",
            "fit",
            "learn",
            "reset_random_state",
            "add",
            "cancel",
            "replace",
        }
    )
