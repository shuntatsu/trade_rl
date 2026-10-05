"""The optional allocation family composes the existing ledger from evaluation."""

from __future__ import annotations

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_allocation_owner_and_lower_layer_imports() -> None:
    path = PACKAGE / "strategies" / "allocation.py"
    assert path.is_file()
    imports = ImportCollector(PACKAGE).collect(path)
    forbidden = (
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.risk",
        "trade_rl.strategies.forecasts",
        "trade_rl.strategies.rl",
    )
    assert not any(
        within_module(name, prefix) for name in imports for prefix in forbidden
    )


def test_execution_composition_has_no_ledger_or_compatibility_cache() -> None:
    path = PACKAGE / "evaluation" / "allocation.py"
    assert path.is_file()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(
        {"execute_interval", "apply_fill", "apply_funding", "apply_borrow"}
    )
    imports = ImportCollector(PACKAGE).collect(path)
    assert "trade_rl.simulation.targets.execution" in imports
    assert "trade_rl.risk.pretrade" in imports


def test_allocation_public_family_leaves_existing_facades_unchanged() -> None:
    import trade_rl.evaluation as evaluation
    import trade_rl.strategies as strategies
    from trade_rl.evaluation.allocation import (
        execute_nonrl_proposal,
        propose_nonrl_target,
    )
    from trade_rl.strategies.allocation import AfterCostTargetAllocator

    assert callable(execute_nonrl_proposal)
    assert callable(propose_nonrl_target)
    assert callable(AfterCostTargetAllocator)
    assert not hasattr(strategies, "AfterCostTargetAllocator")
    assert not hasattr(evaluation, "execute_nonrl_proposal")
