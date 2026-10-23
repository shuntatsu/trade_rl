"""Simple predictions compose canonical allocation without training in admission."""

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

PACKAGE = Path(__file__).resolve().parents[2] / "trade_rl"


def test_simple_forecasts_have_no_ledger_rl_or_evaluation_dependency():
    collector = ImportCollector(PACKAGE)
    for name in ("_ridge_math", "simple_return", "simple_stream", "simple_prequential"):
        imports = collector.collect(PACKAGE / "strategies" / "forecasts" / f"{name}.py")
        assert not any(
            within_module(module, prefix)
            for module in imports
            for prefix in (
                "trade_rl.evaluation",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.strategies.rl",
            )
        )


def test_forecast_admission_uses_allocation_without_fit_or_account_updates():
    path = PACKAGE / "evaluation" / "forecast_allocation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert {
        "propose_nonrl_target",
        "execute_nonrl_proposal",
        "validate_processing_clock",
    } <= calls
    assert calls.isdisjoint(
        {
            "fit",
            "fit_prequential_simple_ridge",
            "execute_interval",
            "apply_fill",
            "apply_funding",
            "apply_borrow",
        }
    )
    from trade_rl import evaluation, strategies

    assert not hasattr(evaluation, "execute_forecast_proposal")
    assert not hasattr(strategies, "fit_prequential_simple_ridge")
