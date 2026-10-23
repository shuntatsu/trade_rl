"""Forecast diagnostics read declarations without fitting or economic execution."""

import ast
import importlib
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

PACKAGE = Path(__file__).resolve().parents[2] / "trade_rl"


def test_forecast_diagnostics_has_no_accounting_rl_or_learner_dependencies():
    path = PACKAGE / "evaluation" / "forecast_diagnostics.py"
    imports = ImportCollector(PACKAGE).collect(path)
    assert not any(
        within_module(module, prefix)
        for module in imports
        for prefix in (
            "trade_rl.simulation",
            "trade_rl.risk",
            "trade_rl.strategies.rl",
            "trade_rl.evaluation.rl_allocation",
            "trade_rl.integrations",
            "trade_rl.strategies.forecasts.simple_prequential",
            "torch",
            "stable_baselines3",
            "sklearn",
        )
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(
        {
            "fit",
            "predict",
            "predict_selected",
            "fit_prequential_simple_ridge",
            "from_payload",
            "FrozenSimpleReturnStream",
            "apply_fill",
            "apply_funding",
            "apply_borrow",
            "execute_interval",
        }
    )


def test_forecast_diagnostics_public_surface_stays_in_its_own_module():
    module = importlib.import_module("trade_rl.evaluation.forecast_diagnostics")
    assert module.__all__ == [
        "SimpleReturnForecastDiagnostics",
        "evaluate_simple_return_forecasts",
    ]
    from trade_rl import evaluation, strategies

    assert not hasattr(evaluation, "evaluate_simple_return_forecasts")
    assert not hasattr(strategies, "evaluate_simple_return_forecasts")
