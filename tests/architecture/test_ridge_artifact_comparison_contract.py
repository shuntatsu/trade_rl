from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation as evaluation
import trade_rl.evaluation.runs as runs
from trade_rl.evaluation.runs import ridge_artifact_comparison as owner

ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "trade_rl" / "evaluation" / "runs" / "ridge_artifact_comparison.py"


def test_comparison_api_is_owned_and_reexported_by_runs_facade() -> None:
    for name in (
        "RidgeArtifactComparison",
        "RidgeArtifactComparisonConfig",
        "compare_ridge_artifact_shared_cash",
    ):
        assert getattr(runs, name) is getattr(owner, name)
        assert not hasattr(evaluation, name)


def test_comparison_owner_reuses_run_core_without_fitting_or_study_imports() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"), filename=str(OWNER_PATH))
    imported_modules: set[str] = set()
    imported_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_symbols.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)

    assert "trade_rl.evaluation.comparison.strategies" in imported_modules
    assert "trade_rl.data" in imported_modules
    assert "trade_rl.strategies.forecasts" in imported_modules
    assert {
        "compare_shared_cash_strategy_factories",
        "load_published_market_dataset_artifact",
        "load_ridge_forecast_artifact",
    } <= imported_symbols
    assert not any(
        module.startswith("trade_rl.evaluation.experiments")
        or module
        in {
            "trade_rl.evaluation.runs.candidate",
            "trade_rl.evaluation.runs.candidate_suite",
            "trade_rl.evaluation.runs.execute",
        }
        for module in imported_modules
    )
    assert (
        not {
            "fit_ridge_forecast",
            "fit_lightgbm_forecast",
            "fit_ppo_strategy",
            "run_lean_candidate_suite",
        }
        & imported_symbols
    )
