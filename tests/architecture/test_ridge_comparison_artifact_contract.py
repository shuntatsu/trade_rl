from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation as evaluation
import trade_rl.evaluation.runs as runs
from trade_rl.evaluation.runs import ridge_comparison_artifact as owner

ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "trade_rl" / "evaluation" / "runs" / "ridge_comparison_artifact.py"


def test_comparison_artifact_is_owned_and_reexported_by_runs_facade() -> None:
    for name in (
        "LoadedRidgeComparisonArtifact",
        "load_ridge_comparison_artifact",
        "publish_ridge_comparison_artifact",
    ):
        assert getattr(runs, name) is getattr(owner, name)
        assert not hasattr(evaluation, name)


def test_artifact_owner_uses_public_comparison_and_ledger_owners_only() -> None:
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

    assert "trade_rl.evaluation.runs.ridge_artifact_comparison" in imported_modules
    assert "trade_rl.evaluation.comparison.strategies" in imported_modules
    assert "trade_rl.evaluation.metrics" in imported_modules
    assert "trade_rl.evaluation.replay" not in imported_modules
    assert not any(
        module.startswith("trade_rl.evaluation.experiments")
        or module.startswith("trade_rl.rl")
        or module.startswith("stable_baselines3")
        for module in imported_modules
    )
    assert (
        not {
            "fit_ridge_forecast",
            "fit_lightgbm_forecast",
            "fit_ppo_strategy",
            "run_lean_candidate_suite",
            "_create_verified_market_order_profile",
        }
        & imported_symbols
    )
