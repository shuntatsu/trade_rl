from __future__ import annotations

import ast
from pathlib import Path

from trade_rl.strategies import forecasts
from trade_rl.strategies.forecasts import ridge_artifact

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "trade_rl" / "strategies" / "forecasts" / "ridge_artifact.py"


def test_ridge_artifact_owns_only_strategy_model_persistence() -> None:
    tree = ast.parse(ARTIFACT.read_text(encoding="utf-8"), filename=str(ARTIFACT))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(name.startswith("trade_rl.evaluation") for name in imports)
    assert not any("lightgbm" in name.casefold() for name in imports)
    assert not any("pickle" in name.casefold() for name in imports)


def test_ridge_artifact_public_api_is_owned_by_forecast_facade() -> None:
    for name in (
        "RidgeForecastArtifactBinding",
        "publish_ridge_forecast_artifact",
        "load_ridge_forecast_artifact",
    ):
        assert getattr(forecasts, name) is getattr(ridge_artifact, name)
