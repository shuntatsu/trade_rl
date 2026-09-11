from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.strategies as strategies
import trade_rl.strategies.dataset_scope as shared_dataset_scope
from trade_rl.strategies.forecasts import supervised as supervised_forecasts

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ROOT / "trade_rl" / "strategies"

EXPECTED_PUBLIC_API = {
    "CausalForecastTrainingSet",
    "ConstantIntentStrategy",
    "ForecastIntentConfig",
    "ForecastIntentController",
    "LightGBMForecastModel",
    "LightGBMForecastStrategy",
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "PPOIntentStrategy",
    "PPOTradingEnv",
    "PositionIntent",
    "RidgeForecastModel",
    "RidgeForecastStrategy",
    "SingleSymbolStrategy",
    "StrategyObservation",
    "TrendIntentConfig",
    "TrendIntentStrategy",
    "build_causal_forecast_training_set",
    "fit_lightgbm_forecast",
    "fit_ppo_strategy",
    "fit_ridge_forecast",
    "target_weight_for_intent",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_strategy_family_packages_exist() -> None:
    for relative in (
        "dataset_scope.py",
        "rules/__init__.py",
        "rules/trend.py",
        "rules/mean_reversion.py",
        "forecasts/__init__.py",
        "forecasts/controller.py",
        "forecasts/supervised.py",
        "forecasts/ridge.py",
        "forecasts/lightgbm.py",
        "rl/__init__.py",
        "rl/ppo.py",
    ):
        assert (STRATEGIES / relative).is_file(), relative


def test_old_flat_strategy_modules_are_absent() -> None:
    for name in (
        "forecast.py",
        "supervised.py",
        "ridge.py",
        "lightgbm.py",
        "trend.py",
        "mean_reversion.py",
        "ppo.py",
    ):
        assert not (STRATEGIES / name).exists(), name


def test_strategy_package_preserves_public_api() -> None:
    assert set(strategies.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(strategies, name), name


def test_supervised_module_preserves_dataset_scope_exports() -> None:
    assert {
        "validated_feature_indices",
        "validated_symbol_indices",
    }.issubset(set(supervised_forecasts.__all__))
    assert (
        supervised_forecasts.validated_feature_indices
        is shared_dataset_scope.validated_feature_indices
    )
    assert (
        supervised_forecasts.validated_symbol_indices
        is shared_dataset_scope.validated_symbol_indices
    )


def test_strategy_families_do_not_depend_on_evaluation() -> None:
    for family in ("rules", "forecasts", "rl"):
        root = STRATEGIES / family
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            imports = _imports(path)
            assert not any(
                name.startswith("trade_rl.evaluation") for name in imports
            ), path
