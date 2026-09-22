from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.strategies as strategies
import trade_rl.strategies.dataset_scope as shared_dataset_scope
from trade_rl.strategies.forecasts import supervised as supervised_forecasts

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ROOT / "trade_rl" / "strategies"

EXPECTED_PUBLIC_API = {
    "A2CFitMetadata",
    "A2CIntentStrategy",
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
    "fit_a2c_strategy",
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
        "carry.py",
        "rules/__init__.py",
        "rules/trend.py",
        "rules/mean_reversion.py",
        "rules/channel_breakout.py",
        "forecasts/__init__.py",
        "forecasts/controller.py",
        "forecasts/supervised.py",
        "forecasts/ridge.py",
        "forecasts/lightgbm.py",
        "rl/__init__.py",
        "rl/intent.py",
        "rl/a2c.py",
        "rl/a2c_artifact.py",
        "rl/ppo.py",
        "rl/ppo_normalization.py",
        "rl/ppo_artifact.py",
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


def test_ppo_and_a2c_use_the_shared_three_action_intent_adapter() -> None:
    from trade_rl.strategies import A2CIntentStrategy, PPOIntentStrategy
    from trade_rl.strategies.rl.intent import _ThreeActionIntentStrategy

    assert issubclass(PPOIntentStrategy, _ThreeActionIntentStrategy)
    assert issubclass(A2CIntentStrategy, _ThreeActionIntentStrategy)


def test_a2c_artifact_module_owns_its_inference_bundle_api() -> None:
    from trade_rl.strategies.rl import a2c_artifact

    assert set(a2c_artifact.__all__) == {
        "load_a2c_inference_bundle",
        "save_a2c_inference_bundle",
    }


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



def _imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_shared_rl_observation_contract_has_one_owner() -> None:
    intent = STRATEGIES / "rl" / "intent.py"
    ppo = STRATEGIES / "rl" / "ppo.py"
    consumers = (
        STRATEGIES / "rl" / "a2c_artifact.py",
        STRATEGIES / "rl" / "ppo_artifact.py",
        ROOT / "trade_rl" / "evaluation" / "runs" / "artifact.py",
        ROOT / "trade_rl" / "evaluation" / "experiments" / "contracts" / "run.py",
    )

    intent_source = intent.read_text(encoding="utf-8")
    ppo_source = ppo.read_text(encoding="utf-8")

    assert "def ppo_observation_contract_payload" in intent_source
    assert 'PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"' in intent_source
    assert "PPO_GLOBAL_FEATURE_NAMES" in intent_source

    # PPO keeps the historical public names only as imports/re-exports.
    assert "def ppo_observation_contract_payload" not in ppo_source
    assert 'PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"' not in ppo_source

    contract_names = {
        "PPO_GLOBAL_FEATURE_NAMES",
        "PPO_OBSERVATION_SCHEMA",
        "ppo_observation_contract_payload",
    }
    for path in consumers:
        from_intent = _imported_names(path, "trade_rl.strategies.rl.intent")
        from_ppo = _imported_names(path, "trade_rl.strategies.rl.ppo")
        assert from_intent & contract_names
        assert not (from_ppo & contract_names)
