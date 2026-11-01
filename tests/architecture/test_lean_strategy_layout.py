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
        "position_duration.py",
        "portfolio_adapter.py",
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
        "forecasts/prequential.py",
        "rl/__init__.py",
        "rl/intent.py",
        "rl/a2c.py",
        "rl/a2c_artifact.py",
        "rl/ppo.py",
        "rl/ppo_normalization.py",
        "rl/ppo_artifact.py",
        "rl/portfolio_runtime_scenario.py",
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


def test_supervised_cutoff_validation_has_one_forecast_owner() -> None:
    supervised_path = STRATEGIES / "forecasts" / "supervised.py"
    tree = ast.parse(supervised_path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    validator = next(
        node for node in definitions if node.name == "_validated_fit_cutoff_ns"
    )
    assert validator is not None

    training_set = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CausalForecastTrainingSet"
    )
    for owner in (validator, training_set):
        calls_shared_converter = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_datetime64_values_to_ns_exact"
            for node in ast.walk(owner)
        )
        assert calls_shared_converter
    builder = next(
        node
        for node in definitions
        if node.name == "build_causal_forecast_training_set"
    )
    for owner in (training_set, builder):
        calls_validator = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validated_fit_cutoff_ns"
            for node in ast.walk(owner)
        )
        assert calls_validator

    for relative, model_name, fit_name in (
        ("forecasts/ridge.py", "RidgeForecastModel", "fit_ridge_forecast"),
        (
            "forecasts/lightgbm.py",
            "LightGBMForecastModel",
            "fit_lightgbm_forecast",
        ),
    ):
        fit_tree = ast.parse((STRATEGIES / relative).read_text(encoding="utf-8"))
        model_class = next(
            node
            for node in fit_tree.body
            if isinstance(node, ast.ClassDef) and node.name == model_name
        )
        model_post_init = next(
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        uses_shared_cutoff_validator = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validated_fit_cutoff_ns"
            for node in ast.walk(model_post_init)
        )
        assert uses_shared_cutoff_validator

        fit_function = next(
            node
            for node in fit_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == fit_name
        )
        calls_shared_builder = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_causal_forecast_training_set"
            for node in ast.walk(fit_function)
        )
        assert calls_shared_builder


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


def test_prequential_packets_reuse_the_shared_causal_training_rows() -> None:
    path = STRATEGIES / "forecasts" / "prequential.py"
    imports = _imports(path)
    assert "trade_rl.strategies.forecasts.supervised" in imports
    assert not any(
        name.startswith(
            (
                "trade_rl.evaluation",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.runtime",
            )
        )
        for name in imports
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_causal_forecast_training_set" in called
    # Label construction stays in supervised.py: no log-return label math here.
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "log"
        and isinstance(node.value, ast.Name)
        and node.value.id in {"math", "np"}
        for node in ast.walk(tree)
    )
