from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import (
    MarketCalendarKind,
    MarketDataset,
    publish_market_dataset_artifact,
)
from trade_rl.data.artifacts.publication import PublishedDatasetArtifact
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    publish_ridge_forecast_artifact,
)
from trade_rl.strategies.forecasts.controller import (
    CostAwareForecastIntentController,
    ForecastIntentController,
)


def _comparison_api():
    try:
        module = importlib.import_module(
            "trade_rl.evaluation.runs.ridge_artifact_comparison"
        )
    except ImportError as error:
        pytest.fail(
            f"the frozen Ridge shared-cash comparison API is unavailable: {error}"
        )
    return module


def test_mean_absolute_avoids_overflow_before_finite_average() -> None:
    api = _comparison_api()

    assert api._mean_absolute((1e308, 1e308)) == 1e308


def test_mse_skill_is_defined_for_constant_nonzero_targets() -> None:
    api = _comparison_api()

    metrics = api._error_metrics((0.0, 0.0), (0.1, 0.1))

    assert metrics.zero_rmse == pytest.approx(0.1)
    assert metrics.mse_skill == pytest.approx(0.0)


def test_mse_skill_validation_names_zero_forecast_error() -> None:
    api = _comparison_api()

    with pytest.raises(ValueError, match="zero-forecast squared error is zero"):
        api.ForecastErrorMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _dataset() -> MarketDataset:
    offsets_minutes = np.asarray([0, 30, 120, 180, 360, 390], dtype=np.int64)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        offsets_minutes * np.timedelta64(1, "m")
    )
    signal = np.asarray([-0.20, -0.15, 0.20, 0.25, -0.20, -0.25], dtype=np.float32)
    secondary = np.asarray([0.4, 0.3, 0.2, 0.1, -0.1, -0.2], dtype=np.float32)
    row_features = np.stack((signal, secondary), axis=1)
    features = np.repeat(row_features[:, None, :], 2, axis=1)
    close = np.asarray(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [102.0, 196.0],
            [103.0, 194.0],
            [104.0, 192.0],
            [105.0, 190.0],
        ],
        dtype=np.float64,
    )
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((6, 2), 100_000.0, dtype=np.float64),
        funding_rate=np.zeros((6, 2), dtype=np.float64),
        tradable=np.ones((6, 2), dtype=np.bool_),
        feature_available=np.ones((6, 2, 2), dtype=np.bool_),
        feature_names=("momentum", "secondary"),
        global_feature_names=("regime",),
        periods_per_year=252,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
    )
    return dataset.with_content_identity({"fixture": "ridge-four-arm-session"})


def _model(dataset: MarketDataset, *, cutoff_index: int = 2) -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(1, 0),
        feature_mean=np.asarray([0.0, 0.0], dtype=np.float64),
        feature_scale=np.asarray([2.0, 1.0], dtype=np.float64),
        coefficients=np.asarray([0.0, 1.0], dtype=np.float64),
        intercept=0.0,
        horizon_hours=2,
        alpha=0.5,
        n_samples=4,
        fit_cutoff=dataset.timestamps[cutoff_index].astype("datetime64[ns]"),
    )


def _binding(
    dataset: MarketDataset,
    artifact: PublishedDatasetArtifact,
    model: RidgeForecastModel,
) -> RidgeForecastArtifactBinding:
    symbol_indices = (1, 0)
    return RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=artifact.schema_version,
        dataset_artifact_sha256=artifact.artifact_digest,
        fit_symbol_indices=symbol_indices,
        fit_symbol_names=tuple(dataset.symbols[index] for index in symbol_indices),
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(model.fit_cutoff.astype("int64")),
        horizon_hours=model.horizon_hours,
        alpha_hex=float(model.alpha).hex(),
        n_samples=model.n_samples,
        vintage_id="literal-test-vintage",
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        runtime_sha256="c" * 64,
    )


def _config(api, **overrides):
    values = {
        "start_index": 2,
        "stop_index": 5,
        "gross_budget": 0.5,
        "initial_capital": 10_000.0,
        "entry_threshold": 0.10,
        "exit_threshold": 0.02,
        "one_way_switch_cost": 0.01,
        "execution_cost": ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0001,
            max_participation_rate=1.0,
        ),
        "risk": PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        ),
        "market_order_profile": None,
        "minimum_hold_bars": 0,
    }
    values.update(overrides)
    return api.RidgeArtifactComparisonConfig(**values)


def _fake_dataset_artifact(tmp_path: Path) -> PublishedDatasetArtifact:
    return PublishedDatasetArtifact(
        root=tmp_path / "dataset",
        manifest_path=tmp_path / "dataset" / "manifest.json",
        arrays_path=tmp_path / "dataset" / "arrays.npz",
        artifact_digest="d" * 64,
    )


def _published_comparison_inputs(tmp_path: Path, dataset: MarketDataset):
    api = _comparison_api()
    dataset_root = tmp_path / "dataset"
    dataset_artifact = publish_market_dataset_artifact(dataset_root, dataset)
    model = _model(dataset)
    binding = _binding(dataset, dataset_artifact, model)
    ridge_root = tmp_path / "ridge-model"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)
    return api, dataset_root, ridge_root, ridge_digest, binding, model


def test_ridge_artifact_shared_cash_comparison_module_is_available() -> None:
    module = _comparison_api()
    assert callable(module.compare_ridge_artifact_shared_cash)


def test_four_arm_comparison_loads_one_model_and_returns_terminal_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _comparison_api()
    candidate_suite = importlib.import_module(
        "trade_rl.evaluation.runs.candidate_suite"
    )

    def unexpected_fit(*_args, **_kwargs):
        raise AssertionError("the frozen artifact comparison must not fit models")

    for fit_name in (
        "fit_ridge_forecast",
        "fit_lightgbm_forecast",
        "fit_ppo_strategy",
    ):
        monkeypatch.setattr(candidate_suite, fit_name, unexpected_fit)

    dataset = _dataset()
    api, dataset_root, ridge_root, ridge_digest, binding, model = (
        _published_comparison_inputs(tmp_path, dataset)
    )

    dataset_loads: list[Path] = []
    actual_dataset_loader = api.load_published_market_dataset_artifact

    def record_dataset_load(root: str | Path):
        dataset_loads.append(Path(root))
        return actual_dataset_loader(root)

    monkeypatch.setattr(
        api, "load_published_market_dataset_artifact", record_dataset_load
    )

    model_loads: list[RidgeForecastModel] = []
    actual_model_loader = api.load_ridge_forecast_artifact

    def record_model_load(*args, **kwargs):
        loaded = actual_model_loader(*args, **kwargs)
        model_loads.append(loaded)
        return loaded

    monkeypatch.setattr(api, "load_ridge_forecast_artifact", record_model_load)

    adapters = []
    actual_strategy = api.RidgeForecastStrategy

    class RecordingRidgeStrategy(actual_strategy):
        def __init__(self, strategy_model, **kwargs):
            super().__init__(strategy_model, **kwargs)
            adapters.append(self)

    monkeypatch.setattr(api, "RidgeForecastStrategy", RecordingRidgeStrategy)

    comparator_calls: list[
        tuple[MarketDataset, tuple[str, ...], dict[str, object]]
    ] = []
    actual_comparator = api.compare_shared_cash_strategy_factories

    def record_comparison(dataset_arg, factories, **kwargs):
        comparator_calls.append((dataset_arg, tuple(factories), kwargs))
        return actual_comparator(dataset_arg, factories, **kwargs)

    monkeypatch.setattr(
        api, "compare_shared_cash_strategy_factories", record_comparison
    )

    config = _config(api)
    result = api.compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=config,
    )

    arm_names = (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )
    assert dataset_loads == [dataset_root]
    assert len(model_loads) == 1
    assert model_loads[0] is adapters[0].model
    assert len(adapters) == dataset.n_symbols * 3
    assert len({id(adapter) for adapter in adapters}) == len(adapters)
    assert all(adapter.model is model_loads[0] for adapter in adapters)
    legacy_adapters = adapters[: dataset.n_symbols]
    zero_cost_adapters = adapters[dataset.n_symbols : 2 * dataset.n_symbols]
    positive_cost_adapters = adapters[2 * dataset.n_symbols :]
    assert all(
        type(adapter.controller) is ForecastIntentController
        for adapter in legacy_adapters
    )
    assert all(
        type(adapter.controller) is CostAwareForecastIntentController
        and adapter.controller.one_way_switch_cost == 0.0
        for adapter in zero_cost_adapters
    )
    assert all(
        type(adapter.controller) is CostAwareForecastIntentController
        and adapter.controller.one_way_switch_cost == config.one_way_switch_cost
        for adapter in positive_cost_adapters
    )
    assert len(comparator_calls) == 1
    called_dataset, called_arms, kwargs = comparator_calls[0]
    assert called_dataset.dataset_id == dataset.dataset_id
    assert called_arms == arm_names
    risk = kwargs.pop("risk")
    assert type(risk) is PreTradeRisk
    assert risk.config is config.risk
    assert kwargs == {
        "start_index": config.start_index,
        "stop_index": config.stop_index,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
        "execution_cost": config.execution_cost,
        "market_order_profile": config.market_order_profile,
        "minimum_hold_bars": config.minimum_hold_bars,
        "settle_terminal_position": True,
        "capture_ledger_evidence": True,
    }

    assert result.dataset_artifact.root == dataset_root
    assert result.dataset_artifact.artifact_digest == binding.dataset_artifact_sha256
    assert result.dataset_id == dataset.dataset_id
    assert result.ridge_artifact_digest == ridge_digest
    assert result.binding == binding
    assert result.config is config
    assert result.symbols == ("BTCUSDT", "ETHUSDT")
    assert result.symbols == dataset.symbols
    assert result.evaluation_start_ns == 1_767_232_800_000_000_000
    assert result.evaluation_stop_ns == 1_767_249_000_000_000_000
    assert not dataset.regular_cadence
    assert dataset.elapsed_hours(config.start_index, 4) == pytest.approx(4.0)
    assert result.binding.horizon_hours == 2
    assert tuple(entry.name for entry in result.entries) == arm_names
    assert len({id(entry.replay.book) for entry in result.entries}) == len(arm_names)
    for entry in result.entries:
        ledger = entry.replay.ledger_evidence
        assert ledger is not None
        assert ledger.dataset_id == dataset.dataset_id
        assert ledger.start_index == config.start_index
        assert ledger.stop_index == config.stop_index
        assert ledger.terminal_exact_quantities == ("0", "0")
        assert ledger.active_order_remainders == ()
        assert entry.metrics.total_cost == pytest.approx(ledger.final_total_cost)
        assert entry.metrics.funding_pnl == pytest.approx(ledger.final_funding_pnl)
        assert entry.metrics.borrow_cost == pytest.approx(ledger.final_borrow_cost)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "e" * 64),
        ("dataset_schema_version", "market_dataset_other"),
        ("dataset_artifact_sha256", "f" * 64),
        ("feature_names", ("secondary", "momentum")),
        ("feature_indices", (0,)),
        ("fit_symbol_indices", (0, 1)),
        ("fit_symbol_names", ("BTCUSDT", "ETHUSDT")),
    ],
)
def test_binding_mismatch_fails_before_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    api = _comparison_api()
    dataset = _dataset()
    artifact = _fake_dataset_artifact(tmp_path)
    model = _model(dataset)
    binding = _binding(dataset, artifact, model)
    bad_binding = replace(binding, **{field: value})
    comparator_calls: list[object] = []

    monkeypatch.setattr(
        api,
        "load_published_market_dataset_artifact",
        lambda _root: (artifact, dataset),
    )
    monkeypatch.setattr(api, "load_ridge_forecast_artifact", lambda *_a, **_k: model)

    def forbidden_comparison(*_args, **_kwargs):
        comparator_calls.append(True)
        raise AssertionError("comparator must not run for a mismatched binding")

    monkeypatch.setattr(
        api, "compare_shared_cash_strategy_factories", forbidden_comparison
    )
    with pytest.raises((TypeError, ValueError)):
        api.compare_ridge_artifact_shared_cash(
            dataset_root=artifact.root,
            ridge_artifact_root=tmp_path / "model",
            expected_ridge_digest="a" * 64,
            expected_binding=bad_binding,
            config=_config(api),
        )
    assert comparator_calls == []


def test_model_cutoff_after_decision_start_fails_before_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _comparison_api()
    dataset = _dataset()
    artifact = _fake_dataset_artifact(tmp_path)
    model = _model(dataset, cutoff_index=3)
    binding = _binding(dataset, artifact, model)
    comparator_calls: list[object] = []

    monkeypatch.setattr(
        api,
        "load_published_market_dataset_artifact",
        lambda _root: (artifact, dataset),
    )
    monkeypatch.setattr(api, "load_ridge_forecast_artifact", lambda *_a, **_k: model)
    monkeypatch.setattr(
        api,
        "compare_shared_cash_strategy_factories",
        lambda *_a, **_k: comparator_calls.append(True),
    )

    with pytest.raises(ValueError, match="cutoff"):
        api.compare_ridge_artifact_shared_cash(
            dataset_root=artifact.root,
            ridge_artifact_root=tmp_path / "model",
            expected_ridge_digest="a" * 64,
            expected_binding=binding,
            config=_config(api),
        )
    assert comparator_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gross_budget", True),
        ("gross_budget", np.bool_(True)),
        ("gross_budget", float("nan")),
        ("gross_budget", float("inf")),
        ("initial_capital", False),
        ("initial_capital", float("inf")),
        ("initial_capital", 0.0),
        ("entry_threshold", True),
        ("entry_threshold", np.bool_(True)),
        ("entry_threshold", float("nan")),
        ("exit_threshold", float("inf")),
        ("one_way_switch_cost", False),
        ("one_way_switch_cost", 0.0),
        ("one_way_switch_cost", float("nan")),
        ("start_index", True),
        ("stop_index", 4.0),
        ("minimum_hold_bars", True),
        ("risk", PreTradeRisk()),
    ],
)
def test_comparison_config_rejects_boolean_and_invalid_numeric_values(
    field: str,
    value: object,
) -> None:
    api = _comparison_api()
    with pytest.raises((TypeError, ValueError)):
        _config(api, **{field: value})


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("execution", "fee_rate", True),
        ("execution", "fee_rate", np.bool_(True)),
        ("execution", "trigger_volume_fractions", (1.0, True, 0.25, 0.0)),
        (
            "execution",
            "trigger_volume_fractions",
            (1.0, np.bool_(True), 0.25, 0.0),
        ),
        ("risk", "max_gross", True),
        ("risk", "max_gross", np.bool_(True)),
        ("risk", "max_turnover", False),
        ("risk", "max_turnover", np.bool_(False)),
    ],
)
def test_comparison_config_rejects_nested_boolean_numeric_values(
    owner: str,
    field: str,
    value: object,
) -> None:
    api = _comparison_api()
    execution_cost = ExecutionCostConfig()
    risk = PreTradeRiskConfig()
    if owner == "execution":
        execution_cost = replace(execution_cost, **{field: value})
    else:
        risk = replace(risk, **{field: value})

    with pytest.raises(ValueError, match="boolean"):
        _config(api, execution_cost=execution_cost, risk=risk)


def test_comparison_config_preserves_nested_boolean_policy_flags() -> None:
    api = _comparison_api()
    execution_cost = ExecutionCostConfig(
        allow_short=False,
        partial_fill_carry=False,
        processing_bar_volume_capacity=False,
    )
    risk = PreTradeRiskConfig(emergency_turnover_override=False)

    config = _config(api, execution_cost=execution_cost, risk=risk)

    assert config.execution_cost is execution_cost
    assert config.risk is risk


def test_opt_in_forecast_diagnostics_scores_only_shared_mature_pairs(
    tmp_path: Path,
) -> None:
    api = _comparison_api()
    base = _dataset()
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    availability = base.feature_available.copy()
    availability[2, 1, :] = False
    dataset = replace(
        base,
        timestamps=timestamps,
        available_at=np.repeat(timestamps[:, None], 2, axis=1),
        feature_available=availability,
        feature_staleness=np.where(availability, 0.0, 1.0).astype(np.float32),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "ridge-forecast-diagnostics"})
    dataset_root = tmp_path / "dataset"
    dataset_artifact = publish_market_dataset_artifact(dataset_root, dataset)
    model = _model(dataset)
    binding = _binding(dataset, dataset_artifact, model)
    ridge_root = tmp_path / "ridge-model"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)

    result = api.compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=_config(api, include_forecast_diagnostics=True),
    )

    diagnostics = result.forecast_diagnostics
    assert diagnostics is not None
    assert diagnostics.horizon_hours == model.horizon_hours == 2
    assert diagnostics.fit_cutoff_ns == int(model.fit_cutoff.astype("int64"))
    assert diagnostics.evaluation_start_index == 2
    assert diagnostics.evaluation_stop_index == 5
    assert diagnostics.scoring_as_of_ns == int(
        dataset.timestamps[5].astype("datetime64[ns]").astype("int64")
    )
    assert diagnostics.candidate_count == 4
    assert diagnostics.eligible_count == 1
    assert dict(diagnostics.excluded_counts) == {
        "feature_unavailable_or_nonfinite": 1,
        "starting_close_unavailable_at_decision": 0,
        "target_not_mature_by_scoring_as_of": 2,
        "target_close_unavailable_by_scoring_as_of": 0,
    }
    assert diagnostics.pooled.pair_count == 1
    assert diagnostics.by_symbol[0].symbol == "BTCUSDT"
    assert diagnostics.by_symbol[0].metrics.pair_count == 1
    assert diagnostics.by_symbol[1].symbol == "ETHUSDT"
    assert diagnostics.by_symbol[1].metrics.pair_count == 0
    target = float(np.log(dataset.close[4, 0] / dataset.close[2, 0]))
    forecast = model.predict(dataset.features[2, 0])
    assert diagnostics.pooled.mae == pytest.approx(abs(forecast - target))
    assert diagnostics.pooled.zero_mae == pytest.approx(abs(target))


def test_forecast_diagnostics_excludes_target_close_late_at_scoring_as_of() -> None:
    api = _comparison_api()
    base = _dataset()
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    available_at = np.repeat(timestamps[:, None], 2, axis=1)
    available_at[2, 0] = timestamps[2] + np.timedelta64(1, "ns")
    available_at[4, 1] = timestamps[5] + np.timedelta64(1, "ns")
    information_available = np.ones((6, 2), dtype=np.bool_)
    information_available[2, 0] = False
    information_available[4, 1] = False
    dataset = replace(
        base,
        timestamps=timestamps,
        available_at=available_at,
        information_available=information_available,
        identity_payload_json=None,
    ).with_content_identity({"fixture": "late-target-close-at-asof"})
    model = replace(_model(dataset), horizon_hours=1)

    diagnostics = api._ridge_forecast_diagnostics(
        dataset,
        model,
        _config(api, include_forecast_diagnostics=True),
    )

    assert diagnostics.candidate_count == 4
    assert diagnostics.eligible_count == 2
    assert diagnostics.pooled.pair_count == 2
    assert dict(diagnostics.excluded_counts) == {
        "feature_unavailable_or_nonfinite": 0,
        "starting_close_unavailable_at_decision": 1,
        "target_not_mature_by_scoring_as_of": 0,
        "target_close_unavailable_by_scoring_as_of": 1,
    }
    assert diagnostics.by_symbol[0].eligible_count == 1
    assert diagnostics.by_symbol[1].eligible_count == 1


def test_forecast_diagnostics_leave_zero_baseline_skill_undefined(
    tmp_path: Path,
) -> None:
    base = _dataset()
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    close = base.close.copy()
    close[4] = close[2]
    dataset = replace(
        base,
        timestamps=timestamps,
        available_at=np.repeat(timestamps[:, None], 2, axis=1),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        identity_payload_json=None,
    ).with_content_identity({"fixture": "zero-baseline-skill"})
    api, dataset_root, ridge_root, ridge_digest, binding, _ = (
        _published_comparison_inputs(tmp_path, dataset)
    )

    result = api.compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=_config(api, include_forecast_diagnostics=True),
    )

    diagnostics = result.forecast_diagnostics
    assert diagnostics is not None
    assert diagnostics.pooled.pair_count == 2
    assert diagnostics.pooled.zero_rmse == 0.0
    assert diagnostics.pooled.mse_skill is None


def test_forecast_diagnostics_reports_empty_mature_pair_scope(
    tmp_path: Path,
) -> None:
    base = _dataset()
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    dataset = replace(
        base,
        timestamps=timestamps,
        available_at=np.repeat(timestamps[:, None], 2, axis=1),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "empty-forecast-diagnostics"})
    api, dataset_root, ridge_root, ridge_digest, binding, _ = (
        _published_comparison_inputs(tmp_path, dataset)
    )

    result = api.compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=_config(
            api,
            stop_index=4,
            include_forecast_diagnostics=True,
        ),
    )

    diagnostics = result.forecast_diagnostics
    assert diagnostics is not None
    assert diagnostics.candidate_count == 2
    assert diagnostics.eligible_count == 0
    assert diagnostics.pooled.pair_count == 0
    assert diagnostics.pooled.mae is None
    assert dict(diagnostics.excluded_counts)["target_not_mature_by_scoring_as_of"] == 2


def test_forecast_diagnostics_rejects_nonfinite_model_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _dataset()
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    dataset = replace(
        base,
        timestamps=timestamps,
        available_at=np.repeat(timestamps[:, None], 2, axis=1),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "nonfinite-forecast-diagnostics"})
    api, dataset_root, ridge_root, ridge_digest, binding, _ = (
        _published_comparison_inputs(tmp_path, dataset)
    )
    monkeypatch.setattr(
        RidgeForecastModel, "predict", lambda _self, _features: float("inf")
    )

    with pytest.raises(ValueError, match="Ridge forecast must be a finite number"):
        api.compare_ridge_artifact_shared_cash(
            dataset_root=dataset_root,
            ridge_artifact_root=ridge_root,
            expected_ridge_digest=ridge_digest,
            expected_binding=binding,
            config=_config(api, include_forecast_diagnostics=True),
        )


def test_forecast_diagnostics_flag_requires_a_builtin_boolean() -> None:
    api = _comparison_api()
    with pytest.raises((TypeError, ValueError), match="boolean"):
        _config(api, include_forecast_diagnostics=1)
