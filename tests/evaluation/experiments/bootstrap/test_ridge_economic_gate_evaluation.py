from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateEvaluation,
    RidgeEconomicGateSymbolResult,
    canonical_ridge_economic_gate_evaluation_spec,
    evaluate_ridge_economic_gate,
    research_status_from_counts,
    ridge_model_sha256,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.forecasts.ridge_economic_gate import RidgeEconomicGateStrategy

_FEATURE_NAMES = (
    "1h__log_return_1bar",
    "1h__log_return_4bar",
    "1h__log_return_24bar",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
    "1h__rsi_14bar",
    "1h__macd_histogram_12_26_9",
    "4h__log_return_4bar",
    "4h__realized_volatility_24bar",
    "1d__log_return_1bar",
    "1d__realized_volatility_24bar",
)
_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def _feature_roster() -> tuple[str, ...]:
    names = [f"unused_{index}" for index in range(119)]
    for index, name in zip(_FEATURE_INDICES, _FEATURE_NAMES, strict=True):
        names[index] = name
    return tuple(names)


def _dataset(*, fee_rate: float = 0.0005) -> MarketDataset:
    timestamps = np.arange(
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2025-01-03", "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    n_bars = timestamps.size
    n_symbols = len(_SYMBOLS)
    n_features = 119
    close = np.full((n_bars, n_symbols), 100.0)
    return MarketDataset(
        dataset_id="6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        symbols=_SYMBOLS,
        timestamps=timestamps,
        features=np.zeros((n_bars, n_symbols, n_features), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, n_features), dtype=np.bool_),
        feature_names=_feature_roster(),
        global_feature_names=("regime",),
        fee_rate=np.full((n_bars, n_symbols), fee_rate),
        taker_fee_rate=np.zeros((n_bars, n_symbols)),
        spread_rate=np.full((n_bars, n_symbols), 0.0002),
        max_participation_rate=np.broadcast_to(
            np.asarray(
                [
                    0.0021629560553901974,
                    0.002044685341258238,
                    0.002184898995567895,
                    0.0020480213652913385,
                    0.002346308308284808,
                ]
            ),
            (n_bars, n_symbols),
        ).copy(),
        periods_per_year=365,
    )


def _model(*, coefficient_shift: float = 0.0) -> RidgeForecastModel:
    n = len(_FEATURE_INDICES)
    return RidgeForecastModel(
        feature_indices=_FEATURE_INDICES,
        feature_mean=np.zeros(n),
        feature_scale=np.ones(n),
        coefficients=np.arange(1, n + 1, dtype=np.float64) + coefficient_shift,
        intercept=0.001,
        horizon_hours=24,
        alpha=1.0,
        n_samples=12345,
        fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
    )


def _entry(name: str, index: int) -> SimpleNamespace:
    baseline = name == "baseline"
    total_return = 0.10 + index * 0.01 if baseline else 0.11 + index * 0.01
    total_cost = 10.0 + index if baseline else 8.0 + index
    turnover = 100.0 + index if baseline else 80.0 + index
    drawdown = 0.20 + index * 0.001 if baseline else 0.19 + index * 0.001
    values = (0.01 + index * 0.001, -0.002)
    metrics = SimpleNamespace(
        total_return=total_return,
        total_cost=total_cost,
        turnover_total=turnover,
        max_drawdown=drawdown,
        termination_count=0,
        n_periods=len(values),
    )
    replay = SimpleNamespace(
        diagnostics=SimpleNamespace(termination_reasons=()),
        returns=SimpleNamespace(values=values),
    )
    return SimpleNamespace(name=name, metrics=metrics, replay=replay)


def _comparison() -> SimpleNamespace:
    by_symbol = []
    for index, symbol in enumerate(_SYMBOLS):
        by_symbol.append(
            SimpleNamespace(
                symbol=symbol,
                comparison=SimpleNamespace(
                    entries=(_entry("baseline", index), _entry("candidate", index))
                ),
            )
        )
    return SimpleNamespace(by_symbol=tuple(by_symbol))


def test_spec_binds_sealed_preregistration_and_exact_ridge_configuration() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()

    assert spec.protocol_digest == (
        "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3"
    )
    assert spec.prereg_head == "75999e53c70224c31a62b106e4a8d2caa4b920ac"
    assert spec.prereg_seal_run_id == 35126253392
    assert spec.prereg_seal_artifact_id == 10459431804
    assert spec.prereg_fresh_artifact_id == 10458759061
    assert spec.dataset_id == _dataset().dataset_id
    assert spec.symbols == _SYMBOLS
    assert spec.feature_names == _FEATURE_NAMES
    assert spec.feature_indices == _FEATURE_INDICES
    assert spec.fit_symbol_indices == (0, 1, 2, 3, 4)
    assert spec.ridge_horizon_hours == 24
    assert spec.ridge_alpha == 1.0
    assert spec.forecast_entry_threshold == 0.0025
    assert spec.forecast_exit_threshold == 0.0005
    assert spec.one_way_explicit_cost == 0.0007
    assert spec.evaluation_pnl_inspected is False
    assert spec.evaluation_execution_authorized is False
    assert spec.production_eligible is False
    assert spec.final_test_authorized is False
    assert spec.merge_authorized is False


def test_model_fingerprint_is_deterministic_and_semantic() -> None:
    first = _model()
    same = _model()
    changed = _model(coefficient_shift=1.0)

    assert ridge_model_sha256(first) == ridge_model_sha256(same)
    assert ridge_model_sha256(first) != ridge_model_sha256(changed)


def test_research_status_replays_preregistered_rule() -> None:
    assert (
        research_status_from_counts(
            positive_effect_symbols=4,
            median_excess_total_return=0.01,
            cost_reduction_symbols=4,
            turnover_reduction_symbols=4,
            drawdown_nonworse_symbols=4,
            new_termination_symbols=0,
        )
        == "PROMOTE_RESEARCH_REFERENCE"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=2,
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=0,
        )
        == "REJECT_MECHANISM"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=3,
            median_excess_total_return=0.01,
            cost_reduction_symbols=3,
            turnover_reduction_symbols=3,
            drawdown_nonworse_symbols=3,
            new_termination_symbols=0,
        )
        == "INCONCLUSIVE"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=5,
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=1,
        )
        == "REJECT_MECHANISM"
    )


def test_pairwise_evaluator_fits_once_and_shares_exact_model(monkeypatch) -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    dataset = _dataset()
    model = _model()
    calls: dict[str, object] = {}

    def fake_fit(dataset_arg, **kwargs):
        calls["fit_dataset"] = dataset_arg
        calls["fit_kwargs"] = kwargs
        return model

    def fake_compare(dataset_arg, strategies, **kwargs):
        calls["compare_dataset"] = dataset_arg
        calls["strategies"] = strategies
        calls["compare_kwargs"] = kwargs
        return _comparison()

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation.fit_ridge_forecast",
        fake_fit,
    )
    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation.compare_strategies_by_symbol",
        fake_compare,
    )

    result = evaluate_ridge_economic_gate(dataset, spec)

    assert calls["fit_dataset"] is dataset
    assert calls["fit_kwargs"] == {
        "feature_indices": _FEATURE_INDICES,
        "fit_symbol_indices": (0, 1, 2, 3, 4),
        "fit_cutoff": np.datetime64("2023-01-01T00:00:00", "ns"),
        "horizon_hours": 24,
        "alpha": 1.0,
    }
    strategies = calls["strategies"]
    assert isinstance(strategies["baseline"], RidgeForecastStrategy)
    assert isinstance(strategies["candidate"], RidgeEconomicGateStrategy)
    assert strategies["baseline"].model is model
    assert strategies["candidate"].config.model is model
    assert result.model_sha256 == ridge_model_sha256(model)
    assert result.positive_effect_symbols == 5
    assert result.cost_reduction_symbols == 5
    assert result.turnover_reduction_symbols == 5
    assert result.drawdown_nonworse_symbols == 5
    assert result.research_status == "PROMOTE_RESEARCH_REFERENCE"


def test_cost_drift_fails_before_fit_or_replay(monkeypatch) -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    called = False

    def forbidden_fit(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("fit must not execute after cost drift")

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation.fit_ridge_forecast",
        forbidden_fit,
    )

    with pytest.raises(ValueError, match="evaluation cost drift"):
        evaluate_ridge_economic_gate(_dataset(fee_rate=0.0006), spec)
    assert called is False


def _symbol_result() -> RidgeEconomicGateSymbolResult:
    return RidgeEconomicGateSymbolResult(
        symbol="BTCUSDT",
        baseline_total_return=0.10,
        candidate_total_return=0.11,
        excess_total_return=0.01,
        baseline_total_cost=10.0,
        candidate_total_cost=8.0,
        baseline_turnover_total=100.0,
        candidate_turnover_total=80.0,
        baseline_max_drawdown=0.20,
        candidate_max_drawdown=0.19,
        baseline_termination_count=0,
        candidate_termination_count=0,
        baseline_termination_reasons=(),
        candidate_termination_reasons=(),
        new_termination=False,
        baseline_n_periods=2,
        candidate_n_periods=2,
        baseline_return_sha256="a" * 64,
        candidate_return_sha256="b" * 64,
    )


def test_symbol_result_rejects_arithmetic_and_termination_forgery() -> None:
    with pytest.raises(ValueError, match="excess_total_return"):
        replace(_symbol_result(), excess_total_return=99.0)
    with pytest.raises(ValueError, match="new_termination"):
        replace(_symbol_result(), new_termination=True)


def test_result_rejects_aggregate_or_status_forgery() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    rows = tuple(replace(_symbol_result(), symbol=symbol) for symbol in _SYMBOLS)
    kwargs = dict(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        model_sha256="c" * 64,
        symbols=_SYMBOLS,
        by_symbol=rows,
        positive_effect_symbols=5,
        median_excess_total_return=0.01,
        cost_reduction_symbols=5,
        turnover_reduction_symbols=5,
        drawdown_nonworse_symbols=5,
        new_termination_symbols=0,
        candidate_positive_total_return_symbols=5,
        research_status="PROMOTE_RESEARCH_REFERENCE",
    )
    valid = RidgeEconomicGateEvaluation(**kwargs)
    assert valid.research_status == "PROMOTE_RESEARCH_REFERENCE"

    with pytest.raises(ValueError, match="spec_digest"):
        RidgeEconomicGateEvaluation(**{**kwargs, "spec_digest": "d" * 64})
    with pytest.raises(ValueError, match="dataset_id"):
        RidgeEconomicGateEvaluation(**{**kwargs, "dataset_id": "e" * 64})

    with pytest.raises(ValueError, match="positive_effect_symbols"):
        RidgeEconomicGateEvaluation(**{**kwargs, "positive_effect_symbols": 4})
    with pytest.raises(ValueError, match="research_status"):
        RidgeEconomicGateEvaluation(**{**kwargs, "research_status": "INCONCLUSIVE"})
