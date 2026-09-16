from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateCostAuthority,
    RidgeEconomicGateSymbolResult,
    canonical_ridge_economic_gate_evaluation_spec,
    evaluate_ridge_economic_gate,
)
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_REQUIRED_NAMES = {
    0: "1h__log_return_1bar",
    1: "1h__log_return_4bar",
    2: "1h__log_return_24bar",
    4: "1h__realized_volatility_24bar",
    5: "1h__volume_zscore_24bar",
    6: "1h__funding_bps",
    7: "1h__rsi_14bar",
    10: "1h__macd_histogram_12_26_9",
    60: "4h__log_return_4bar",
    63: "4h__realized_volatility_24bar",
    114: "1d__log_return_1bar",
    118: "1d__realized_volatility_24bar",
}


def _feature_names() -> tuple[str, ...]:
    names = [f"unused_{index}" for index in range(119)]
    for index, name in _REQUIRED_NAMES.items():
        names[index] = name
    return tuple(names)


def _dataset() -> MarketDataset:
    n_bars = 3
    n_symbols = len(_SYMBOLS)
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        symbols=_SYMBOLS,
        timestamps=np.asarray(
            [
                "2023-01-01T00:00:00",
                "2025-01-01T00:00:00",
                "2025-01-01T01:00:00",
            ],
            dtype="datetime64[ns]",
        ),
        features=np.zeros((n_bars, n_symbols, 119), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 119), dtype=np.bool_),
        feature_names=_feature_names(),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        calendar_kind="session_calendar",
        fee_rate=np.full((n_bars, n_symbols), 0.0005, dtype=np.float64),
        taker_fee_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        spread_rate=np.full((n_bars, n_symbols), 0.0002, dtype=np.float64),
    )


def _cost_authority() -> RidgeEconomicGateCostAuthority:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    return RidgeEconomicGateCostAuthority(
        source_run_id=1,
        source_artifact_id=2,
        source_artifact_api_digest="a" * 64,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        fee_rate=spec.fee_rate,
        taker_fee_rate=spec.taker_fee_rate,
        spread_rate=spec.spread_rate,
        one_way_explicit_cost=spec.one_way_explicit_cost,
        verified=True,
    )


def _model(*, horizon_hours: int = 24) -> RidgeForecastModel:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    width = len(spec.feature_indices)
    return RidgeForecastModel(
        feature_indices=spec.feature_indices,
        feature_mean=np.zeros(width, dtype=np.float64),
        feature_scale=np.ones(width, dtype=np.float64),
        coefficients=np.full(width, 0.001, dtype=np.float64),
        intercept=0.0,
        horizon_hours=horizon_hours,
        alpha=1.0,
        n_samples=100,
        fit_cutoff=np.datetime64(spec.fit_cutoff, "ns"),
    )


def _entry(
    name: str,
    *,
    returns: tuple[float, ...] = (0.0, 0.0),
    total_return: float | None = None,
    n_periods: int | None = None,
    termination_count: int = 0,
    termination_reasons: tuple[str, ...] = (),
) -> Any:
    resolved_total = (
        float(np.prod(1.0 + np.asarray(returns, dtype=np.float64)) - 1.0)
        if total_return is None
        else total_return
    )
    return SimpleNamespace(
        name=name,
        metrics=SimpleNamespace(
            total_return=resolved_total,
            total_cost=1.0,
            turnover_total=1.0,
            max_drawdown=0.1,
            termination_count=termination_count,
            n_periods=len(returns) if n_periods is None else n_periods,
        ),
        replay=SimpleNamespace(
            returns=SimpleNamespace(values=returns),
            diagnostics=SimpleNamespace(termination_reasons=termination_reasons),
        ),
    )


def _comparison(
    *,
    baseline: Any | None = None,
    candidate: Any | None = None,
    omit_candidate: bool = False,
    reverse_symbols: bool = False,
) -> Any:
    baseline_entry = baseline or _entry("baseline")
    candidate_entry = candidate or _entry("candidate", returns=(0.01, 0.0))
    entries = (baseline_entry,) if omit_candidate else (baseline_entry, candidate_entry)
    symbols = tuple(reversed(_SYMBOLS)) if reverse_symbols else _SYMBOLS
    return SimpleNamespace(
        by_symbol=tuple(
            SimpleNamespace(
                symbol_index=index,
                symbol=symbol,
                comparison=SimpleNamespace(entries=entries),
            )
            for index, symbol in enumerate(symbols)
        )
    )


def _patch_economic_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: RidgeForecastModel | None = None,
    comparison: Any | None = None,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation as module

    monkeypatch.setattr(
        module,
        "fit_ridge_forecast",
        lambda *args, **kwargs: model or _model(),
    )
    monkeypatch.setattr(
        module,
        "compare_strategies_by_symbol",
        lambda *args, **kwargs: comparison or _comparison(),
    )


def test_dataset_explicit_cost_drift_fails_before_economic_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    dataset.fee_rate[1, 2] = 0.0006
    _patch_economic_boundaries(monkeypatch)

    with pytest.raises(ValueError, match="evaluation cost drift: fee_rate"):
        evaluate_ridge_economic_gate(
            dataset,
            canonical_ridge_economic_gate_evaluation_spec(),
            _cost_authority(),
        )


def test_dataset_feature_identity_and_clock_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_economic_boundaries(monkeypatch)
    spec = canonical_ridge_economic_gate_evaluation_spec()

    feature_drift = _dataset()
    names = list(feature_drift.feature_names)
    names[2] = "wrong_signal"
    object.__setattr__(feature_drift, "feature_names", tuple(names))
    with pytest.raises(ValueError, match="feature identity"):
        evaluate_ridge_economic_gate(feature_drift, spec, _cost_authority())

    clock_drift = _dataset()
    timestamps = np.asarray(clock_drift.timestamps).copy()
    timestamps[1] = np.datetime64("2024-12-31T23:00:00", "ns")
    object.__setattr__(clock_drift, "timestamps", timestamps)
    with pytest.raises(ValueError, match="evaluation_stop_exclusive"):
        evaluate_ridge_economic_gate(clock_drift, spec, _cost_authority())


def test_fitted_ridge_identity_drift_fails_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation as module

    monkeypatch.setattr(module, "fit_ridge_forecast", lambda *args, **kwargs: _model(horizon_hours=12))
    called = False

    def compare(*args: object, **kwargs: object) -> Any:
        nonlocal called
        called = True
        return _comparison()

    monkeypatch.setattr(module, "compare_strategies_by_symbol", compare)
    with pytest.raises(RuntimeError, match="model identity drifted"):
        evaluate_ridge_economic_gate(
            _dataset(),
            canonical_ridge_economic_gate_evaluation_spec(),
            _cost_authority(),
        )
    assert called is False


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        (_comparison(omit_candidate=True), "strategy roster drifted"),
        (_comparison(reverse_symbols=True), "output symbol roster drifted"),
        (
            _comparison(baseline=_entry("baseline", total_return=0.123)),
            "baseline total return differs from raw return path",
        ),
        (
            _comparison(candidate=_entry("candidate", n_periods=3)),
            "candidate return count differs from metrics",
        ),
        (
            _comparison(
                candidate=_entry(
                    "candidate",
                    termination_count=1,
                    termination_reasons=(),
                )
            ),
            "candidate termination evidence is inconsistent",
        ),
    ],
)
def test_pairwise_evidence_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    comparison: Any,
    message: str,
) -> None:
    _patch_economic_boundaries(monkeypatch, comparison=comparison)
    with pytest.raises(RuntimeError, match=message):
        evaluate_ridge_economic_gate(
            _dataset(),
            canonical_ridge_economic_gate_evaluation_spec(),
            _cost_authority(),
        )


def test_termination_multiset_detects_new_reason_without_count_increase() -> None:
    common = dict(
        symbol="BTCUSDT",
        baseline_total_return=0.0,
        candidate_total_return=0.0,
        excess_total_return=0.0,
        baseline_total_cost=1.0,
        candidate_total_cost=1.0,
        baseline_turnover_total=1.0,
        candidate_turnover_total=1.0,
        baseline_max_drawdown=0.1,
        candidate_max_drawdown=0.1,
        baseline_termination_count=1,
        candidate_termination_count=1,
        baseline_termination_reasons=("margin_call",),
        candidate_termination_reasons=("liquidation",),
        baseline_n_periods=2,
        candidate_n_periods=2,
        baseline_return_sha256="b" * 64,
        candidate_return_sha256="c" * 64,
    )
    RidgeEconomicGateSymbolResult(new_termination=True, **common)
    with pytest.raises(ValueError, match="new_termination"):
        RidgeEconomicGateSymbolResult(new_termination=False, **common)


def test_cost_authority_rejects_bool_ids_and_result_flags_remain_false() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    with pytest.raises(ValueError, match="source_run_id"):
        RidgeEconomicGateCostAuthority(
            source_run_id=True,  # type: ignore[arg-type]
            source_artifact_id=2,
            source_artifact_api_digest="a" * 64,
            dataset_id=spec.dataset_id,
            dataset_artifact_digest=spec.dataset_artifact_digest,
            evaluation_start=spec.evaluation_start,
            evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
            fee_rate=spec.fee_rate,
            taker_fee_rate=spec.taker_fee_rate,
            spread_rate=spec.spread_rate,
            one_way_explicit_cost=spec.one_way_explicit_cost,
            verified=True,
        )
