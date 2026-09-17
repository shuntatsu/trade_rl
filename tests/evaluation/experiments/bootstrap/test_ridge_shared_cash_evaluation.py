from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
    RidgeSharedCashArmEvidence,
    RidgeSharedCashEvaluation,
    canonical_ridge_shared_cash_evaluation_spec,
    evaluate_ridge_shared_cash,
    shared_cash_return_sha256,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.forecasts.ridge_economic_gate import RidgeEconomicGateStrategy

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_FEATURE_INDEX_TO_NAME = {
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


def _git_blob_sha(path: str) -> str:
    payload = Path(path).read_bytes()
    framed = f"blob {len(payload)}\0".encode() + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _feature_names() -> tuple[str, ...]:
    return tuple(
        _FEATURE_INDEX_TO_NAME.get(index, f"unused_{index}") for index in range(119)
    )


def _dataset() -> MarketDataset:
    timestamps = np.arange(
        np.datetime64("2022-12-31", "D"),
        np.datetime64("2025-01-02", "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    n_bars = len(timestamps)
    n_symbols = len(_SYMBOLS)
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        symbols=_SYMBOLS,
        timestamps=timestamps,
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
        periods_per_year=365,
    )


def _model() -> RidgeForecastModel:
    indices = tuple(_FEATURE_INDEX_TO_NAME)
    width = len(indices)
    return RidgeForecastModel(
        feature_indices=indices,
        feature_mean=np.zeros(width, dtype=np.float64),
        feature_scale=np.ones(width, dtype=np.float64),
        coefficients=np.full(width, 0.001, dtype=np.float64),
        intercept=0.0,
        horizon_hours=24,
        alpha=1.0,
        n_samples=100,
        fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
    )


def _replay(
    values: tuple[float, ...],
    *,
    total_cost: float,
    turnover: float,
    termination_reasons: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        returns=ReturnSeries(
            values=values,
            kind=ReturnKind.BASE_BAR,
            periods_per_year=365,
        ),
        diagnostics=SimpleNamespace(
            total_cost=total_cost,
            turnover_total=turnover,
            funding_pnl=0.0,
            borrow_cost=0.0,
            n_trades=1,
            rebalance_events=1,
            termination_reasons=termination_reasons,
        ),
    )


def _arm(
    name: str,
    values: tuple[float, ...],
    *,
    total_cost: float,
    turnover: float,
    max_drawdown: float,
    year_returns: tuple[tuple[int, float], ...],
    termination_reasons: tuple[str, ...] = (),
) -> RidgeSharedCashArmEvidence:
    total = float(np.prod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0)
    return RidgeSharedCashArmEvidence(
        arm=name,
        returns=values,
        return_sha256=shared_cash_return_sha256(values),
        total_return=total,
        calendar_year_returns=year_returns,
        total_cost=total_cost,
        turnover_total=turnover,
        max_drawdown=max_drawdown,
        termination_count=len(termination_reasons),
        termination_reasons=termination_reasons,
        n_periods=len(values),
    )


def test_carrier_sources_match_sealed_runtime_authorities() -> None:
    assert _git_blob_sha("trade_rl/strategies/forecasts/ridge_economic_gate.py") == (
        "db4fac918a421ce5223fa77257adcc234dadee66"
    )
    assert _git_blob_sha("trade_rl/evaluation/replay.py") == (
        "73c15bc8555bb9d1cbf124bf13df9d8ec6c1f15e"
    )
    assert _git_blob_sha("trade_rl/evaluation/runs/execute.py") == (
        "24dc24970502a598c6c74c75efe0a348f11cb51d"
    )
    assert _git_blob_sha("trade_rl/evaluation/runs/__init__.py") == (
        "c64142b8e7b2dce82f00f70dc883ceced35123a2"
    )
    assert (
        _git_blob_sha(
            "trade_rl/evaluation/experiments/bootstrap/ridge_shared_cash_prereg.py"
        )
        == "e848a0f4eec434f534455f88244eda75d97f4d04"
    )


def test_canonical_spec_binds_protocol_and_execution_authorities() -> None:
    spec = canonical_ridge_shared_cash_evaluation_spec()

    assert spec.issue_number == 630
    assert spec.protocol_head == "6615e30773cd3035a3f2e67e608aef2afc4e143c"
    assert spec.protocol_digest == (
        "14aa47651bc075c067c09403cf69e43f45d4151e9eb9935b6ccef61c5e7b9c5a"
    )
    assert spec.protocol_module_blob == "e848a0f4eec434f534455f88244eda75d97f4d04"
    assert spec.protocol_seal_run_id == 35209118815
    assert spec.protocol_primary_artifact_id == 10490873666
    assert spec.protocol_primary_artifact_api_digest == (
        "9d77d3593a406485db981ec6409dbe6ea7d2473300bbe78948997bfc23d572c0"
    )
    assert spec.protocol_fresh_artifact_id == 10491447310
    assert spec.protocol_fresh_artifact_api_digest == (
        "e2ff68438ad4bdecc0518527204f83a3c70219cf7912e59d6b80506f87a5475e"
    )
    assert spec.protocol_seal_sha256 == (
        "a0cf4db80d4aa4c25e0ee0409d96e9674d0dac5109a25dde687907da3cb74a27"
    )
    assert spec.ridge_implementation_head == (
        "222a082ee28f4f0fd35081912a33649cce27c585"
    )
    assert spec.shared_cash_main_head == "d18434799651cfc6c07e0840c40600dcf1dfa763"
    assert spec.execution_overlay == (
        "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
    )
    assert spec.symbols == _SYMBOLS
    assert spec.feature_indices == tuple(_FEATURE_INDEX_TO_NAME)
    assert spec.gross_budget == 0.5
    assert spec.initial_capital == 100_000.0
    assert spec.one_way_explicit_cost == 0.0007
    assert spec.unused_data_accessed is False
    assert spec.final_test_accessed is False
    assert spec.production_eligible is False
    assert spec.live_trading_authorized is False
    assert spec.merge_authorized is False

    with pytest.raises(ValueError, match="frozen"):
        replace(spec, gross_budget=0.6)


def test_evaluator_fits_once_and_replays_two_shared_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation as module

    dataset = _dataset()
    model = _model()
    fit_calls: list[dict[str, object]] = []
    replay_calls: list[dict[str, object]] = []

    def fit(dataset_arg: MarketDataset, **kwargs: object) -> RidgeForecastModel:
        assert dataset_arg is dataset
        fit_calls.append(dict(kwargs))
        return model

    baseline_values = np.zeros(731, dtype=np.float64)
    baseline_values[0] = 0.01
    baseline_values[1] = 0.02
    baseline_values[365] = 0.01
    candidate_values = np.zeros(731, dtype=np.float64)
    candidate_values[0] = 0.02
    candidate_values[1] = 0.03
    candidate_values[365] = 0.02
    replay_results = iter(
        (
            _replay(
                tuple(float(value) for value in baseline_values),
                total_cost=10.0,
                turnover=4.0,
            ),
            _replay(
                tuple(float(value) for value in candidate_values),
                total_cost=5.0,
                turnover=2.0,
            ),
        )
    )

    def replay(
        dataset_arg: MarketDataset,
        strategies: object,
        **kwargs: object,
    ) -> object:
        assert dataset_arg is dataset
        strategy_tuple = tuple(strategies)  # type: ignore[arg-type]
        assert len(strategy_tuple) == 5
        assert len({id(strategy) for strategy in strategy_tuple}) == 5
        replay_calls.append({"strategies": strategy_tuple, **kwargs})
        return next(replay_results)

    monkeypatch.setattr(module, "fit_ridge_forecast", fit)
    monkeypatch.setattr(module, "run_shared_cash_replay", replay)

    result = evaluate_ridge_shared_cash(dataset)

    assert len(fit_calls) == 1
    assert fit_calls[0] == {
        "feature_indices": tuple(_FEATURE_INDEX_TO_NAME),
        "fit_symbol_indices": (0, 1, 2, 3, 4),
        "fit_cutoff": np.datetime64("2023-01-01T00:00:00", "ns"),
        "horizon_hours": 24,
        "alpha": 1.0,
    }
    assert len(replay_calls) == 2
    baseline_strategies = replay_calls[0]["strategies"]
    candidate_strategies = replay_calls[1]["strategies"]
    assert all(isinstance(item, RidgeForecastStrategy) for item in baseline_strategies)
    assert all(
        isinstance(item, RidgeEconomicGateStrategy) for item in candidate_strategies
    )
    assert all(item.model is model for item in baseline_strategies)
    assert all(item.model is model for item in candidate_strategies)
    assert replay_calls[0]["execution_cost"] is replay_calls[1]["execution_cost"]
    assert (
        getattr(replay_calls[0]["execution_cost"], "processing_bar_volume_capacity")
        is False
    )
    for call in replay_calls:
        assert call["start_index"] == 1
        assert call["stop_index"] == 732
        assert call["gross_budget"] == 0.5
        assert call["initial_capital"] == 100_000.0
        assert call["risk"] is None

    assert result.status == "QUALIFY_UNUSED_VALIDATION"
    assert result.baseline.n_periods == 731
    assert result.candidate.n_periods == 731
    assert tuple(year for year, _ in result.baseline.calendar_year_returns) == (
        2023,
        2024,
    )
    assert tuple(
        value for _, value in result.baseline.calendar_year_returns
    ) == pytest.approx((0.0302, 0.01))
    assert tuple(year for year, _ in result.candidate.calendar_year_returns) == (
        2023,
        2024,
    )
    assert tuple(
        value for _, value in result.candidate.calendar_year_returns
    ) == pytest.approx((0.0506, 0.02))


def test_result_truth_table_and_fail_closed_invariants() -> None:
    baseline = _arm(
        "baseline",
        (0.01, 0.01, 0.01),
        total_cost=10.0,
        turnover=4.0,
        max_drawdown=0.20,
        year_returns=((2023, 0.0201), (2024, 0.01)),
    )
    candidate = _arm(
        "candidate",
        (0.02, 0.02, 0.02),
        total_cost=5.0,
        turnover=2.0,
        max_drawdown=0.10,
        year_returns=((2023, 0.0404), (2024, 0.02)),
    )
    spec = canonical_ridge_shared_cash_evaluation_spec()
    result = RidgeSharedCashEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        baseline=baseline,
        candidate=candidate,
        status="QUALIFY_UNUSED_VALIDATION",
    )
    assert result.status == "QUALIFY_UNUSED_VALIDATION"

    with pytest.raises(ValueError, match="return_sha256"):
        replace(candidate, return_sha256="0" * 64)
    with pytest.raises(ValueError, match="total_return"):
        replace(candidate, total_return=999.0)
    with pytest.raises(ValueError, match="status"):
        replace(result, status="STOP_BEFORE_UNUSED_VALIDATION")
    with pytest.raises(ValueError, match="period"):
        replace(result, candidate=replace(candidate, returns=(0.02, 0.02)))
    with pytest.raises(ValueError, match="production"):
        replace(result, production_eligible=True)


def test_new_candidate_termination_forces_stop() -> None:
    spec = canonical_ridge_shared_cash_evaluation_spec()
    baseline = _arm(
        "baseline",
        (0.01, 0.01),
        total_cost=10.0,
        turnover=4.0,
        max_drawdown=0.20,
        year_returns=((2023, 0.01), (2024, 0.01)),
    )
    candidate = _arm(
        "candidate",
        (0.02, 0.02),
        total_cost=5.0,
        turnover=2.0,
        max_drawdown=0.10,
        year_returns=((2023, 0.02), (2024, 0.02)),
        termination_reasons=("margin_exhausted",),
    )
    result = RidgeSharedCashEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        baseline=baseline,
        candidate=candidate,
        status="STOP_BEFORE_UNUSED_VALIDATION",
    )
    assert result.status == "STOP_BEFORE_UNUSED_VALIDATION"
