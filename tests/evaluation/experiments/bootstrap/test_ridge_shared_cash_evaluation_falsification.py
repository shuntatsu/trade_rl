from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
    RidgeSharedCashArmEvidence,
    evaluate_ridge_shared_cash,
    shared_cash_return_sha256,
    shared_cash_status,
)

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


def _arm(
    name: str,
    values: tuple[float, ...],
    *,
    total_cost: float = 5.0,
    turnover: float = 2.0,
    max_drawdown: float = 0.10,
    year_returns: tuple[tuple[int, float], ...] = (
        (2023, 0.02),
        (2024, 0.02),
    ),
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


def _baseline() -> RidgeSharedCashArmEvidence:
    return _arm(
        "baseline",
        (0.01, 0.01),
        total_cost=10.0,
        turnover=4.0,
        max_drawdown=0.20,
        year_returns=((2023, 0.01), (2024, 0.01)),
    )


def _candidate() -> RidgeSharedCashArmEvidence:
    return _arm("candidate", (0.02, 0.02))


@pytest.mark.parametrize(
    "gate",
    (
        "candidate_positive_total",
        "candidate_beats_baseline_total",
        "candidate_2023_positive_and_better",
        "candidate_2024_positive_and_better",
        "cost_lower",
        "turnover_lower",
        "drawdown_nonworse",
        "equal_period_count",
    ),
)
def test_each_frozen_shared_cash_gate_can_force_stop(gate: str) -> None:
    baseline = _baseline()
    candidate = _candidate()

    if gate == "candidate_positive_total":
        candidate = _arm("candidate", (-0.01, -0.01))
    elif gate == "candidate_beats_baseline_total":
        candidate = _arm("candidate", (0.005, 0.005))
    elif gate == "candidate_2023_positive_and_better":
        candidate = replace(
            candidate,
            calendar_year_returns=((2023, 0.0), (2024, 0.02)),
        )
    elif gate == "candidate_2024_positive_and_better":
        candidate = replace(
            candidate,
            calendar_year_returns=((2023, 0.02), (2024, 0.01)),
        )
    elif gate == "cost_lower":
        candidate = replace(candidate, total_cost=10.0)
    elif gate == "turnover_lower":
        candidate = replace(candidate, turnover_total=4.0)
    elif gate == "drawdown_nonworse":
        candidate = replace(candidate, max_drawdown=0.21)
    elif gate == "equal_period_count":
        candidate = _arm("candidate", (0.02,))
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(gate)

    assert shared_cash_status(baseline, candidate) == "STOP_BEFORE_UNUSED_VALIDATION"


def test_new_termination_multiset_count_forces_stop() -> None:
    baseline = _arm(
        "baseline",
        (0.01, 0.01),
        total_cost=10.0,
        turnover=4.0,
        max_drawdown=0.20,
        year_returns=((2023, 0.01), (2024, 0.01)),
        termination_reasons=("margin_exhausted",),
    )
    candidate = _arm(
        "candidate",
        (0.02, 0.02),
        termination_reasons=("margin_exhausted", "margin_exhausted"),
    )
    assert shared_cash_status(baseline, candidate) == "STOP_BEFORE_UNUSED_VALIDATION"


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
        close=close.copy(),
        volume=np.full((n_bars, n_symbols), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 119), dtype=np.bool_),
        feature_names=_feature_names(),
        global_feature_names=("regime",),
        periods_per_year=365,
    )


@pytest.mark.parametrize("drift", ("dataset", "feature"))
def test_authority_drift_fails_before_fit_or_replay(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation as module

    source = _dataset()
    if drift == "dataset":
        dataset = replace(source, dataset_id="f" * 64)
        message = "Dataset identity"
    else:
        names = list(source.feature_names)
        names[0] = "tampered_feature"
        dataset = replace(source, feature_names=tuple(names))
        message = "feature identity"

    calls = {"fit": 0, "replay": 0}

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        calls["fit"] += 1
        raise AssertionError("fit must not run after authority drift")

    def forbidden_replay(*args: object, **kwargs: object) -> object:
        calls["replay"] += 1
        raise AssertionError("replay must not run after authority drift")

    monkeypatch.setattr(module, "fit_ridge_forecast", forbidden_fit)
    monkeypatch.setattr(module, "run_shared_cash_replay", forbidden_replay)

    with pytest.raises(ValueError, match=message):
        evaluate_ridge_shared_cash(dataset)
    assert calls == {"fit": 0, "replay": 0}
