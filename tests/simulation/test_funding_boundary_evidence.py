from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState

_HOUR = np.timedelta64(1, "h")


def _dataset(*, second_boundary: bool = False) -> MarketDataset:
    n_bars = 5
    shape = (n_bars, 1)
    open_price = np.full(shape, 100.0, dtype=np.float64)
    close = open_price.copy()
    mark_price = open_price.copy()
    mark_price[1, 0] = 120.0
    mark_price[2, 0] = 110.0
    funding_rate = np.zeros(shape, dtype=np.float64)
    funding_rate[1, 0] = 0.001
    funding_rate[2, 0] = 0.002
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[1, 0] = True
    funding_due[2, 0] = second_boundary
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns") + np.arange(n_bars) * _HOUR,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, mark_price),
        low=np.minimum(open_price, mark_price),
        close=close,
        volume=np.full(shape, 1_000.0, dtype=np.float64),
        funding_rate=funding_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
        funding_due=funding_due,
        mark_price=mark_price,
        contract_multipliers=np.array([1.0], dtype=np.float64),
    )


def _executor(dataset: MarketDataset) -> MarketExecutor:
    return MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
    )


def _book(dataset: MarketDataset, *, weight: float = 1.0) -> BookState:
    return BookState.from_weights(
        weights=np.array([weight], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )


def test_stateful_result_preserves_funding_boundary_inputs_and_equity() -> None:
    dataset = _dataset()
    result = _executor(dataset).execute_orders(
        _book(dataset),
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    assert len(result.funding_evidence) == 1
    evidence = result.funding_evidence[0]
    assert evidence.processing_index == 1
    assert evidence.timestamp_ns == int(dataset.timestamps[1].astype(np.int64))
    assert evidence.funding_due == (True,)
    assert evidence.signed_quantities == pytest.approx((10.0,))
    assert evidence.mark_prices == pytest.approx((120.0,))
    assert evidence.contract_multipliers == pytest.approx((1.0,))
    assert evidence.funding_rates == pytest.approx((0.001,))
    assert evidence.funding_amount == pytest.approx(-1.2)
    assert evidence.equity_before_funding == pytest.approx(1_200.0)
    assert evidence.equity_after_funding == pytest.approx(1_198.8)
    assert result.interval_funding == pytest.approx(-1.2)


def test_zero_position_funding_boundary_is_still_preserved() -> None:
    dataset = _dataset()
    result = _executor(dataset).execute_orders(
        _book(dataset, weight=0.0),
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    assert len(result.funding_evidence) == 1
    evidence = result.funding_evidence[0]
    assert evidence.signed_quantities == pytest.approx((0.0,))
    assert evidence.funding_amount == pytest.approx(0.0)
    assert evidence.equity_before_funding == pytest.approx(1_000.0)
    assert evidence.equity_after_funding == pytest.approx(1_000.0)


def test_multi_bar_execution_keeps_funding_boundaries_separate() -> None:
    dataset = _dataset(second_boundary=True)
    result = _executor(dataset).execute_orders(
        _book(dataset),
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=2,
    )

    assert [item.processing_index for item in result.funding_evidence] == [1, 2]
    assert [item.timestamp_ns for item in result.funding_evidence] == [
        int(dataset.timestamps[1].astype(np.int64)),
        int(dataset.timestamps[2].astype(np.int64)),
    ]
    assert [item.funding_amount for item in result.funding_evidence] == pytest.approx(
        [-1.2, -2.2]
    )
    assert result.interval_funding == pytest.approx(-3.4)
