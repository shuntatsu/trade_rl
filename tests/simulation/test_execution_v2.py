from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig


def market(**overrides: object) -> MarketDataset:
    n = 5
    close = np.full((n, 1), 100.0)
    open_price = close.copy()
    values: dict[str, object] = {
        "dataset_id": "b" * 64,
        "symbols": ("BTC",),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        "features": np.zeros((n, 1, 1), dtype=np.float32),
        "global_features": np.zeros((n, 1), dtype=np.float32),
        "open": open_price,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full((n, 1), 1_000_000.0),
        "funding_rate": np.zeros_like(close),
        "tradable": np.ones_like(close, dtype=np.bool_),
        "feature_available": np.ones((n, 1, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }
    values.update(overrides)
    return MarketDataset(**values)


def test_next_open_execution_uses_processing_bar_volume() -> None:
    volume = np.array([[1.0], [1_000.0], [1_000.0], [1_000.0], [1_000.0]])
    dataset = market(volume=volume)
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert result.filled_turnover == pytest.approx(1.0)
    assert result.unfilled_turnover == pytest.approx(0.0)


def test_minimum_notional_lot_tick_and_borrow_constraints_are_enforced() -> None:
    shape = (5, 1)
    dataset = market(
        minimum_notional=np.full(shape, 200.0),
        lot_size=np.full(shape, 0.5),
        tick_size=np.full(shape, 0.1),
        borrow_available=np.zeros(shape, dtype=np.bool_),
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, dataset.close[0])
    small = executor.execute_interval(book, np.array([0.1]), start_index=0, bars=1)
    assert small.filled_turnover == 0.0
    short = executor.execute_interval(book, np.array([-1.0]), start_index=0, bars=1)
    assert short.filled_turnover == 0.0


def test_limit_order_fills_only_when_bar_touches_limit() -> None:
    dataset = market(low=np.full((5, 1), 99.9), high=np.full((5, 1), 100.1))
    executor = MarketExecutor(
        dataset,
        ExecutionCostConfig.zero().__class__(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            order_type="limit",
            limit_offset_rate=0.01,
        ),
    )
    result = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert result.filled_turnover == 0.0


def test_episode_random_streams_can_be_paired_but_changed_between_episodes() -> None:
    dataset = market()
    config = ExecutionCostConfig(
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=1.0,
        slippage_std=0.01,
    )
    first = MarketExecutor(dataset, config)
    second = MarketExecutor(dataset, config)
    first.reset_random_state(7)
    second.reset_random_state(7)
    a = first.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    b = second.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert a.interval_cost == pytest.approx(b.interval_cost)
    first.reset_random_state(8)
    c = first.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert c.interval_cost != pytest.approx(a.interval_cost)
