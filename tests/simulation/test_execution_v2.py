from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def market(
    *,
    open_price: np.ndarray | None = None,
    close: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
) -> MarketDataset:
    resolved_close = (
        np.asarray(close, dtype=np.float64)
        if close is not None
        else np.array([[100.0], [120.0], [120.0], [120.0]])
    )
    resolved_open = (
        np.asarray(open_price, dtype=np.float64)
        if open_price is not None
        else np.array([[100.0], [110.0], [120.0], [120.0]])
    )
    n_bars = len(resolved_close)
    high = np.maximum(resolved_open, resolved_close) + 1.0
    low = np.minimum(resolved_open, resolved_close) - 1.0
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=resolved_open,
        high=high,
        low=low,
        close=resolved_close,
        volume=(
            np.asarray(volume, dtype=np.float64)
            if volume is not None
            else np.full((n_bars, 1), 1_000_000.0)
        ),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=(
            np.asarray(tradable, dtype=np.bool_)
            if tradable is not None
            else np.ones((n_bars, 1), dtype=np.bool_)
        ),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_decision_after_close_executes_at_next_open() -> None:
    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, dataset.close[0])

    result = executor.execute_interval(
        book,
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.book.portfolio_value == pytest.approx(1_000.0 * 120.0 / 110.0)
    assert result.filled_turnover == pytest.approx(1.0)
    assert result.unfilled_turnover == pytest.approx(0.0)


def test_participation_cap_causes_partial_fill() -> None:
    dataset = market(
        open_price=np.full((4, 1), 100.0),
        close=np.full((4, 1), 100.0),
        volume=np.ones((4, 1), dtype=np.float64),
    )
    executor = MarketExecutor(
        dataset,
        ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.10,
        ),
    )
    book = BookState.zero(1, 1_000.0, dataset.close[0])

    result = executor.execute_interval(
        book,
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.requested_turnover == pytest.approx(1.0)
    assert result.filled_turnover == pytest.approx(0.01)
    assert result.unfilled_turnover == pytest.approx(0.99)
    assert result.book.weights[0] == pytest.approx(0.01)


def test_non_tradable_next_bar_does_not_fill() -> None:
    tradable = np.ones((4, 1), dtype=np.bool_)
    tradable[1, 0] = False
    dataset = market(tradable=tradable)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, dataset.close[0])

    result = executor.execute_interval(
        book,
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.filled_turnover == pytest.approx(0.0)
    assert result.unfilled_turnover == pytest.approx(1.0)
    assert result.book.fill_count == 0
    assert result.book.rebalance_events == 0


def two_symbol_flat_market(close: np.ndarray, open_price: np.ndarray) -> MarketDataset:
    n_bars = len(close)
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_multi_symbol_rebalance_counts_fills_not_only_decisions() -> None:
    close = np.full((4, 2), 100.0)
    dataset = two_symbol_flat_market(close, close.copy())
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(2, 1_000.0, dataset.close[0]),
        np.array([0.5, -0.5]),
        start_index=0,
        bars=1,
    )

    assert result.book.rebalance_events == 1
    assert result.book.fill_count == 2


def test_fully_filled_decision_holds_quantities_until_next_decision() -> None:
    close = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
            [121.0, 100.0],
            [121.0, 100.0],
        ]
    )
    open_price = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [110.0, 100.0],
            [121.0, 100.0],
        ]
    )
    dataset = two_symbol_flat_market(close, open_price)

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(2, 1_000.0, dataset.close[0]),
        np.array([0.5, 0.5]),
        start_index=0,
        bars=2,
    )

    assert result.book.rebalance_events == 1
    assert result.book.fill_count == 2
    np.testing.assert_allclose(result.book.quantities, np.array([5.0, 5.0]))
    assert result.book.weights[0] > 0.5
    assert result.book.weights[1] < 0.5
