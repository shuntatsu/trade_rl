from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
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


def two_symbol_market(*, late_symbol: int | None = None) -> MarketDataset:
    n = 5
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    close = np.full((n, 2), 100.0)
    available_at = np.broadcast_to(timestamps[:, None], close.shape).copy()
    if late_symbol is not None:
        available_at[1, late_symbol] = timestamps[2]
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTC", "ETH"),
        timestamps=timestamps,
        features=np.zeros((n, 2, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        available_at=available_at,
    )


def late_single_symbol_market() -> MarketDataset:
    dataset = market()
    available_at = np.broadcast_to(
        dataset.timestamps[:, None], dataset.close.shape
    ).copy()
    available_at[1, 0] = dataset.timestamps[2]
    return market(available_at=available_at)


@pytest.mark.parametrize("mark", [100.0, 125.0])
@pytest.mark.parametrize("weight", [-0.5, 0.5])
def test_initial_cash_book_without_prices_sizes_at_current_market_mark(
    mark: float, weight: float
) -> None:
    dataset = market(mark_price=np.full((5, 1), mark))
    book = BookState.zero(1, 1000.0)
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        book, np.array([weight]), start_index=0, bars=1
    )

    # Omitted initial marks do not determine the price of the first entry.
    # Value 500 at the decision mark, then fill those signed units at 100.
    units = weight * 1000.0 / mark
    assert result.book.quantities == pytest.approx([units])
    assert result.book.cash == pytest.approx(1000.0 - units * 100.0)
    assert result.book.portfolio_value == pytest.approx(1000.0 + units * (mark - 100.0))
    assert result.book.fill_count == 1
    assert book.quantities == pytest.approx([0.0])
    assert book.mark_prices == pytest.approx([1.0])


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


def test_causal_capacity_uses_previous_completed_bar_volume() -> None:
    config = replace(
        ExecutionCostConfig.zero(),
        processing_bar_volume_capacity=False,
        max_participation_rate=1.0,
    )

    first_volume = np.array([[1.0], [1_000.0], [1_000.0], [1_000.0], [1_000.0]])
    first = market(volume=first_volume)
    first_result = MarketExecutor(first, config).execute_interval(
        BookState.zero(1, 1_000.0, first.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert first_result.filled_turnover == pytest.approx(0.1)
    assert first_result.unfilled_turnover == pytest.approx(0.9)

    current_volume = np.array([[1.0], [10_000.0], [1_000.0], [1_000.0], [1_000.0]])
    different_current_volume = market(volume=current_volume)
    current_result = MarketExecutor(different_current_volume, config).execute_interval(
        BookState.zero(1, 1_000.0, different_current_volume.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert current_result.filled_turnover == pytest.approx(first_result.filled_turnover)

    previous_volume = np.array([[2.0], [1_000.0], [1_000.0], [1_000.0], [1_000.0]])
    different_previous_volume = market(volume=previous_volume)
    previous_result = MarketExecutor(
        different_previous_volume, config
    ).execute_interval(
        BookState.zero(1, 1_000.0, different_previous_volume.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert previous_result.filled_turnover == pytest.approx(0.2)


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


def test_generated_limit_is_snapped_to_submit_tick_grid() -> None:
    shape = (5, 1)
    dataset = market(
        low=np.full(shape, 99.0),
        tick_size=np.full(shape, 0.5),
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            max_participation_rate=1.0,
            order_type="limit",
            limit_offset_rate=0.007,
        ),
    )

    result = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([0.5]),
        start_index=0,
        bars=1,
    )

    assert result.filled_turnover == pytest.approx(0.495)
    terminal = executor.compatibility_order_book.terminal_orders
    assert len(terminal) == 1
    assert terminal[0].intent.limit_price == pytest.approx(99.0)


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


def test_late_decision_mark_rejects_cash_entry_before_mutating_state() -> None:
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(5) * np.timedelta64(
        1, "h"
    )
    close = np.full((5, 1), 100.0)
    close[1, 0] = 200.0
    open_price = close.copy()
    open_price[2, 0] = 100.0
    high = np.maximum(open_price, close) + 1.0
    low = np.minimum(open_price, close) - 1.0
    available_at = np.broadcast_to(timestamps[:, None], close.shape).copy()
    available_at[1, 0] = timestamps[2]
    mark_price = close.copy()
    mark_price[2, 0] = 100.0
    dataset = market(
        timestamps=timestamps,
        close=close,
        open=open_price,
        high=high,
        low=low,
        mark_price=mark_price,
        available_at=available_at,
    )
    assert not dataset.information_available[1, 0]
    assert dataset.available_at[1, 0] > dataset.timestamps[1]

    book = BookState.zero(1, 1_000.0)
    initial_cash = book.cash
    initial_quantities = book.quantities.copy()
    initial_marks = book.mark_prices.copy()
    initial_exact_quantities = book.exact_quantities
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    initial_orders = executor.compatibility_order_book

    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        executor.execute_interval(
            book,
            np.array([0.5]),
            start_index=1,
            bars=1,
        )

    assert book.cash == initial_cash
    np.testing.assert_array_equal(book.quantities, initial_quantities)
    np.testing.assert_array_equal(book.mark_prices, initial_marks)
    assert book.exact_quantities == initial_exact_quantities
    current_orders = executor.compatibility_order_book
    assert current_orders.active_orders == initial_orders.active_orders
    assert current_orders.terminal_orders == initial_orders.terminal_orders


def test_late_held_symbol_is_checked_even_when_targeting_it_flat() -> None:
    dataset = two_symbol_market(late_symbol=1)
    book = BookState(
        quantities=np.array([0.0, 1.0]),
        cash=900.0,
        mark_prices=np.array([100.0, 100.0]),
        peak_value=1_000.0,
    )

    with pytest.raises(ValueError, match="decision-time mark unavailable.*ETH"):
        MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
            book,
            np.array([0.5, 0.0]),
            start_index=1,
            bars=1,
        )


def test_late_exact_dust_holding_is_checked_when_float_quantity_is_zero() -> None:
    dataset = late_single_symbol_market()
    dust = Fraction(1, 10**400)
    book = BookState(
        quantities=np.array([0.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0]),
        peak_value=1_000.0,
        _exact_quantities=(str(dust),),
    )
    assert book.quantities[0] == 0.0
    assert book.exact_quantities[0] != 0

    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
            book,
            np.array([0.0]),
            start_index=1,
            bars=1,
        )


def test_late_unselected_flat_symbol_does_not_block_selected_target() -> None:
    dataset = two_symbol_market(late_symbol=1)
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(2, 1_000.0),
        np.array([0.5, 0.0]),
        start_index=1,
        bars=1,
    )

    assert result.book.fill_count == 1


def test_on_time_price_clock_allows_explicitly_unavailable_information_flag() -> None:
    dataset = market(
        information_available=np.zeros((5, 1), dtype=np.bool_),
    )
    assert not dataset.information_available[1, 0]
    assert dataset.available_at[1, 0] == dataset.timestamps[1]

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 1_000.0),
        np.array([0.5]),
        start_index=1,
        bars=1,
    )

    assert result.book.fill_count == 1


def test_late_future_bar_remains_available_to_physical_interval_processing() -> None:
    dataset = late_single_symbol_market()

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 1_000.0),
        np.array([0.5]),
        start_index=0,
        bars=1,
    )

    assert result.book.fill_count == 1


def test_pending_only_flat_target_can_cancel_at_declared_late_row() -> None:
    dataset = late_single_symbol_market()
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), order_latency_bars=2),
    )
    pending = executor.execute_interval(
        BookState.zero(1, 1_000.0),
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert executor.compatibility_order_book.active_orders

    executor.execute_interval(
        pending.book,
        np.array([0.0]),
        start_index=1,
        bars=1,
    )

    assert not executor.compatibility_order_book.active_orders


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.array([0.5, 0.5]), "target, price, multiplier and holdings shapes"),
        (np.array([1.0 + 2e-12]), "target gross exceeds maximum_gross"),
        (np.array([np.nan]), "canonical JSON floats must be finite"),
    ],
)
def test_invalid_direct_target_keeps_its_existing_error_before_late_guard(
    target: np.ndarray, message: str
) -> None:
    dataset = late_single_symbol_market()

    with pytest.raises(ValueError, match=message):
        MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
            BookState.zero(1, 1_000.0),
            target,
            start_index=1,
            bars=1,
        )


def test_late_mark_guard_applies_inside_existing_gross_tolerance() -> None:
    dataset = late_single_symbol_market()

    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
            BookState.zero(1, 1_000.0),
            np.array([1.0 + 0.5e-12]),
            start_index=1,
            bars=1,
        )
