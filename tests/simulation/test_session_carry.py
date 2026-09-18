from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState


def test_borrow_and_cash_interest_use_actual_elapsed_time() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.full((3, 1), 100.0)
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        borrow_rate=np.full((3, 1), 0.365),
        cash_rate=np.full(3, 0.365),
    )
    book = BookState.from_weights(
        weights=np.array([-0.5]),
        capital=1_000.0,
        prices=prices[0],
        contract_multipliers=dataset.contract_multipliers,
    )
    result = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0),
    ).execute_interval(book, np.array([-0.5]), start_index=0, bars=1)

    year_fraction = 65.0 / (365.0 * 24.0)
    expected_borrow = 500.0 * 0.365 * year_fraction
    # Initial short creates 1,500 cash; interest is charged/credited over 65 hours.
    expected_interest = 1_500.0 * 0.365 * year_fraction
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.interval_cash_interest == pytest.approx(expected_interest)


def test_next_open_entry_excludes_prior_gap_carry() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.full((3, 1), 100.0)
    dataset = MarketDataset(
        dataset_id="b" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        borrow_rate=np.full((3, 1), 0.365),
        cash_rate=np.full(3, 0.365),
    )
    book = BookState.zero(
        1,
        1_000.0,
        prices[0],
        contract_multipliers=dataset.contract_multipliers,
    )

    result = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0),
    ).execute_interval(book, np.array([-0.5]), start_index=0, bars=1)

    gap_fraction = 64.0 / (365.0 * 24.0)
    processing_bar_fraction = 1.0 / (365.0 * 24.0)
    expected_borrow = 500.0 * 0.365 * processing_bar_fraction
    gap_interest = 1_000.0 * 0.365 * gap_fraction
    processing_interest = 1_500.0 * 0.365 * processing_bar_fraction
    expected_interest = gap_interest + processing_interest
    assert result.book.quantities[0] == pytest.approx(-5.0)
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.interval_cash_interest == pytest.approx(expected_interest)


def test_session_gap_borrow_uses_previous_close_before_next_open() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    open_price = np.array([[100.0], [120.0], [120.0]])
    close = open_price.copy()
    dataset = MarketDataset(
        dataset_id="c" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=open_price,
        high=open_price,
        low=open_price,
        close=close,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        borrow_rate=np.full((3, 1), 0.365),
        cash_rate=np.full(3, 0.365),
    )
    book = BookState.from_weights(
        weights=np.array([-0.5]),
        capital=1_000.0,
        prices=np.array([100.0]),
        contract_multipliers=dataset.contract_multipliers,
    )

    result = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0),
    ).execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    gap_fraction = 64.0 / (365.0 * 24.0)
    processing_fraction = 1.0 / (365.0 * 24.0)
    expected_borrow = (
        500.0 * 0.365 * gap_fraction + 600.0 * 0.365 * processing_fraction
    )
    expected_interest = 1_500.0 * 0.365 * (gap_fraction + processing_fraction)
    assert result.book.quantities[0] == pytest.approx(-5.0)
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.interval_cash_interest == pytest.approx(expected_interest)


def test_gap_margin_call_flattens_at_next_open_not_previous_close() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    open_price = np.array([[100.0], [120.0], [120.0]])
    dataset = MarketDataset(
        dataset_id="e" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=open_price,
        high=open_price,
        low=open_price,
        close=open_price,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        borrow_rate=np.full((3, 1), 219.0),
        cash_rate=np.zeros(3),
    )
    book = BookState.from_weights(
        weights=np.array([-0.5]),
        capital=1_000.0,
        prices=np.array([100.0]),
        contract_multipliers=dataset.contract_multipliers,
    )

    result = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            borrow_rate_multiplier=1.0,
            maintenance_margin_rate=0.5,
        ),
    ).execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    gap_fraction = 64.0 / (365.0 * 24.0)
    expected_borrow = 500.0 * 219.0 * gap_fraction
    assert expected_borrow == pytest.approx(800.0)
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.termination_reason == EconomicTerminationReason.MARGIN_CALL.value
    assert result.book.quantities[0] == pytest.approx(0.0)
    assert result.book.cash == pytest.approx(100.0)
