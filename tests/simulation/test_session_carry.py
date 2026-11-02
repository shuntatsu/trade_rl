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


def test_next_open_exit_excludes_post_fill_gap_carry() -> None:
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
        dataset_id="d" * 64,
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
    ).execute_interval(book, np.array([0.0]), start_index=0, bars=1)

    gap_fraction = 64.0 / (365.0 * 24.0)
    processing_bar_fraction = 1.0 / (365.0 * 24.0)
    expected_borrow = 500.0 * 0.365 * gap_fraction
    gap_interest = 1_500.0 * 0.365 * gap_fraction
    processing_interest = 1_000.0 * 0.365 * processing_bar_fraction
    expected_interest = gap_interest + processing_interest
    assert result.book.quantities[0] == pytest.approx(0.0)
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
    expected_borrow = 500.0 * 0.365 * gap_fraction + 600.0 * 0.365 * processing_fraction
    expected_interest = 1_500.0 * 0.365 * (gap_fraction + processing_fraction)
    assert result.book.quantities[0] == pytest.approx(-5.0)
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.interval_cash_interest == pytest.approx(expected_interest)


@pytest.mark.parametrize(
    ("next_open", "borrow_rate", "expected_borrow_amount", "expected_cash"),
    [
        (120.0, 219.0, 800.0, 100.0),
        (260.0, 0.0, 0.0, 200.0),
    ],
)
def test_gap_margin_call_flattens_at_next_open_not_previous_close(
    next_open: float,
    borrow_rate: float,
    expected_borrow_amount: float,
    expected_cash: float,
) -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    open_price = np.array([[100.0], [next_open], [next_open]])
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
        borrow_rate=np.full((3, 1), borrow_rate),
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
    expected_borrow = 500.0 * borrow_rate * gap_fraction
    assert expected_borrow == pytest.approx(expected_borrow_amount)
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.termination_reason == EconomicTerminationReason.MARGIN_CALL.value
    assert result.book.quantities[0] == pytest.approx(0.0)
    assert getattr(result.book, "_exact_quantities") == ("0",)
    assert result.book.cash == pytest.approx(expected_cash)


def test_session_gap_carry_drawdown_is_recorded_before_open_recovery() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    open_price = np.array([[100.0], [80.0], [80.0]])
    dataset = MarketDataset(
        dataset_id="f" * 64,
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
        borrow_rate=np.full((3, 1), 27.375),
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
        replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0),
    ).execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    gap_fraction = 64.0 / (365.0 * 24.0)
    gap_borrow = 500.0 * 27.375 * gap_fraction
    assert gap_borrow == pytest.approx(100.0)
    assert result.book.max_drawdown == pytest.approx(0.1)


def test_continuous_bar_retains_post_fill_one_bar_carry() -> None:
    timestamps = np.array(
        [
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            "2026-01-01T02:00:00",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.full((3, 1), 100.0)
    dataset = MarketDataset(
        dataset_id="f" * 64,
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
        periods_per_year=8_760,
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

    one_hour = 1.0 / (365.0 * 24.0)
    assert result.book.quantities[0] == pytest.approx(-5.0)
    assert result.interval_borrow_cost == pytest.approx(500.0 * 0.365 * one_hour)
    assert result.interval_cash_interest == pytest.approx(1_500.0 * 0.365 * one_hour)


@pytest.mark.parametrize(
    ("annual_rate", "dividend_per_unit", "processing_close"),
    [
        (0.365, 1.0, 100.0),
        (0.365, 0.0, 100.0),
        (0.0, 1.0, 100.0),
        (0.0, 1.0, 99.0),
    ],
)
def test_session_gap_interest_and_processing_dividend_preserve_account_state(
    annual_rate: float,
    dividend_per_unit: float,
    processing_close: float,
) -> None:
    """The public order path must not clone a transient pre-mark/peak state."""

    timestamps = np.array(
        ["2026-01-02T16:00:00", "2026-01-05T09:00:00", "2026-01-05T10:00:00"],
        dtype="datetime64[ns]",
    )
    opens = np.full((3, 1), 100.0)
    closes = opens.copy()
    closes[1, 0] = processing_close
    dataset = MarketDataset(
        dataset_id="9" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=opens,
        high=opens,
        low=np.minimum(opens, closes),
        close=closes,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        cash_rate=np.full(3, annual_rate),
        dividend=np.full((3, 1), dividend_per_unit),
    )
    book = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=opens[0],
        contract_multipliers=dataset.contract_multipliers,
    )

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    gap_interest = 500.0 * annual_rate * (64.0 / 8_760.0)
    dividend = 5.0 * dividend_per_unit
    processing_interest = (500.0 + dividend) * annual_rate / 8_760.0
    assert result.interval_cash_interest == pytest.approx(
        gap_interest + processing_interest
    )
    assert result.interval_dividend == pytest.approx(dividend)
    assert result.book.cash == pytest.approx(
        500.0 + gap_interest + dividend + processing_interest
    )
    assert result.book.quantities[0] == pytest.approx(5.0)
    assert result.book.exact_quantities == (5,)
    assert result.next_index == 1
    assert result.bars_advanced == 1
    assert result.fill_count == 0
    assert result.rebalance_events == 0
    assert not result.order_book.active_orders
    assert not result.order_events
    assert book.cash == 500.0
    assert book.peak_value == 1_000.0
    if annual_rate == 0.0 and processing_close == 99.0:
        assert result.book.portfolio_value == pytest.approx(1_000.0)
        assert result.book.peak_value == pytest.approx(1_000.0)
        assert result.book.max_drawdown == pytest.approx(0.0)
