from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.execution import StatefulExecutionResult

_TIMESTAMPS = np.array(
    ["2026-01-02T16:00:00", "2026-01-05T09:00:00", "2026-01-05T10:00:00"],
    dtype="datetime64[ns]",
)
_OPEN = np.full((3, 1), 100.0)
_CLOSE = np.array([[100.0], [99.0], [99.0]])


def _market(*, cash_rate: float, dividend: float, tag: str) -> MarketDataset:
    dividends = np.zeros((3, 1), dtype=np.float64)
    dividends[1, 0] = dividend
    return MarketDataset(
        dataset_id=tag * 64,
        symbols=("A",),
        timestamps=_TIMESTAMPS,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=_OPEN,
        high=_OPEN,
        low=_CLOSE,
        close=_CLOSE,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        dividend=dividends,
        cash_rate=np.array([0.0, cash_rate, cash_rate], dtype=np.float64),
    )


def _run_case(
    *, cash_rate: float, dividend: float, tag: str
) -> StatefulExecutionResult:
    market = _market(cash_rate=cash_rate, dividend=dividend, tag=tag)
    book = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=_OPEN[0],
        contract_multipliers=market.contract_multipliers,
    )
    return MarketExecutor(market, ExecutionCostConfig.zero()).execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )


def _assert_one_bar_without_trades(result: StatefulExecutionResult) -> None:
    assert result.bars_advanced == 1
    assert result.next_index == 1
    assert result.fill_count == 0
    assert result.rebalance_events == 0
    assert result.order_events == ()
    assert result.order_book.active_orders == ()
    assert result.termination_reason is None
    assert not result.book.insolvent
    assert result.book.termination_reason is None
    assert result.book.fill_count == 0
    assert result.book.rebalance_events == 0
    assert result.book.exact_quantities == (Fraction(5),)


def test_session_gap_interest_and_processing_dividend_use_original_basis() -> None:
    rate = 0.365
    gap_fraction = 64.0 / (365.0 * 24.0)
    processing_fraction = 1.0 / (365.0 * 24.0)
    gap_interest = 500.0 * rate * gap_fraction
    processing_interest = 505.0 * rate * processing_fraction
    expected_interest = gap_interest + processing_interest

    result = _run_case(cash_rate=rate, dividend=1.0, tag="a")

    _assert_one_bar_without_trades(result)
    assert result.interval_cash_interest == pytest.approx(expected_interest)
    assert result.interval_dividend == pytest.approx(5.0)
    assert result.book.cash == pytest.approx(
        500.0 + gap_interest + 5.0 + processing_interest
    )
    assert result.book.portfolio_value == pytest.approx(result.book.cash + 5.0 * 99.0)
    assert result.book.peak_value == pytest.approx(result.book.portfolio_value)
    assert result.book.max_drawdown == pytest.approx(0.0)


def test_session_gap_interest_without_dividend_preserves_gap_basis() -> None:
    rate = 0.365
    gap_fraction = 64.0 / (365.0 * 24.0)
    processing_fraction = 1.0 / (365.0 * 24.0)
    gap_interest = 500.0 * rate * gap_fraction
    processing_interest = 500.0 * rate * processing_fraction
    expected_interest = gap_interest + processing_interest
    expected_peak = 1_000.0 + gap_interest
    expected_end_value = 500.0 + expected_interest + 5.0 * 99.0

    result = _run_case(cash_rate=rate, dividend=0.0, tag="b")

    _assert_one_bar_without_trades(result)
    assert result.interval_cash_interest == pytest.approx(expected_interest)
    assert result.interval_dividend == pytest.approx(0.0)
    assert result.book.cash == pytest.approx(500.0 + expected_interest)
    assert result.book.portfolio_value == pytest.approx(expected_end_value)
    assert result.book.peak_value == pytest.approx(expected_peak)
    assert result.book.max_drawdown == pytest.approx(
        (expected_peak - expected_end_value) / expected_peak
    )


def test_processing_dividend_does_not_create_intrabar_drawdown_peak() -> None:
    result = _run_case(cash_rate=0.0, dividend=1.0, tag="c")

    _assert_one_bar_without_trades(result)
    assert result.interval_cash_interest == pytest.approx(0.0)
    assert result.interval_dividend == pytest.approx(5.0)
    assert result.book.cash == pytest.approx(505.0)
    assert result.book.portfolio_value == pytest.approx(1_000.0)
    assert result.book.peak_value == pytest.approx(1_000.0)
    assert result.book.max_drawdown == pytest.approx(0.0)
