from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig


@dataclass(frozen=True)
class ManualBook:
    cash: float
    quantities: tuple[float, float]
    marks: tuple[float, float]

    @property
    def position_values(self) -> tuple[float, float]:
        return tuple(
            qty * mark for qty, mark in zip(self.quantities, self.marks, strict=True)
        )

    @property
    def equity(self) -> float:
        return self.cash + sum(self.position_values)


def _manual_trade(
    book: ManualBook,
    *,
    targets: tuple[float, float],
    fill_prices: tuple[float, float],
    fee_rate: float,
) -> ManualBook:
    deltas = tuple(
        target - current
        for target, current in zip(targets, book.quantities, strict=True)
    )
    traded_notional = sum(
        abs(delta * price) for delta, price in zip(deltas, fill_prices, strict=True)
    )
    cash = book.cash - sum(
        delta * price for delta, price in zip(deltas, fill_prices, strict=True)
    )
    cash -= traded_notional * fee_rate
    return ManualBook(cash=cash, quantities=targets, marks=fill_prices)


def _manual_mark(book: ManualBook, marks: tuple[float, float]) -> ManualBook:
    return ManualBook(cash=book.cash, quantities=book.quantities, marks=marks)


def _manual_split(book: ManualBook, factors: tuple[float, float]) -> ManualBook:
    return ManualBook(
        cash=book.cash,
        quantities=tuple(
            qty * factor for qty, factor in zip(book.quantities, factors, strict=True)
        ),
        marks=tuple(
            mark / factor for mark, factor in zip(book.marks, factors, strict=True)
        ),
    )


def _manual_delist(
    book: ManualBook,
    *,
    symbol: int,
    settlement_price: float,
    recovery: float,
) -> ManualBook:
    quantities = list(book.quantities)
    proceeds = quantities[symbol] * settlement_price * recovery
    quantities[symbol] = 0.0
    marks = list(book.marks)
    marks[symbol] = settlement_price
    return ManualBook(
        cash=book.cash + proceeds,
        quantities=tuple(quantities),
        marks=tuple(marks),
    )


def _market_for_partial_fill() -> MarketDataset:
    n = 5
    prices = np.tile(np.asarray((100.0, 50.0)), (n, 1))
    return MarketDataset(
        dataset_id="8" * 64,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 2, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.tile(np.asarray((4.0, 8.0)), (n, 1)),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


@pytest.mark.parametrize("fee_rate", [0.0, 0.001])
def test_trade_mark_and_log_growth_match_independent_manual_calculation(
    fee_rate: float,
) -> None:
    manual = ManualBook(
        cash=10_000.0,
        quantities=(0.0, 0.0),
        marks=(100.0, 50.0),
    )
    manual = _manual_trade(
        manual,
        targets=(40.0, 40.0),
        fill_prices=(100.0, 50.0),
        fee_rate=fee_rate,
    )
    manual = _manual_mark(manual, (110.0, 45.0))

    production = BookState.zero(2, 10_000.0, np.asarray((100.0, 50.0)))
    cost = (40.0 * 100.0 + 40.0 * 50.0) * fee_rate
    production.execute(
        fill_prices=np.asarray((100.0, 50.0)),
        target_quantities=np.asarray((40.0, 40.0)),
        cost_amount=cost,
        turnover=0.6,
    )
    production.mark_to_market(
        mark_prices=np.asarray((110.0, 45.0)),
        funding_amount=0.0,
        period_start_value=10_000.0,
    )

    assert production.cash == pytest.approx(manual.cash)
    assert production.quantities == pytest.approx(manual.quantities)
    assert production.position_values == pytest.approx(manual.position_values)
    assert production.portfolio_value == pytest.approx(manual.equity)
    expected_pnl = manual.equity - 10_000.0
    assert production.portfolio_value - 10_000.0 == pytest.approx(expected_pnl)

    expected_log_return = math.log(manual.equity / 10_000.0)
    actual_log_return = math.log(production.portfolio_value / 10_000.0)
    assert actual_log_return == pytest.approx(expected_log_return)


def test_partial_fill_matches_capacity_calculation() -> None:
    dataset = _market_for_partial_fill()
    executor = MarketExecutor(
        dataset,
        ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.5,
            maintenance_margin_rate=0.0,
        ),
    )
    result = executor.execute_interval(
        BookState.zero(2, 1_000.0, dataset.close[0]),
        np.asarray((0.5, 0.5)),
        start_index=0,
        bars=1,
    )

    assert result.requested_notional_by_symbol == pytest.approx((500.0, 500.0))
    assert result.filled_notional_by_symbol == pytest.approx((200.0, 200.0))
    assert result.book.quantities == pytest.approx((2.0, 4.0))
    assert result.book.cash == pytest.approx(600.0)
    assert result.book.portfolio_value == pytest.approx(1_000.0)
    assert result.fill_ratio == pytest.approx(0.4)
    assert result.unfilled_turnover == pytest.approx(0.6)


def test_split_and_delisting_match_independent_manual_calculation() -> None:
    manual = ManualBook(
        cash=1_000.0,
        quantities=(10.0, 20.0),
        marks=(100.0, 50.0),
    )
    manual = _manual_split(manual, (2.0, 1.0))
    assert manual.equity == pytest.approx(3_000.0)
    manual = _manual_delist(
        manual,
        symbol=1,
        settlement_price=40.0,
        recovery=0.25,
    )
    manual = _manual_mark(manual, (55.0, 40.0))

    production = BookState(
        quantities=np.asarray((10.0, 20.0)),
        cash=1_000.0,
        mark_prices=np.asarray((100.0, 50.0)),
        peak_value=3_000.0,
    )
    production.apply_split(np.asarray((2.0, 1.0)))
    production.settle_positions(
        mask=np.asarray((False, True)),
        prices=np.asarray((55.0, 40.0)),
        recovery=np.asarray((1.0, 0.25)),
    )

    assert production.cash == pytest.approx(manual.cash)
    assert production.quantities == pytest.approx(manual.quantities)
    assert production.position_values == pytest.approx(manual.position_values)
    assert production.portfolio_value == pytest.approx(manual.equity)


def test_margin_shortfall_matches_independent_manual_calculation() -> None:
    production = BookState.from_weights(
        weights=np.asarray((1.5, -0.5)),
        capital=1_000.0,
        prices=np.asarray((100.0, 100.0)),
        max_gross=2.0,
    )
    gross_notional = 2_000.0
    maintenance_requirement = 0.60 * gross_notional
    expected_deficit = maintenance_requirement - 1_000.0

    production.set_margin(
        margin_used=1_000.0,
        maintenance_margin=0.60,
        maintenance_requirement=maintenance_requirement,
    )

    assert production.margin_deficit == pytest.approx(expected_deficit)
    assert production.insolvent is True
    assert production.termination_reason is EconomicTerminationReason.MARGIN_CALL


@pytest.mark.parametrize("direction", [1, -1])
def test_signed_multisymbol_lifecycle_matches_fraction_cash_and_equity(
    direction: int,
) -> None:
    # Each event uses independently calculated signed principal and carry.
    # Dyadic quantities, prices, multipliers and rates are exactly representable.
    cash = Fraction(1024)
    quantities = (Fraction(0), Fraction(0))
    marks = (Fraction(16), Fraction(32))
    multipliers = (Fraction(2), Fraction(1, 2))

    def vector(values: tuple[Fraction, Fraction]) -> np.ndarray:
        return np.array([float(value) for value in values])

    def equity() -> Fraction:
        return cash + sum(
            quantity * mark * multiplier
            for quantity, mark, multiplier in zip(
                quantities, marks, multipliers, strict=True
            )
        )

    book = BookState.zero(2, float(cash), vector(marks), vector(multipliers))

    def check() -> None:
        assert book.cash == float(cash)
        assert book.exact_quantities == quantities
        assert book.portfolio_value == float(equity())

    targets = (Fraction(4 * direction), Fraction(-6 * direction))
    fee = Fraction(7, 4)
    cash -= (
        sum(
            target * mark * multiplier
            for target, mark, multiplier in zip(
                targets, marks, multipliers, strict=True
            )
        )
        + fee
    )
    quantities = targets
    book.execute(
        fill_prices=vector(marks),
        target_quantities=vector(targets),
        cost_amount=float(fee),
        turnover=0.5,
    )
    check()
    assert book.total_cost == float(fee)
    assert book.fill_count == 2

    before_split = equity()
    factors = (Fraction(2), Fraction(1, 2))
    quantities = (quantities[0] * factors[0], quantities[1] * factors[1])
    marks = (marks[0] / factors[0], marks[1] / factors[1])
    book.apply_split(vector(factors))
    assert equity() == before_split
    check()

    dividends = (Fraction(1, 4), Fraction(1))
    dividend = sum(
        quantity * multiplier * amount
        for quantity, multiplier, amount in zip(
            quantities, multipliers, dividends, strict=True
        )
    )
    cash += dividend
    assert book.apply_dividend(vector(dividends)) == float(dividend)
    check()

    interest = cash * Fraction(1, 8) * Fraction(1, 4)
    cash += interest
    assert book.apply_cash_interest(0.125, year_fraction=0.25) == float(interest)
    check()
    borrow = Fraction(3, 8)
    cash -= borrow
    book.charge_borrow(float(borrow))
    check()
    assert book.borrow_cost == float(borrow)

    before_mark = equity()
    marks = (Fraction(9), Fraction(60))
    rates = (Fraction(1, 64), Fraction(1, 32))
    funding = -sum(
        quantity * mark * multiplier * rate
        for quantity, mark, multiplier, rate in zip(
            quantities, marks, multipliers, rates, strict=True
        )
    )
    cash += funding
    net_return = book.mark_to_market(
        mark_prices=vector(marks),
        funding_amount=float(funding),
        period_start_value=float(before_mark),
    )
    check()
    assert book.funding_pnl == float(funding)
    assert net_return == pytest.approx(float(equity() / before_mark - 1), abs=1e-15)

    recovery = (Fraction(3, 4), Fraction(1, 2))
    proceeds = sum(
        quantity * mark * multiplier * ratio
        for quantity, mark, multiplier, ratio in zip(
            quantities, marks, multipliers, recovery, strict=True
        )
    )
    cash += proceeds
    quantities = (Fraction(0), Fraction(0))
    assert book.settle_positions(
        mask=np.array([True, True]), prices=vector(marks), recovery=vector(recovery)
    ) == float(proceeds)
    check()
    assert book.insolvent is False
