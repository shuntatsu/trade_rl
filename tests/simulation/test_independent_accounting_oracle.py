from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.rewards import RewardConfig, RewardTracker
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
def test_trade_mark_and_reward_match_independent_manual_calculation(
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
    tracker = RewardTracker(
        RewardConfig(
            scale=100.0,
            absolute_growth_weight=1.0,
            excess_growth_weight=0.0,
            incremental_drawdown_weight=0.0,
            baseline_underperformance_weight=0.0,
            projection_penalty_weight=0.0,
            terminal_equity_weight=0.0,
            margin_deficit_weight=0.0,
        )
    )
    reward = tracker.step(
        hybrid_log_return=expected_log_return,
        shadow_log_return=0.0,
        hybrid_drawdown=production.max_drawdown,
        shadow_drawdown=0.0,
    )
    assert reward.scaled_total == pytest.approx(100.0 * expected_log_return)


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

    # Independent calculation: prior closed-bar capacity is
    # 100*4*0.5 = 200 and 50*8*0.5 = 200 quote units.
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
