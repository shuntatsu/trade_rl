from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.simulation.test_critical_branch_coverage import market
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def test_slippage_lot_rounding_and_unavailable_borrow_paths() -> None:
    executor = MarketExecutor(
        market(
            lot_size=np.full((5, 1), 0.25),
            borrow_available=np.zeros((5, 1), dtype=np.bool_),
        ),
        ExecutionCostConfig(
            slippage_std=0.01,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=2.0,
            random_seed=7,
        ),
    )
    assert executor._slippage_rates(0).size == 0
    assert np.all(executor._slippage_rates(3) > 0.0)
    assert executor._round_quantities(np.array([1.13]), index=1).tolist() == pytest.approx(
        [1.0]
    )
    constrained = executor._constrain_borrow(
        np.array([-2.0]), current=np.array([-1.0]), index=1
    )
    assert constrained.tolist() == [-1.0]


def test_limit_buy_sell_maker_cost_and_untouched_order_paths() -> None:
    shape = (5, 2)
    low = np.full(shape, 90.0)
    high = np.full(shape, 110.0)
    dataset = market(2, low=low, high=high)
    executor = MarketExecutor(
        dataset,
        ExecutionCostConfig(
            fee_rate=0.0,
            maker_fee_rate=0.001,
            spread_rate=0.002,
            impact_rate=0.0,
            max_participation_rate=1.0,
            order_type="limit",
            limit_offset_rate=0.01,
            maintenance_margin_rate=0.0,
        ),
    )
    book = BookState.zero(2, 1_000.0, dataset.close[0])
    fill = executor._fill_toward_quantities(
        book,
        np.array([1.0, -1.0]),
        prices=dataset.open[1],
        capacity_volume=dataset.volume[0],
        tradable=dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert fill.fill_count == 2
    assert fill.cost_amount > 0.0
    assert book.quantities.tolist() == pytest.approx([1.0, -1.0])

    untouched_low = np.full(shape, 100.0)
    untouched = MarketExecutor(
        market(2, low=untouched_low, high=np.full(shape, 100.0)),
        replace(executor.cost, maker_fee_rate=0.0, spread_rate=0.0),
    )
    untouched_book = BookState.zero(2, 1_000.0, np.array([100.0, 100.0]))
    result = untouched._fill_toward_quantities(
        untouched_book,
        np.array([1.0, -1.0]),
        prices=np.array([100.0, 100.0]),
        capacity_volume=np.array([1_000.0, 1_000.0]),
        tradable=np.array([True, True]),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert result.filled_notional == 0.0


def test_direction_minimum_notional_zero_capacity_and_lot_paths() -> None:
    shape = (5, 2)
    dataset = market(
        2,
        buy_allowed=np.zeros(shape, dtype=np.bool_),
        sell_allowed=np.zeros(shape, dtype=np.bool_),
        minimum_notional=np.full(shape, 50.0),
        max_participation_rate=np.full(shape, 0.5),
        lot_size=np.full(shape, 0.1),
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(2, 1_000.0, dataset.close[0])
    blocked = executor._fill_toward_quantities(
        book,
        np.array([1.0, -1.0]),
        prices=dataset.open[1],
        capacity_volume=np.array([0.0, 0.0]),
        tradable=np.array([True, True]),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert blocked.filled_notional == 0.0
    assert blocked.max_participation == 0.0

    allowed = market(
        1,
        minimum_notional=np.full((5, 1), 50.0),
        lot_size=np.full((5, 1), 0.1),
    )
    allowed_executor = MarketExecutor(allowed, ExecutionCostConfig.zero())
    too_small = allowed_executor._fill_toward_quantities(
        BookState.zero(1, 1_000.0, allowed.close[0]),
        np.array([0.1]),
        prices=allowed.open[1],
        capacity_volume=allowed.volume[0],
        tradable=allowed.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert too_small.requested_notional == 0.0


def test_interval_latency_split_income_funding_and_cash_interest_paths() -> None:
    shape = (5, 1)
    split = np.ones(shape)
    split[1, 0] = 2.0
    dividend = np.zeros(shape)
    dividend[1, 0] = 1.0
    funding = np.zeros(shape)
    funding[1, 0] = 0.01
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[1, 0] = True
    cash_rate = np.zeros(5)
    cash_rate[1] = 0.05
    dataset = market(
        split_factor=split,
        dividend=dividend,
        funding_rate=funding,
        funding_due=funding_due,
        cash_rate=cash_rate,
    )
    delayed = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), order_latency_bars=2),
    )
    no_order = delayed.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert no_order.requested_turnover == 0.0
    assert no_order.fill_ratio == 1.0
    assert no_order.interval_cash_interest > 0.0

    invested = BookState.from_weights(
        weights=np.array([0.5]), capital=1_000.0, prices=dataset.close[0]
    )
    immediate = MarketExecutor(dataset, ExecutionCostConfig.zero())
    result = immediate.execute_interval(
        invested,
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert result.interval_dividend > 0.0
    assert result.interval_funding < 0.0


def test_liquidation_full_partial_and_zero_position_paths() -> None:
    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    invested = BookState.from_weights(
        weights=np.array([1.0]), capital=1_000.0, prices=dataset.close[0]
    )
    full = executor.liquidate_at_close(invested, index=1)
    assert full.termination_reason == EconomicTerminationReason.LIQUIDATION.value
    assert full.fill_ratio == 1.0

    zero = executor.liquidate_at_close(
        BookState.zero(1, 1_000.0, dataset.close[0]), index=1
    )
    assert zero.fill_ratio == 1.0

    low_volume = market(volume=np.full((5, 1), 0.01))
    partial_executor = MarketExecutor(
        low_volume,
        replace(ExecutionCostConfig.zero(), max_participation_rate=0.1),
    )
    partial_book = BookState.from_weights(
        weights=np.array([1.0]), capital=1_000.0, prices=low_volume.close[0]
    )
    partial = partial_executor.liquidate_at_close(partial_book, index=1)
    assert 0.0 < partial.fill_ratio < 1.0
    assert partial.termination_reason is None
