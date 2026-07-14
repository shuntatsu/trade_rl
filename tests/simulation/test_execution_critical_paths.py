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
    assert np.isfinite(executor._slippage_rates(3)).all()
    assert executor._round_quantities(
        np.array([1.13]), index=1
    ).tolist() == pytest.approx([1.0])
    constrained = executor._constrain_borrow(
        np.array([-2.0]), current=np.array([-1.0]), index=1
    )
    assert constrained.tolist() == [-1.0]


def _limit_executor(*, touched: bool) -> MarketExecutor:
    low = np.full((5, 1), 90.0 if touched else 100.0)
    high = np.full((5, 1), 110.0 if touched else 100.0)
    return MarketExecutor(
        market(low=low, high=high),
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


def test_limit_buy_sell_and_untouched_order_paths() -> None:
    buy_executor = _limit_executor(touched=True)
    buy_book = BookState.zero(1, 1_000.0, buy_executor.dataset.close[0])
    buy = buy_executor._fill_toward_quantities(
        buy_book,
        np.array([1.0]),
        prices=buy_executor.dataset.open[1],
        capacity_volume=buy_executor.dataset.volume[0],
        tradable=buy_executor.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert buy.filled_notional > 0.0
    assert buy.cost_amount > 0.0

    sell_book = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=buy_executor.dataset.close[0],
    )
    sell = buy_executor._fill_toward_quantities(
        sell_book,
        np.array([0.0]),
        prices=buy_executor.dataset.open[1],
        capacity_volume=buy_executor.dataset.volume[0],
        tradable=buy_executor.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert sell.filled_notional > 0.0

    untouched_executor = _limit_executor(touched=False)
    untouched = untouched_executor._fill_toward_quantities(
        BookState.zero(1, 1_000.0, untouched_executor.dataset.close[0]),
        np.array([1.0]),
        prices=untouched_executor.dataset.open[1],
        capacity_volume=untouched_executor.dataset.volume[0],
        tradable=untouched_executor.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert untouched.filled_notional == 0.0


def test_direction_minimum_notional_and_zero_capacity_paths() -> None:
    shape = (5, 1)
    blocked_dataset = market(
        buy_allowed=np.zeros(shape, dtype=np.bool_),
        sell_allowed=np.zeros(shape, dtype=np.bool_),
    )
    blocked_executor = MarketExecutor(blocked_dataset, ExecutionCostConfig.zero())
    blocked = blocked_executor._fill_toward_quantities(
        BookState.zero(1, 1_000.0, blocked_dataset.close[0]),
        np.array([1.0]),
        prices=blocked_dataset.open[1],
        capacity_volume=np.array([0.0]),
        tradable=np.array([True]),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert blocked.filled_notional == 0.0
    assert blocked.max_participation == 0.0

    minimum_dataset = market(minimum_notional=np.full(shape, 50.0))
    minimum_executor = MarketExecutor(minimum_dataset, ExecutionCostConfig.zero())
    too_small = minimum_executor._fill_toward_quantities(
        BookState.zero(1, 1_000.0, minimum_dataset.close[0]),
        np.array([0.1]),
        prices=minimum_dataset.open[1],
        capacity_volume=minimum_dataset.volume[0],
        tradable=minimum_dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert too_small.requested_notional == 0.0


def test_interval_latency_and_income_paths() -> None:
    cash_rate = np.zeros(5)
    cash_rate[1] = 0.05
    delayed_dataset = market(cash_rate=cash_rate)
    delayed = MarketExecutor(
        delayed_dataset,
        replace(ExecutionCostConfig.zero(), order_latency_bars=2),
    )
    no_order = delayed.execute_interval(
        BookState.zero(1, 1_000.0, delayed_dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert no_order.requested_turnover == 0.0
    assert no_order.fill_ratio == 1.0
    assert no_order.interval_cash_interest >= 0.0

    dividend = np.zeros((5, 1))
    dividend[1, 0] = 1.0
    funding = np.zeros((5, 1))
    funding[1, 0] = 0.01
    due = np.zeros((5, 1), dtype=np.bool_)
    due[1, 0] = True
    income_dataset = market(
        dividend=dividend,
        funding_rate=funding,
        funding_due=due,
    )
    invested = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=income_dataset.close[0],
    )
    result = MarketExecutor(
        income_dataset,
        ExecutionCostConfig.zero(),
    ).execute_interval(
        invested,
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert result.interval_dividend >= 0.0
    assert result.interval_funding <= 0.0


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
    assert 0.0 <= partial.fill_ratio < 1.0
    assert partial.termination_reason is None
