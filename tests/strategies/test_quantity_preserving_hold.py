"""Corporate actions change units, never an unchanged strategy's exposure."""

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    run_shared_cash_replay,
    run_single_symbol_replay,
)
from trade_rl.simulation import BookState, ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def split_market(factor: float) -> MarketDataset:
    prices = np.array([100.0, 100.0, *([100.0 / factor] * 4)])[:, None]
    splits = np.array([1.0, 1.0, factor, 1.0, 1.0, 1.0])[:, None]
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("SPLIT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((6, 1, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full((6, 1), 1_000_000.0),
        funding_rate=np.zeros((6, 1)),
        tradable=np.ones((6, 1), dtype=np.bool_),
        feature_available=np.ones((6, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        split_factor=splits,
    )


def held_book(dataset: MarketDataset, route: str, intent: PositionIntent) -> BookState:
    cost = replace(ExecutionCostConfig.zero(), fee_rate=0.001)
    if route == "training":
        env = PPOTradingEnv(
            dataset,
            feature_indices=(0,),
            start_index=0,
            stop_index=5,
            gross_budget=0.5,
            initial_capital=1000.0,
            execution_cost=cost,
        )
        env.reset()
        for _ in range(5):
            env.step(2 if intent is PositionIntent.LONG else 0)
        return env.book
    strategy = ConstantIntentStrategy(intent)
    kwargs = dict(
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000.0,
        execution_cost=cost,
    )
    if route == "single":
        return run_single_symbol_replay(dataset, strategy, **kwargs).book
    return run_shared_cash_replay(dataset, (strategy,), **kwargs).book


@pytest.mark.parametrize("route", ["training", "single", "shared"])
@pytest.mark.parametrize("intent", [PositionIntent.LONG, PositionIntent.SHORT])
@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_unchanged_intent_preserves_split_adjusted_quantity(
    route: str, intent: PositionIntent, factor: float
) -> None:
    book = held_book(split_market(factor), route, intent)

    # Independent cash/unit oracle: one 500 notional entry costs 0.5; a split
    # changes the five signed units and their price inversely without a fill.
    sign = 1.0 if intent is PositionIntent.LONG else -1.0
    assert book.quantities == pytest.approx([sign * 5.0 * factor])
    assert book.fill_count == 1
    assert book.total_cost == pytest.approx(0.5)
    assert book.turnover_total == pytest.approx(0.5)
    assert book.cash == pytest.approx(1000.0 - sign * 500.0 - 0.5)
    assert book.portfolio_value == pytest.approx(999.5)


@pytest.mark.parametrize("route", ["training", "single", "shared"])
@pytest.mark.parametrize("intent", [PositionIntent.LONG, PositionIntent.SHORT])
def test_unchanged_intent_uses_mark_prices_to_preserve_quantity(
    route: str, intent: PositionIntent
) -> None:
    dataset = replace(
        split_market(1.0),
        mark_price=np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])[:, None],
    )
    book = held_book(dataset, route, intent)

    # Trade at 100, value at 150: a mark change changes weight and equity,
    # never the five signed units or the one entry fee.
    sign = 1.0 if intent is PositionIntent.LONG else -1.0
    assert book.quantities == pytest.approx([sign * 5.0])
    assert book.fill_count == 1
    assert book.total_cost == pytest.approx(0.5)
    assert book.cash == pytest.approx(1000.0 - sign * 500.0 - 0.5)
    assert book.portfolio_value == pytest.approx(999.5 + sign * 250.0)


@pytest.mark.parametrize("route", ["training", "single", "shared"])
def test_split_preserves_unfilled_entry_proposal(route: str) -> None:
    volume = np.full((6, 1), 1_000_000.0)
    volume[1, 0] = 1.0
    book = held_book(
        replace(split_market(2.0), volume=volume), route, PositionIntent.LONG
    )

    # Only one of five units fills before the split. The remaining entry is
    # still due in the new units; rebinding to the two filled units would lose it.
    assert book.quantities == pytest.approx([10.0])
    assert book.fill_count == 2
    assert book.total_cost == pytest.approx(0.5)
    assert book.cash == pytest.approx(499.5)
    assert book.portfolio_value == pytest.approx(999.5)


@pytest.mark.parametrize("route", ["training", "single", "shared"])
@pytest.mark.parametrize(
    "intent,expected_funding,expected_cash,expected_value",
    [
        (PositionIntent.LONG, -15.25, 484.25, 1234.25),
        (PositionIntent.SHORT, 15.25, 1514.75, 764.75),
    ],
)
def test_held_quantity_uses_settlement_products_separately_from_valuation_marks(
    route: str,
    intent: PositionIntent,
    expected_funding: float,
    expected_cash: float,
    expected_value: float,
) -> None:
    rates = np.zeros((6, 1))
    settlement_products = np.zeros((6, 1))
    counts = np.zeros((6, 1), dtype=np.int64)
    rates[2, 0] = 0.03
    settlement_products[2, 0] = 3.05
    counts[2, 0] = 2
    dataset = replace(
        split_market(1.0),
        mark_price=np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])[:, None],
        funding_rate=rates,
        funding_price_rate=settlement_products,
        funding_event_count=counts,
        funding_due=counts > 0,
    )
    book = held_book(dataset, route, intent)

    # Five signed units trade at 100, then remain unchanged. Two settlements
    # contribute 0.01 * 95 + 0.02 * 105 = 3.05 per unit, not 0.03 * bar mark.
    sign = 1.0 if intent is PositionIntent.LONG else -1.0
    assert book.quantities == pytest.approx([sign * 5.0])
    assert book.fill_count == 1
    assert book.total_cost == pytest.approx(0.5)
    assert book.funding_pnl == pytest.approx(expected_funding)
    assert book.cash == pytest.approx(expected_cash)
    assert book.portfolio_value == pytest.approx(expected_value)
