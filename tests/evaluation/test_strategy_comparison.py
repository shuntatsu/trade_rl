from __future__ import annotations

import numpy as np
import pytest

import trade_rl.evaluation as evaluation
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import (
    UniversalStrategyComparison,
    compare_strategies,
    compare_strategies_by_symbol,
    compare_strategy_factories_by_symbol,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


class StatefulFirstLongStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, observation) -> PositionIntent:
        del observation
        self.calls += 1
        return PositionIntent.LONG if self.calls == 1 else PositionIntent.FLAT


def market() -> MarketDataset:
    close = np.asarray([[100.0], [100.0], [110.0], [121.0]])
    return MarketDataset(
        dataset_id="8" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 1), 1_000_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def two_symbol_market() -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [110.0, 90.0],
            [121.0, 81.0],
        ]
    )
    return MarketDataset(
        dataset_id="9" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 2, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 2), 1_000_000.0),
        funding_rate=np.zeros((4, 2)),
        tradable=np.ones((4, 2), dtype=np.bool_),
        feature_available=np.ones((4, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_strategies_share_one_replay_and_metric_contract() -> None:
    comparison = compare_strategies(
        market(),
        {
            "cash": ConstantIntentStrategy(PositionIntent.FLAT),
            "long": ConstantIntentStrategy(PositionIntent.LONG),
            "short": ConstantIntentStrategy(PositionIntent.SHORT),
        },
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert tuple(entry.name for entry in comparison.entries) == (
        "cash",
        "long",
        "short",
    )
    cash, long, short = comparison.entries
    assert cash.metrics.total_return == pytest.approx(0.0)
    assert long.metrics.total_return > 0.0
    assert short.metrics.total_return < 0.0
    assert {entry.metrics.n_periods for entry in comparison.entries} == {3}
    assert {entry.replay.returns.kind for entry in comparison.entries} == {
        cash.replay.returns.kind
    }


def test_comparison_keeps_each_symbol_result_separate() -> None:
    comparison = compare_strategies_by_symbol(
        two_symbol_market(),
        {"long": ConstantIntentStrategy(PositionIntent.LONG)},
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert tuple(item.symbol for item in comparison.by_symbol) == (
        "BTCUSDT",
        "ETHUSDT",
    )
    btc = comparison.by_symbol[0].comparison.entries[0]
    eth = comparison.by_symbol[1].comparison.entries[0]
    assert btc.metrics.total_return > 0.0
    assert eth.metrics.total_return < 0.0
    assert btc.name == eth.name == "long"


def test_universal_comparison_is_exposed_from_evaluation_public_api() -> None:
    assert evaluation.UniversalStrategyComparison is UniversalStrategyComparison
    assert evaluation.compare_strategies_by_symbol is compare_strategies_by_symbol


def test_comparison_rejects_empty_or_invalid_strategy_names() -> None:
    with pytest.raises(ValueError, match="strategy"):
        compare_strategies(
            market(),
            {},
            start_index=0,
            stop_index=3,
            gross_budget=1.0,
        )

    with pytest.raises(ValueError, match="name"):
        compare_strategies(
            market(),
            {"": ConstantIntentStrategy(PositionIntent.FLAT)},
            start_index=0,
            stop_index=3,
            gross_budget=1.0,
        )


def test_factory_comparison_creates_fresh_strategy_for_each_symbol() -> None:
    created: list[StatefulFirstLongStrategy] = []

    def factory() -> StatefulFirstLongStrategy:
        strategy = StatefulFirstLongStrategy()
        created.append(strategy)
        return strategy

    comparison = compare_strategy_factories_by_symbol(
        two_symbol_market(),
        {"stateful": factory},
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert len(created) == 2
    assert created[0] is not created[1]
    assert tuple(
        item.comparison.entries[0].replay.decisions[0].intent
        for item in comparison.by_symbol
    ) == (PositionIntent.LONG, PositionIntent.LONG)


def test_factory_comparison_is_exposed_from_evaluation_public_api() -> None:
    assert (
        evaluation.compare_strategy_factories_by_symbol
        is compare_strategy_factories_by_symbol
    )
