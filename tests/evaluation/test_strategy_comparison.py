from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import trade_rl.evaluation as evaluation
import trade_rl.evaluation.comparison.strategies as comparison_strategies
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import (
    UniversalStrategyComparison,
    compare_strategies,
    compare_strategies_by_symbol,
    compare_strategy_factories_by_symbol,
)
from trade_rl.evaluation.replay import run_shared_cash_replay
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


def charged_two_symbol_market() -> MarketDataset:
    return replace(
        two_symbol_market(),
        dataset_id="a" * 64,
        funding_rate=np.full((4, 2), 0.001),
        funding_due=np.ones((4, 2), dtype=np.bool_),
        borrow_rate=np.full((4, 2), 0.25),
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


def test_shared_cash_factory_comparison_uses_one_identical_replay_per_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = charged_two_symbol_market()
    cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.001,
        borrow_rate_multiplier=1.0,
    )
    risk = comparison_strategies.PreTradeRisk()
    created: list[tuple[str, object]] = []

    def factory(name: str, intent: PositionIntent):
        def create() -> ConstantIntentStrategy:
            strategy = ConstantIntentStrategy(intent)
            created.append((name, strategy))
            return strategy

        return create

    def hold_factory() -> StatefulFirstLongStrategy:
        strategy = StatefulFirstLongStrategy()
        created.append(("hold", strategy))
        return strategy

    factories = {
        "short": factory("short", PositionIntent.SHORT),
        "hold": hold_factory,
        "cash": factory("cash", PositionIntent.FLAT),
    }
    replay_calls: list[tuple[MarketDataset, tuple[object, ...], dict[str, object]]] = []

    def record_replay(
        replay_dataset: MarketDataset,
        strategies: tuple[object, ...],
        **kwargs: object,
    ):
        assert len(created) == 6
        replay_calls.append((replay_dataset, strategies, kwargs))
        return run_shared_cash_replay(replay_dataset, strategies, **kwargs)

    monkeypatch.setattr(
        comparison_strategies,
        "run_shared_cash_replay",
        record_replay,
        raising=False,
    )
    comparison = evaluation.compare_shared_cash_strategy_factories(
        dataset,
        factories,
        start_index=0,
        stop_index=3,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=cost,
        risk=risk,
        minimum_hold_bars=(2, 2),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert tuple(entry.name for entry in comparison) == ("short", "hold", "cash")
    assert len(replay_calls) == 3
    assert len(created) == len({id(strategy) for _, strategy in created}) == 6
    expected_kwargs = {
        "start_index": 0,
        "stop_index": 3,
        "gross_budget": 0.2,
        "initial_capital": 1_000.0,
        "execution_cost": cost,
        "risk": risk,
        "market_order_profile": None,
        "minimum_hold_bars": (2, 2),
        "settle_terminal_position": True,
        "capture_ledger_evidence": True,
    }
    for replay_dataset, strategies, kwargs in replay_calls:
        assert replay_dataset is dataset
        assert len(strategies) == dataset.n_symbols
        assert len({id(strategy) for strategy in strategies}) == dataset.n_symbols
        assert kwargs == expected_kwargs

    for entry in comparison:
        replay = entry.replay
        assert len(replay.book.quantities) == dataset.n_symbols
        assert replay.ledger_evidence is not None
        assert replay.ledger_evidence.intervals[0].cash_before == pytest.approx(1_000.0)
        assert replay.ledger_evidence.intervals[
            0
        ].portfolio_value_before == pytest.approx(1_000.0)
        assert replay.ledger_evidence.terminal_exact_quantities == ("0", "0")
        assert replay.ledger_evidence.active_order_remainders == ()
        diagnostics = replay.diagnostics
        assert entry.metrics.turnover_total == diagnostics.turnover_total
        assert entry.metrics.total_cost == diagnostics.total_cost
        assert entry.metrics.funding_pnl == diagnostics.funding_pnl
        assert entry.metrics.borrow_cost == diagnostics.borrow_cost
        assert entry.metrics.n_trades == diagnostics.n_trades
        assert entry.metrics.rebalance_events == diagnostics.rebalance_events
        assert entry.metrics.termination_count == diagnostics.termination_count

    short_metrics = comparison[0].metrics
    assert short_metrics.turnover_total > 0.0
    assert short_metrics.total_cost > 0.0
    assert short_metrics.funding_pnl > 0.0
    assert short_metrics.borrow_cost > 0.0

    hold_replay = comparison[1].replay
    assert hold_replay.decisions[1].minimum_hold_suppressed == (True, True)
    assert hold_replay.decisions[1].position_quantity_before == pytest.approx(
        hold_replay.decisions[0].position_quantity_after
    )
    assert hold_replay.decisions[1].position_quantity_after == pytest.approx(
        hold_replay.decisions[0].position_quantity_after
    )


@pytest.mark.parametrize(
    ("invalid_kind", "error_type", "message"),
    [
        ("empty", ValueError, "at least one"),
        ("name", ValueError, "name"),
        ("non_callable", TypeError, "callable"),
    ],
)
def test_shared_cash_factory_roster_is_validated_before_replay(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    replay_calls = 0
    factory_calls = 0

    def fail_if_replayed(*args: object, **kwargs: object) -> None:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("invalid roster reached replay")

    def valid_factory() -> ConstantIntentStrategy:
        nonlocal factory_calls
        factory_calls += 1
        return ConstantIntentStrategy(PositionIntent.FLAT)

    factories: object
    if invalid_kind == "empty":
        factories = {}
    elif invalid_kind == "name":
        factories = {"": valid_factory}
    else:
        factories = {"valid": valid_factory, "bad": None}

    monkeypatch.setattr(
        comparison_strategies,
        "run_shared_cash_replay",
        fail_if_replayed,
        raising=False,
    )

    with pytest.raises(error_type, match=message):
        evaluation.compare_shared_cash_strategy_factories(
            two_symbol_market(),
            factories,  # type: ignore[arg-type]
            start_index=0,
            stop_index=3,
            gross_budget=0.2,
        )

    assert replay_calls == 0
    assert factory_calls == 0


@pytest.mark.parametrize("reuse_scope", ["symbols", "arms"])
def test_shared_cash_factory_reuse_fails_before_replay(
    monkeypatch: pytest.MonkeyPatch, reuse_scope: str
) -> None:
    shared = StatefulFirstLongStrategy()
    replay_calls = 0

    def fail_if_replayed(*args: object, **kwargs: object) -> None:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("reused adapter reached replay")

    monkeypatch.setattr(
        comparison_strategies,
        "run_shared_cash_replay",
        fail_if_replayed,
        raising=False,
    )

    factories = (
        {"first": lambda: shared}
        if reuse_scope == "symbols"
        else {"first": lambda: shared, "second": lambda: shared}
    )

    with pytest.raises(ValueError, match="distinct strategy instance"):
        evaluation.compare_shared_cash_strategy_factories(
            two_symbol_market(),
            factories,
            start_index=0,
            stop_index=3,
            gross_budget=0.2,
        )

    assert replay_calls == 0


def test_shared_cash_factory_invalid_adapter_fails_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_calls = 0
    factory_calls = 0

    def invalid_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return object()

    def fail_if_replayed(*args: object, **kwargs: object) -> None:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("invalid adapter reached replay")

    monkeypatch.setattr(
        comparison_strategies,
        "run_shared_cash_replay",
        fail_if_replayed,
        raising=False,
    )

    with pytest.raises(TypeError, match="return a SingleSymbolStrategy"):
        evaluation.compare_shared_cash_strategy_factories(
            two_symbol_market(),
            {"invalid": invalid_factory},
            start_index=0,
            stop_index=3,
            gross_budget=0.2,
        )

    assert factory_calls == 1
    assert replay_calls == 0


def test_shared_cash_factory_comparison_is_exposed_from_evaluation_public_api() -> None:
    assert evaluation.SharedCashStrategyComparisonEntry is (
        comparison_strategies.SharedCashStrategyComparisonEntry
    )
    assert evaluation.compare_shared_cash_strategy_factories is (
        comparison_strategies.compare_shared_cash_strategy_factories
    )
