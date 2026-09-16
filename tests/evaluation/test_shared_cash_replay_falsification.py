from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import replay as replay_module
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class RecordingStrategy:
    intent: PositionIntent
    label: str
    events: list[str]
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.events.append(f"decide:{self.label}")
        self.observations.append(observation)
        return self.intent


def _market(*, n_symbols: int, volume: float = 1_000_000.0) -> MarketDataset:
    n_bars = 6
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    symbols = tuple(f"SYM{index}" for index in range(n_symbols))
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, n_symbols), volume, dtype=np.float64),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_all_symbol_decisions_finish_before_risk_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market(n_symbols=2)
    events: list[str] = []
    strategies = (
        RecordingStrategy(PositionIntent.LONG, "0", events),
        RecordingStrategy(PositionIntent.SHORT, "1", events),
    )
    original_constrain = PreTradeRisk.constrain
    original_execute = MarketExecutor.execute_interval

    def constrain(controller: PreTradeRisk, target: np.ndarray, **kwargs: Any) -> Any:
        assert events[-2:] == ["decide:0", "decide:1"]
        events.append("risk")
        return original_constrain(controller, target, **kwargs)

    def execute(
        executor: MarketExecutor,
        book: object,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> Any:
        assert events[-1] == "risk"
        events.append("execute")
        return original_execute(
            executor,
            book,  # type: ignore[arg-type]
            target,
            start_index=start_index,
            bars=bars,
        )

    monkeypatch.setattr(PreTradeRisk, "constrain", constrain)
    monkeypatch.setattr(MarketExecutor, "execute_interval", execute)

    replay_module.run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert events == [
        "decide:0",
        "decide:1",
        "risk",
        "execute",
        "decide:0",
        "decide:1",
        "risk",
        "execute",
        "decide:0",
        "decide:1",
        "risk",
        "execute",
    ]


def test_one_symbol_equivalence_survives_cost_and_capacity_constraints() -> None:
    dataset = _market(n_symbols=1, volume=10.0)
    cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.0002,
        impact_rate=0.0001,
        max_participation_rate=0.1,
    )
    single_strategy = RecordingStrategy(PositionIntent.LONG, "single", [])
    shared_strategy = RecordingStrategy(PositionIntent.LONG, "shared", [])

    single = replay_module.run_single_symbol_replay(
        dataset,
        single_strategy,
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=cost,
    )
    shared = replay_module.run_shared_cash_replay(
        dataset,
        (shared_strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=cost,
    )

    assert shared.returns.values == pytest.approx(single.returns.values)
    np.testing.assert_allclose(shared.book.quantities, single.book.quantities)
    np.testing.assert_allclose(shared.book.weights, single.book.weights)
    assert shared.book.cash == pytest.approx(single.book.cash)
    assert shared.book.portfolio_value == pytest.approx(single.book.portfolio_value)
    assert shared.book.total_cost == pytest.approx(single.book.total_cost)
    assert shared.book.turnover_total == pytest.approx(single.book.turnover_total)
    assert shared.book.fill_count == single.book.fill_count
    assert shared.book.rebalance_events == single.book.rebalance_events
    assert shared.diagnostics == single.diagnostics
