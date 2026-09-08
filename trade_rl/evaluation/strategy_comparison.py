"""Common replay-and-metrics runner for comparable single-symbol strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.replay import (
    SingleSymbolReplayResult,
    run_single_symbol_replay,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.interface import SingleSymbolStrategy


@dataclass(frozen=True, slots=True)
class StrategyComparisonEntry:
    """One named strategy evaluated on the shared replay contract."""

    name: str
    replay: SingleSymbolReplayResult
    metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    """Ordered strategy results generated under one shared evaluation setup."""

    entries: tuple[StrategyComparisonEntry, ...]


def compare_strategies(
    dataset: MarketDataset,
    strategies: Mapping[str, SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
) -> StrategyComparison:
    """Evaluate named strategies with identical replay and metric semantics."""

    if not strategies:
        raise ValueError("at least one strategy is required")

    entries: list[StrategyComparisonEntry] = []
    for name, strategy in strategies.items():
        if not isinstance(name, str) or not name:
            raise ValueError("strategy name must be non-empty")
        replay = run_single_symbol_replay(
            dataset,
            strategy,
            start_index=start_index,
            stop_index=stop_index,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
        )
        diagnostics = replay.diagnostics
        metrics = evaluate_performance(
            replay.returns,
            turnover_total=diagnostics.turnover_total,
            total_cost=diagnostics.total_cost,
            funding_pnl=diagnostics.funding_pnl,
            borrow_cost=diagnostics.borrow_cost,
            n_trades=diagnostics.n_trades,
            rebalance_events=diagnostics.rebalance_events,
            termination_count=diagnostics.termination_count,
        )
        entries.append(
            StrategyComparisonEntry(
                name=name,
                replay=replay,
                metrics=metrics,
            )
        )

    return StrategyComparison(entries=tuple(entries))


__all__ = ["StrategyComparison", "StrategyComparisonEntry", "compare_strategies"]
