"""Common replay-and-metrics runners for comparable strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.replay import (
    SingleSymbolReplayResult,
    run_single_symbol_replay,
)
from trade_rl.risk import PreTradeRisk
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
    """Ordered strategy results for one independently replayed symbol."""

    entries: tuple[StrategyComparisonEntry, ...]


@dataclass(frozen=True, slots=True)
class SymbolStrategyComparison:
    """One symbol's independent strategy comparison."""

    symbol_index: int
    symbol: str
    comparison: StrategyComparison


@dataclass(frozen=True, slots=True)
class UniversalStrategyComparison:
    """Per-symbol results; no aggregate result can hide a losing symbol."""

    by_symbol: tuple[SymbolStrategyComparison, ...]


def compare_strategies(
    dataset: MarketDataset,
    strategies: Mapping[str, SingleSymbolStrategy],
    *,
    symbol_index: int = 0,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
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
            symbol_index=symbol_index,
            start_index=start_index,
            stop_index=stop_index,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk=risk,
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


def compare_strategies_by_symbol(
    dataset: MarketDataset,
    strategies: Mapping[str, SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
) -> UniversalStrategyComparison:
    """Replay the same strategy objects independently on every dataset symbol."""

    by_symbol = tuple(
        SymbolStrategyComparison(
            symbol_index=symbol_index,
            symbol=symbol,
            comparison=compare_strategies(
                dataset,
                strategies,
                symbol_index=symbol_index,
                start_index=start_index,
                stop_index=stop_index,
                gross_budget=gross_budget,
                initial_capital=initial_capital,
                execution_cost=execution_cost,
                risk=risk,
            ),
        )
        for symbol_index, symbol in enumerate(dataset.symbols)
    )
    return UniversalStrategyComparison(by_symbol=by_symbol)


__all__ = [
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "UniversalStrategyComparison",
    "compare_strategies",
    "compare_strategies_by_symbol",
]
