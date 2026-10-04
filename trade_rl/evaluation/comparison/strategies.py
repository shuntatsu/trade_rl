"""Common replay-and-metrics runners for comparable strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.replay import (
    SharedCashReplayResult,
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
class SharedCashStrategyComparisonEntry:
    """One strategy evaluated across every symbol in a single shared account."""

    name: str
    replay: SharedCashReplayResult
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
    """Per-symbol results plus an optional explicit shared-cash PPO replay."""

    by_symbol: tuple[SymbolStrategyComparison, ...]
    shared_cash_ppo: SharedCashStrategyComparisonEntry | None = None
    ppo_training_timesteps: int | None = None
    ppo_training_minimum_hold_suppressed_count: int | None = None

    def __post_init__(self) -> None:
        if self.ppo_training_timesteps is not None and (
            isinstance(self.ppo_training_timesteps, bool)
            or not isinstance(self.ppo_training_timesteps, int)
            or self.ppo_training_timesteps <= 0
        ):
            raise ValueError("ppo_training_timesteps must be a positive integer")
        if self.ppo_training_minimum_hold_suppressed_count is not None and (
            isinstance(self.ppo_training_minimum_hold_suppressed_count, bool)
            or not isinstance(self.ppo_training_minimum_hold_suppressed_count, int)
            or self.ppo_training_minimum_hold_suppressed_count < 0
        ):
            raise ValueError(
                "ppo_training_minimum_hold_suppressed_count must be a non-negative integer"
            )


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
    settle_terminal_position: bool = False,
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
            settle_terminal_position=settle_terminal_position,
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
    settle_terminal_position: bool = False,
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
                settle_terminal_position=settle_terminal_position,
            ),
        )
        for symbol_index, symbol in enumerate(dataset.symbols)
    )
    return UniversalStrategyComparison(by_symbol=by_symbol)


def compare_strategy_factories_by_symbol(
    dataset: MarketDataset,
    strategy_factories: Mapping[str, Callable[[], SingleSymbolStrategy]],
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
    settle_terminal_position: bool = False,
) -> UniversalStrategyComparison:
    """Replay fresh strategy adapters per symbol while sharing frozen model state."""

    if not strategy_factories:
        raise ValueError("at least one strategy factory is required")
    for name, factory in strategy_factories.items():
        if not isinstance(name, str) or not name:
            raise ValueError("strategy factory name must be non-empty")
        if not callable(factory):
            raise TypeError("strategy factory must be callable")

    produced: dict[str, list[SingleSymbolStrategy]] = {
        name: [] for name in strategy_factories
    }
    by_symbol: list[SymbolStrategyComparison] = []
    for symbol_index, symbol in enumerate(dataset.symbols):
        strategies: dict[str, SingleSymbolStrategy] = {}
        for name, factory in strategy_factories.items():
            strategy = factory()
            if not callable(getattr(strategy, "decide", None)):
                raise TypeError("strategy factory must return a SingleSymbolStrategy")
            if any(strategy is previous for previous in produced[name]):
                raise ValueError(
                    "strategy factory must return a fresh instance for each symbol"
                )
            produced[name].append(strategy)
            strategies[name] = strategy
        by_symbol.append(
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
                    settle_terminal_position=settle_terminal_position,
                ),
            )
        )
    return UniversalStrategyComparison(by_symbol=tuple(by_symbol))


__all__ = [
    "SharedCashStrategyComparisonEntry",
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "UniversalStrategyComparison",
    "compare_strategies",
    "compare_strategies_by_symbol",
    "compare_strategy_factories_by_symbol",
]
