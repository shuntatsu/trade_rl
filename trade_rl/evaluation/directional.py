"""Shared-account directional screening with an explicit drawdown budget."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import compound_return, evaluate_performance
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class CloseAtEndStrategy:
    """Schedule actual closing orders, including configured execution latency."""

    def __init__(self, strategy: SingleSymbolStrategy, *, close_index: int) -> None:
        self.strategy = strategy
        self.close_index = close_index

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        if observation.index >= self.close_index:
            return PositionIntent.FLAT
        return self.strategy.decide(observation)


def evaluate_directional_arm(
    dataset: MarketDataset,
    strategy_factory: Callable[[], SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    latency_bars: int = 0,
    cost_multiplier: float = 1.0,
    symbol_index: int | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed arm; a pass is a development screen only."""
    if (
        isinstance(latency_bars, bool)
        or not isinstance(latency_bars, int)
        or latency_bars < 0
    ):
        raise ValueError("latency_bars must be a nonnegative integer")
    if stop_index - start_index < latency_bars + 3:
        raise ValueError("evaluation window must allow opening and closing")
    if symbol_index is not None and (
        isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not 0 <= symbol_index < dataset.n_symbols
    ):
        raise ValueError("symbol_index is outside the dataset")
    if (
        isinstance(cost_multiplier, bool)
        or not np.isfinite(cost_multiplier)
        or cost_multiplier < 1
    ):
        raise ValueError("cost_multiplier cannot discount observed costs")
    execution = replace(
        ExecutionCostConfig.zero(),
        max_leverage=1.0,
        processing_bar_volume_capacity=False,
        order_latency_bars=latency_bars,
        multiplier=cost_multiplier,
        borrow_rate_multiplier=1.0,
    )
    strategies = tuple(
        CloseAtEndStrategy(
            strategy_factory()
            if symbol_index is None or index == symbol_index
            else ConstantIntentStrategy(PositionIntent.FLAT),
            close_index=stop_index - latency_bars - 1,
        )
        for index in range(dataset.n_symbols)
    )
    replay = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=0.1,
        initial_capital=10_000.0,
        execution_cost=execution,
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.5,
                max_abs_weight=0.1,
                max_turnover=None,
                drawdown_start=0.1,
                drawdown_stop=0.2,
            )
        ),
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
        termination_count=len(diagnostics.termination_reasons),
    )
    years = (
        dataset.timestamps[start_index:stop_index].astype("datetime64[Y]").astype(str)
    )
    values = np.array(replay.returns.values)
    full_length = len(values) == stop_index - start_index
    year_returns = {
        year: compound_return(
            tuple(float(x) for x in values[years[: len(values)] == year])
        )
        for year in sorted(set(years[: len(values)]))
    }
    terminal_flat = bool(np.all(np.abs(replay.book.quantities) <= 1e-10))
    qualified = bool(
        full_length
        and terminal_flat
        and not diagnostics.termination_reasons
        and metrics.total_return > 0
        and replay.book.max_drawdown <= 0.2
        and year_returns
        and all(value > 0 for value in year_returns.values())
    )
    return {
        "schema": "directional_arm_v1",
        "dataset_id": dataset.dataset_id,
        "start_index": start_index,
        "stop_index": stop_index,
        "initial_capital": 10_000.0,
        "latency_bars": latency_bars,
        "cost_multiplier": cost_multiplier,
        "symbol_index": symbol_index,
        "execution_policy_digest": execution.execution_policy_digest,
        "risk": asdict(
            PreTradeRiskConfig(max_gross=0.5, max_abs_weight=0.1, max_turnover=None)
        ),
        "metrics": asdict(metrics),
        "year_returns": year_returns,
        "returns": values.tolist(),
        "terminal_flat": terminal_flat,
        "terminal_quantities": replay.book.quantities.tolist(),
        "ledger_max_drawdown": replay.book.max_drawdown,
        "termination_reasons": list(diagnostics.termination_reasons),
        "qualified": qualified,
        "production_eligible": False,
    }
