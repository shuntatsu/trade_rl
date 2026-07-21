"""Deterministic execution-assumption sensitivity evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


class LimitFillAssumption(StrEnum):
    MARKET = "market"
    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class ExecutionSensitivityScenario:
    name: str
    fee_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    capacity_fraction: float = 1.0
    signal_delay_bars: int = 0
    limit_fill: LimitFillAssumption | str = LimitFillAssumption.MARKET
    tradability_delay_bars: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sensitivity scenario name must not be empty")
        for field_name, value in (
            ("fee_multiplier", self.fee_multiplier),
            ("spread_multiplier", self.spread_multiplier),
            ("slippage_multiplier", self.slippage_multiplier),
            ("capacity_fraction", self.capacity_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.capacity_fraction > 1.0:
            raise ValueError("capacity_fraction must not exceed one")
        for field_name, value in (
            ("signal_delay_bars", self.signal_delay_bars),
            ("tradability_delay_bars", self.tradability_delay_bars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        object.__setattr__(self, "limit_fill", LimitFillAssumption(self.limit_fill))


@dataclass(frozen=True, slots=True)
class ExecutionSensitivityResult:
    scenario: ExecutionSensitivityScenario
    ending_equity: float
    total_return: float
    interval_cost: float
    filled_turnover: float
    unfilled_turnover: float
    fill_ratio: float
    fill_count: int
    termination_reason: str | None


def default_execution_sensitivity_scenarios() -> tuple[ExecutionSensitivityScenario, ...]:
    scenarios: list[ExecutionSensitivityScenario] = []
    scenarios.extend(
        ExecutionSensitivityScenario(name=f"fee_{factor:g}x", fee_multiplier=factor)
        for factor in (1.0, 2.0, 4.0)
    )
    scenarios.extend(
        ExecutionSensitivityScenario(name=f"spread_{factor:g}x", spread_multiplier=factor)
        for factor in (1.0, 2.0)
    )
    scenarios.extend(
        ExecutionSensitivityScenario(
            name=f"slippage_{factor:g}x", slippage_multiplier=factor
        )
        for factor in (1.0, 2.0, 4.0)
    )
    scenarios.extend(
        ExecutionSensitivityScenario(
            name=f"capacity_{int(fraction * 100)}pct", capacity_fraction=fraction
        )
        for fraction in (1.0, 0.5, 0.25)
    )
    scenarios.extend(
        ExecutionSensitivityScenario(name=f"signal_delay_{bars}", signal_delay_bars=bars)
        for bars in (0, 1, 2)
    )
    scenarios.extend(
        ExecutionSensitivityScenario(name=f"limit_{mode.value}", limit_fill=mode)
        for mode in (
            LimitFillAssumption.OPTIMISTIC,
            LimitFillAssumption.NEUTRAL,
            LimitFillAssumption.CONSERVATIVE,
        )
    )
    scenarios.extend(
        ExecutionSensitivityScenario(
            name=f"tradability_delay_{bars}", tradability_delay_bars=bars
        )
        for bars in (0, 1)
    )
    return tuple(scenarios)


def _shift_tradability(dataset: MarketDataset, bars: int) -> MarketDataset:
    if bars == 0:
        return dataset
    shifted = np.zeros_like(dataset.tradable)
    if bars < dataset.n_bars:
        shifted[bars:] = dataset.tradable[:-bars]
    return replace(dataset, tradable=shifted)


def _limit_dataset(
    dataset: MarketDataset,
    assumption: LimitFillAssumption,
) -> MarketDataset:
    if assumption in {LimitFillAssumption.MARKET, LimitFillAssumption.NEUTRAL}:
        return dataset
    if assumption is LimitFillAssumption.OPTIMISTIC:
        span = np.maximum(dataset.high - dataset.low, np.abs(dataset.open) * 0.01)
        return replace(
            dataset,
            high=np.maximum(dataset.high, dataset.open + span),
            low=np.minimum(dataset.low, dataset.open - span),
        )
    conservative_high = np.maximum(
        dataset.close,
        dataset.open + 0.25 * (dataset.high - dataset.open),
    )
    conservative_low = np.minimum(
        dataset.close,
        dataset.open + 0.25 * (dataset.low - dataset.open),
    )
    return replace(
        dataset,
        high=conservative_high,
        low=conservative_low,
    )


def _scenario_cost(
    base: ExecutionCostConfig,
    scenario: ExecutionSensitivityScenario,
) -> ExecutionCostConfig:
    assumption = LimitFillAssumption(scenario.limit_fill)
    order_type = "market" if assumption is LimitFillAssumption.MARKET else "limit"
    limit_offset = base.limit_offset_rate
    if assumption is LimitFillAssumption.OPTIMISTIC:
        limit_offset = 0.0
    elif assumption is LimitFillAssumption.NEUTRAL:
        limit_offset = max(base.limit_offset_rate, 0.0005)
    elif assumption is LimitFillAssumption.CONSERVATIVE:
        limit_offset = max(base.limit_offset_rate, 0.001)
    return replace(
        base,
        fee_rate=base.fee_rate * scenario.fee_multiplier,
        maker_fee_rate=base.maker_fee_rate * scenario.fee_multiplier,
        taker_fee_rate=base.taker_fee_rate * scenario.fee_multiplier,
        spread_rate=base.spread_rate * scenario.spread_multiplier,
        slippage_std=base.slippage_std * scenario.slippage_multiplier,
        max_participation_rate=base.max_participation_rate
        * scenario.capacity_fraction,
        order_latency_bars=scenario.signal_delay_bars,
        order_type=order_type,
        limit_offset_rate=limit_offset,
    )


def evaluate_execution_sensitivity(
    *,
    dataset: MarketDataset,
    initial_book: BookState,
    target: np.ndarray,
    start_index: int,
    horizon_bars: int,
    base_cost: ExecutionCostConfig,
    scenarios: tuple[ExecutionSensitivityScenario, ...] | None = None,
) -> tuple[ExecutionSensitivityResult, ...]:
    selected = (
        default_execution_sensitivity_scenarios()
        if scenarios is None
        else scenarios
    )
    if not selected:
        raise ValueError("at least one execution sensitivity scenario is required")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if start_index < 0 or start_index + horizon_bars >= dataset.n_bars:
        raise ValueError("sensitivity range is outside the dataset")
    starting_equity = initial_book.portfolio_value
    if starting_equity <= 0.0:
        raise ValueError("sensitivity evaluation requires positive starting equity")

    results: list[ExecutionSensitivityResult] = []
    for scenario in selected:
        stressed_dataset = _shift_tradability(dataset, scenario.tradability_delay_bars)
        stressed_dataset = _limit_dataset(
            stressed_dataset,
            LimitFillAssumption(scenario.limit_fill),
        )
        executor = MarketExecutor(stressed_dataset, _scenario_cost(base_cost, scenario))
        execution = executor.execute_interval(
            initial_book,
            target,
            start_index=start_index,
            bars=horizon_bars,
        )
        results.append(
            ExecutionSensitivityResult(
                scenario=scenario,
                ending_equity=execution.book.portfolio_value,
                total_return=execution.book.portfolio_value / starting_equity - 1.0,
                interval_cost=execution.interval_cost,
                filled_turnover=execution.filled_turnover,
                unfilled_turnover=execution.unfilled_turnover,
                fill_ratio=execution.fill_ratio,
                fill_count=execution.fill_count,
                termination_reason=execution.termination_reason,
            )
        )
    return tuple(results)


__all__ = [
    "ExecutionSensitivityResult",
    "ExecutionSensitivityScenario",
    "LimitFillAssumption",
    "default_execution_sensitivity_scenarios",
    "evaluate_execution_sensitivity",
]
