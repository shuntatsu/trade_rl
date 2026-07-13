"""Deterministic base-bar execution, costs, funding, and weight drift."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState


@dataclass(frozen=True, slots=True)
class ExecutionCostConfig:
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        for field, value in (
            ("fee_rate", self.fee_rate),
            ("spread_rate", self.spread_rate),
            ("impact_rate", self.impact_rate),
            ("multiplier", self.multiplier),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")

    @property
    def rate_per_turnover(self) -> float:
        return self.multiplier * (self.fee_rate + self.spread_rate + self.impact_rate)

    @classmethod
    def zero(cls) -> ExecutionCostConfig:
        return cls(fee_rate=0.0, spread_rate=0.0, impact_rate=0.0)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    book: BookState
    next_index: int
    bars_advanced: int
    interval_gross_return: float
    interval_cost: float
    interval_funding: float
    interval_net_return: float
    interval_log_return: float


def _target_weights(value: np.ndarray, *, n_symbols: int) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if target.shape != (n_symbols,):
        raise ValueError("target weights shape does not match market symbols")
    if not np.isfinite(target).all():
        raise ValueError("target weights must be finite")
    gross = float(np.abs(target).sum())
    if gross > 1.0 + 1e-12:
        raise ValueError("target gross exposure cannot exceed one")
    return target


def _drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross_return = float(np.dot(weights, asset_returns))
    denominator = 1.0 + gross_return
    if denominator <= 0.0:
        return np.zeros_like(weights)
    drifted = weights * (1.0 + asset_returns) / denominator
    gross = float(np.abs(drifted).sum())
    if gross > 1.0:
        drifted /= gross
    return drifted


class MarketExecutor:
    """Execute one decision target over a contiguous base-bar interval."""

    def __init__(
        self,
        dataset: MarketDataset,
        cost: ExecutionCostConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()

    def execute_interval(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        if bars <= 0:
            raise ValueError("bars must be positive")
        if start_index < 0 or start_index + bars >= self.dataset.n_bars:
            raise ValueError("execution interval is outside the dataset")
        if book.weights.shape != (self.dataset.n_symbols,):
            raise ValueError("book weights shape does not match market symbols")

        resolved_target = _target_weights(target, n_symbols=self.dataset.n_symbols)
        result_book = book.clone()
        starting_value = result_book.portfolio_value
        turnover = float(np.abs(resolved_target - result_book.weights).sum())
        cost_fraction = turnover * self.cost.rate_per_turnover
        if cost_fraction >= 1.0:
            raise ValueError("execution cost would exhaust the portfolio")
        cost_amount = starting_value * cost_fraction
        result_book.weights = resolved_target.copy()
        result_book.record_rebalance(turnover=turnover, cost_amount=cost_amount)

        gross_factor = 1.0
        funding_return_total = 0.0
        for offset in range(bars):
            index = start_index + offset
            asset_returns = (
                self.dataset.close[index + 1] / self.dataset.close[index] - 1.0
            )
            weights = result_book.weights
            gross_return = float(np.dot(weights, asset_returns))
            funding_return = -float(np.dot(weights, self.dataset.funding_rate[index]))
            applied_cost = cost_fraction if offset == 0 else 0.0
            net_return = gross_return + funding_return - applied_cost
            if not math.isfinite(net_return) or net_return <= -1.0:
                raise ValueError("execution produced an invalid base-bar return")
            value_before = result_book.portfolio_value
            next_weights = _drift_weights(weights, asset_returns)
            result_book.record_base_bar(
                net_return=net_return,
                next_weights=next_weights,
                funding_amount=value_before * funding_return,
            )
            gross_factor *= 1.0 + gross_return
            funding_return_total += funding_return

        interval_net_return = result_book.portfolio_value / starting_value - 1.0
        return ExecutionResult(
            book=result_book,
            next_index=start_index + bars,
            bars_advanced=bars,
            interval_gross_return=gross_factor - 1.0,
            interval_cost=cost_fraction,
            interval_funding=funding_return_total,
            interval_net_return=interval_net_return,
            interval_log_return=math.log1p(interval_net_return),
        )
