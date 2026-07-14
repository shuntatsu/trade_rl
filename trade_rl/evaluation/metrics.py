"""Single authoritative implementation of portfolio performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from trade_rl.evaluation.series import ReturnKind, ReturnSeries


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Resolved metrics for one explicitly identified return series."""

    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover_total: float
    total_cost: float
    funding_pnl: float
    borrow_cost: float
    n_trades: int
    rebalance_events: int
    termination_count: int
    n_periods: int
    return_kind: ReturnKind
    periods_per_year: int


def compound_return(values: tuple[float, ...]) -> float:
    """Compound simple returns into a total return."""

    wealth = 1.0
    for value in values:
        wealth *= 1.0 + value
    return wealth - 1.0


def _max_drawdown(values: tuple[float, ...]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum = max(maximum, 1.0 - wealth / peak)
    return maximum


def _require_non_negative(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if value < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return value


def evaluate_performance(
    returns: ReturnSeries,
    *,
    turnover_total: float = 0.0,
    total_cost: float = 0.0,
    funding_pnl: float = 0.0,
    borrow_cost: float = 0.0,
    n_trades: int = 0,
    rebalance_events: int = 0,
    termination_count: int = 0,
) -> PerformanceMetrics:
    """Compute all standard portfolio metrics from one return-series contract."""

    turnover = _require_non_negative(turnover_total, field="turnover_total")
    cost = _require_non_negative(total_cost, field="total_cost")
    borrow = _require_non_negative(borrow_cost, field="borrow_cost")
    if not math.isfinite(funding_pnl):
        raise ValueError("funding_pnl must be finite")
    for field_name, value in (
        ("n_trades", n_trades),
        ("rebalance_events", rebalance_events),
        ("termination_count", termination_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")

    values = returns.values
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    standard_deviation = math.sqrt(variance)
    annualization = math.sqrt(returns.annualization_periods_per_year)
    sharpe = (
        mean / standard_deviation * annualization if standard_deviation > 0 else 0.0
    )

    downside_rms = math.sqrt(fmean(min(value, 0.0) ** 2 for value in values))
    sortino = mean / downside_rms * annualization if downside_rms > 0 else 0.0

    return PerformanceMetrics(
        total_return=compound_return(values),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_max_drawdown(values),
        turnover_total=turnover,
        total_cost=cost,
        funding_pnl=funding_pnl,
        borrow_cost=borrow,
        n_trades=n_trades,
        rebalance_events=rebalance_events,
        termination_count=termination_count,
        n_periods=len(values),
        return_kind=returns.kind,
        periods_per_year=int(round(returns.annualization_periods_per_year)),
    )
