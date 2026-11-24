"""Rollout result and net-return metrics shared by every learning-lab method.

A method (baseline or learned agent) is reduced to the per-native-bar net returns of the
shared-portfolio engine and the engine's final book facts; nothing here prices a trade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """One episode of one method over one window, as the engine reported it."""

    method: str
    bar_returns: tuple[float, ...]
    final_equity: float
    total_cost: float
    turnover_total: float
    initial_capital: float

    def metrics(self, *, periods_per_year: int) -> dict[str, Any]:
        return summarize_returns(
            self.bar_returns,
            periods_per_year=periods_per_year,
            final_equity=self.final_equity,
            initial_capital=self.initial_capital,
            total_cost=self.total_cost,
            turnover_total=self.turnover_total,
        )


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    """``(sharpe_per_period, skewness, kurtosis)``; zeros when the variance is zero."""
    if returns.size < 3:
        return 0.0, 0.0, 3.0
    std = float(returns.std(ddof=1))
    if std <= 1e-18:
        return 0.0, 0.0, 3.0
    centred = (returns - returns.mean()) / float(returns.std(ddof=0))
    return (
        float(returns.mean()) / std,
        float(np.mean(centred**3)),
        float(np.mean(centred**4)),
    )


def summarize_returns(
    returns: Sequence[float],
    *,
    periods_per_year: int,
    final_equity: float | None = None,
    initial_capital: float | None = None,
    total_cost: float | None = None,
    turnover_total: float | None = None,
) -> dict[str, Any]:
    """Per-period and annual net Sharpe, total return, max drawdown and moments."""
    series = np.asarray(returns, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not bool(np.isfinite(series).all()):
        raise ValueError("returns must be finite")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    sharpe, skew, kurt = _moments(series)
    curve = np.cumprod(1.0 + series) if series.size else np.ones(1)
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))
    drawdown = 1.0 - np.concatenate([[1.0], curve]) / peak
    out: dict[str, Any] = {
        "n_obs": int(series.size),
        "sharpe_per_period": sharpe,
        "sharpe_annual": sharpe * math.sqrt(periods_per_year),
        "skewness": skew,
        "kurtosis": kurt,
        "total_return": float(curve[-1] - 1.0),
        "max_drawdown": float(drawdown.max()),
    }
    if final_equity is not None and initial_capital is not None:
        out["equity_return"] = final_equity / initial_capital - 1.0
    if total_cost is not None:
        out["total_cost"] = float(total_cost)
    if turnover_total is not None:
        out["turnover_total"] = float(turnover_total)
    return out


__all__ = ["RolloutResult", "summarize_returns"]
