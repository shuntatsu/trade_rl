from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def infer_bars_per_year(fs: object) -> int:
    timestamps = np.asarray(getattr(fs, "timestamps"))
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("at least two timestamps are required")
    ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    deltas = np.diff(ns)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        raise ValueError("timestamps must be strictly increasing")
    median_ns = float(np.median(positive))
    year_ns = 365.0 * 24.0 * 60.0 * 60.0 * 1_000_000_000.0
    bars = int(round(year_ns / median_ns))
    if bars <= 0:
        raise ValueError("inferred bars_per_year must be positive")
    return bars


def sharpe_from_equity(equity_curve: object, *, bars_per_year: int) -> float:
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.ndim != 1 or equity.size < 2:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    std = float(returns.std())
    return (
        float(returns.mean() / std * np.sqrt(bars_per_year)) if std > 0.0 else 0.0
    )


def reannualize_strategy_result(result: Any, *, bars_per_year: int) -> Any:
    return replace(
        result,
        sharpe=sharpe_from_equity(
            getattr(result, "equity_curve"),
            bars_per_year=bars_per_year,
        ),
    )


def reannualize_strategy_results(
    results: dict[str, Any],
    *,
    bars_per_year: int,
) -> dict[str, Any]:
    return {
        name: reannualize_strategy_result(value, bars_per_year=bars_per_year)
        for name, value in results.items()
    }
