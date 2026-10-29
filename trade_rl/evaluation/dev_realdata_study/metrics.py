"""Equity-path metrics, paired block bootstrap and the result-blind selection rule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.evaluation.dev_realdata_study.protocol import (
    LABEL_NO_WINNER,
    LABEL_PROMISING,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    DevStudyProtocol,
)

_NS_PER_HOUR = 3_600 * 1_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR


def _equity_path(equity: np.ndarray) -> np.ndarray:
    path = np.asarray(equity, dtype=np.float64).reshape(-1)
    if path.size < 2 or not np.all(np.isfinite(path)) or np.any(path <= 0.0):
        raise ValueError("equity path must hold at least two positive finite values")
    return path


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline as a positive fraction of the running peak."""

    path = _equity_path(equity)
    peaks = np.maximum.accumulate(path)
    return float(np.max(1.0 - path / peaks))


def daily_equity(
    timestamps_ns: np.ndarray, equity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Equity sampled at 00:00 UTC rows, with the matching timestamps."""

    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    path = _equity_path(equity)
    if ts.shape != path.shape:
        raise ValueError("timestamps and equity must have equal length")
    mask = ts % _NS_PER_DAY == 0
    if int(mask.sum()) < 3:
        raise ValueError("equity path needs at least three 00:00 UTC samples")
    return ts[mask], path[mask]


def daily_returns(daily: np.ndarray) -> np.ndarray:
    values = np.asarray(daily, dtype=np.float64)
    return values[1:] / values[:-1] - 1.0


def _year_of(timestamp_ns: int) -> int:
    return int(
        np.datetime64(timestamp_ns, "ns").astype("datetime64[Y]").astype(int)
    ) + (1970)


def per_year_returns(timestamps_ns: np.ndarray, equity: np.ndarray) -> dict[str, float]:
    """Net return per calendar-year segment (the last year may be partial)."""

    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    path = _equity_path(equity)
    result: dict[str, float] = {}
    first_year = _year_of(int(ts[0]))
    last_year = _year_of(int(ts[-1]))
    for year in range(first_year, last_year + 1):
        year_start = int(np.datetime64(f"{year}-01-01", "ns").astype(np.int64))
        year_end = int(np.datetime64(f"{year + 1}-01-01", "ns").astype(np.int64))
        begin = int(np.searchsorted(ts, max(year_start, int(ts[0])), side="left"))
        end = int(np.searchsorted(ts, min(year_end, int(ts[-1])), side="right")) - 1
        if end <= begin:
            continue
        result[str(year)] = float(path[end] / path[begin] - 1.0)
    return result


def performance_summary(
    timestamps_ns: np.ndarray,
    equity: np.ndarray,
    protocol: DevStudyProtocol,
) -> dict[str, Any]:
    """Terminal profit, CAGR, drawdown, daily Sharpe and per-year returns."""

    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    path = _equity_path(equity)
    terminal = float(path[-1] / path[0] - 1.0)
    years = float(ts[-1] - ts[0]) / _NS_PER_DAY / protocol.cagr_days_per_year
    cagr = float((path[-1] / path[0]) ** (1.0 / years) - 1.0)
    _, daily = daily_equity(ts, path)
    returns = daily_returns(daily)
    std = float(np.std(returns, ddof=1))
    sharpe: float | None
    if std > 0.0 and math.isfinite(std):
        sharpe = float(
            np.mean(returns) / std * math.sqrt(protocol.trading_days_per_year)
        )
    else:
        sharpe = None
    return {
        "terminal_net_profit_pct": 100.0 * terminal,
        "cagr_pct": 100.0 * cagr,
        "max_drawdown_pct": 100.0 * max_drawdown(path),
        "sharpe_daily": sharpe,
        "n_daily_returns": int(returns.size),
        "years": years,
        "per_year_return_pct": {
            year: 100.0 * value for year, value in per_year_returns(ts, path).items()
        },
    }


def moving_block_indices(
    n_values: int, *, block: int, resamples: int, seed: int
) -> np.ndarray:
    """Overlapping-block resample indices, shape ``(resamples, n_values)``."""

    if n_values < 2:
        raise ValueError("bootstrap needs at least two observations")
    block = min(block, n_values)
    n_blocks = -(-n_values // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_values - block + 1, size=(resamples, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1)
    return indices[:, :n_values]


@dataclass(frozen=True)
class BootstrapInterval:
    mean_daily: float
    annualized_mean: float
    ci_low: float
    ci_high: float
    bonferroni_ci_low: float
    bonferroni_ci_high: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "mean_daily_excess": self.mean_daily,
            "annualized_mean_excess": self.annualized_mean,
            "ci95_low_annualized": self.ci_low,
            "ci95_high_annualized": self.ci_high,
            "bonferroni_ci_low_annualized": self.bonferroni_ci_low,
            "bonferroni_ci_high_annualized": self.bonferroni_ci_high,
        }


def bootstrap_excess(
    excess_daily: np.ndarray,
    indices: np.ndarray,
    protocol: DevStudyProtocol,
) -> BootstrapInterval:
    """Percentile CI of the mean daily excess return, annualised by simple scaling."""

    excess = np.asarray(excess_daily, dtype=np.float64).reshape(-1)
    if indices.shape[1] != excess.size:
        raise ValueError("bootstrap indices must match the excess series length")
    scale = float(protocol.trading_days_per_year)
    means = excess[indices].mean(axis=1) * scale
    alpha = protocol.bootstrap_alpha
    adjusted = alpha / protocol.bonferroni_comparisons
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    b_low, b_high = np.quantile(means, [adjusted / 2.0, 1.0 - adjusted / 2.0])
    mean_daily = float(excess.mean())
    return BootstrapInterval(
        mean_daily=mean_daily,
        annualized_mean=mean_daily * scale,
        ci_low=float(low),
        ci_high=float(high),
        bonferroni_ci_low=float(b_low),
        bonferroni_ci_high=float(b_high),
    )


@dataclass(frozen=True)
class SelectionOutcome:
    label: str
    selected_arm: str | None
    rows: dict[str, dict[str, object]]

    def to_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "selected_arm": self.selected_arm,
            "arms": self.rows,
        }


def select_development_label(
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
    validity: Mapping[str, bool],
    protocol: DevStudyProtocol,
) -> SelectionOutcome:
    """Apply the preregistered rule; it can only ever return PROMISING or NO_WINNER.

    ``results[arm][scenario]`` holds ``terminal_net_profit_pct`` and
    ``max_drawdown_pct``. Order: validity -> risk gate (max drawdown at or under
    the limit in every scenario) -> beats CASH (terminal net profit above the
    zero CASH profit) under base and cost x2 -> highest base terminal profit.
    """

    limit = 100.0 * protocol.max_drawdown_limit
    rows: dict[str, dict[str, object]] = {}
    passing: list[str] = []
    for arm in protocol.candidate_arms:
        scenarios = results[arm]
        valid = bool(validity.get(arm, False))
        worst_dd = max(
            float(scenarios[name]["max_drawdown_pct"]) for name in protocol.scenarios
        )
        risk_ok = worst_dd <= limit
        beats_cash = all(
            float(scenarios[name]["terminal_net_profit_pct"]) > 0.0
            for name in (SCENARIO_BASE, SCENARIO_COST_X2)
        )
        eligible = valid and risk_ok and beats_cash
        rows[arm] = {
            "valid": valid,
            "worst_max_drawdown_pct": worst_dd,
            "risk_gate_passed": risk_ok,
            "beats_cash_base_and_cost_x2": beats_cash,
            "eligible": eligible,
            "base_terminal_net_profit_pct": float(
                scenarios[SCENARIO_BASE]["terminal_net_profit_pct"]
            ),
        }
        if eligible:
            passing.append(arm)
    if not passing:
        return SelectionOutcome(LABEL_NO_WINNER, None, rows)
    best = max(
        passing,
        key=lambda arm: float(results[arm][SCENARIO_BASE]["terminal_net_profit_pct"]),
    )
    return SelectionOutcome(LABEL_PROMISING, best, rows)


__all__ = [
    "BootstrapInterval",
    "SelectionOutcome",
    "bootstrap_excess",
    "daily_equity",
    "daily_returns",
    "max_drawdown",
    "moving_block_indices",
    "per_year_returns",
    "performance_summary",
    "select_development_label",
]
