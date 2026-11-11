"""Cell metrics, the stop-price haircut and the preregistered gate for the trend study."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.evaluation.dev_realdata_study.metrics import (
    daily_equity,
    daily_returns,
    max_drawdown,
    moving_block_indices,
    per_year_returns,
)
from trade_rl.evaluation.replay import SharedCashLedgerIntervalEvidence
from trade_rl.evaluation.robustness.closed_trades import (
    ClosedTradeDiagnostics,
    ClosedTradeTracker,
)
from trade_rl.evaluation.trend_stop_study.protocol import (
    LABEL_NO_WINNER,
    LABEL_PROMISING,
    TrendStopProtocol,
)
from trade_rl.strategies.rules.trend_stop import StopExitEvent

_NS_PER_DAY = 86_400 * 1_000_000_000
_TOLERANCE = 1e-12


def performance_summary(
    timestamps_ns: np.ndarray, equity: np.ndarray, protocol: TrendStopProtocol
) -> dict[str, Any]:
    """Terminal return, CAGR, Calmar, drawdown, daily Sharpe and per-year returns."""

    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    path = np.asarray(equity, dtype=np.float64).reshape(-1)
    terminal = float(path[-1] / path[0] - 1.0)
    years = float(ts[-1] - ts[0]) / _NS_PER_DAY / protocol.cagr_days_per_year
    cagr = float((path[-1] / path[0]) ** (1.0 / years) - 1.0)
    drawdown = max_drawdown(path)
    _, daily = daily_equity(ts, path)
    returns = daily_returns(daily)
    std = float(np.std(returns, ddof=1))
    sharpe = (
        float(np.mean(returns) / std * math.sqrt(protocol.trading_days_per_year))
        if std > 0.0 and math.isfinite(std)
        else None
    )
    return {
        "terminal_net_return_pct": 100.0 * terminal,
        "cagr_pct": 100.0 * cagr,
        "max_drawdown_pct": 100.0 * drawdown,
        "calmar": (cagr / drawdown) if drawdown > 0.0 else None,
        "sharpe_daily": sharpe,
        "years": years,
        "per_year_return_pct": {
            year: 100.0 * value for year, value in per_year_returns(ts, path).items()
        },
    }


def daily_return_series(timestamps_ns: np.ndarray, equity: np.ndarray) -> np.ndarray:
    _, daily = daily_equity(timestamps_ns, equity)
    return daily_returns(daily)


def bootstrap_lower_bound(
    daily: np.ndarray, protocol: TrendStopProtocol
) -> dict[str, float]:
    """Bonferroni-adjusted lower percentile of the annualised mean daily net return."""

    values = np.asarray(daily, dtype=np.float64).reshape(-1)
    indices = moving_block_indices(
        values.size,
        block=protocol.bootstrap_block_days,
        resamples=protocol.bootstrap_resamples,
        seed=protocol.bootstrap_seed,
    )
    scale = float(protocol.trading_days_per_year)
    means = values[indices].mean(axis=1) * scale
    adjusted = protocol.bootstrap_alpha / protocol.n_configurations
    low, high = np.quantile(means, [adjusted / 2.0, 1.0 - adjusted / 2.0])
    return {
        "annualized_mean_daily_return": float(values.mean() * scale),
        "bonferroni_ci_low": float(low),
        "bonferroni_ci_high": float(high),
        "adjusted_alpha": float(adjusted),
    }


# -- closed trades --------------------------------------------------------------------


def closed_trades_from_ledger(
    intervals: Sequence[SharedCashLedgerIntervalEvidence],
    multipliers: np.ndarray,
    symbols: Sequence[str],
    extra_costs: Mapping[tuple[int, int], float] | None = None,
) -> tuple[ClosedTradeDiagnostics, dict[str, ClosedTradeDiagnostics]]:
    """Reconstruct closed position cycles per symbol from the raw order events.

    The interval execution cost is split across that interval's fills by notional.
    ``extra_costs`` maps ``(interval position, event position)`` to an additional cost
    (the stop haircut) charged on that exit fill, so it lowers that cycle's realized
    PnL. Funding is not attributed to cycles (it stays portfolio-level).
    """

    trackers = {index: ClosedTradeTracker(multipliers) for index in range(len(symbols))}
    extra = extra_costs or {}
    for position, interval in enumerate(intervals):
        fills = [
            (index, event)
            for index, event in enumerate(interval.order_events)
            if abs(float(event.filled_quantity)) > _TOLERANCE
        ]
        total_notional = sum(abs(float(event.filled_notional)) for _, event in fills)
        for index, event in fills:
            if event.execution_price is None:
                raise ValueError("filled order event is missing its execution price")
            cost = (
                0.0
                if total_notional <= _TOLERANCE
                else interval.interval_cost
                * abs(float(event.filled_notional))
                / total_notional
            ) + extra.get((position, index), 0.0)
            trackers[int(event.symbol_index)].record_fill(
                symbol=int(event.symbol_index),
                quantity=float(event.filled_quantity),
                price=float(event.execution_price),
                execution_cost=max(0.0, float(cost)),
            )
    per_symbol = {
        symbols[index]: trackers[index].diagnostics() for index in sorted(trackers)
    }
    combined = ClosedTradeDiagnostics.combine(tuple(per_symbol.values()))
    return combined, per_symbol


# -- stop haircut ---------------------------------------------------------------------


@dataclass
class _ActiveStop:
    event: StopExitEvent
    closed_quantity: float = 0.0
    closed_notional: float = 0.0
    improvement: float = 0.0


def stop_exit_audit(
    intervals: Sequence[SharedCashLedgerIntervalEvidence],
    stop_events: Sequence[StopExitEvent],
    multipliers: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    """Compare every stop exit's fills with its stop level.

    Returns the aggregate audit and the haircut of each closing fill keyed by
    ``(interval position, event position)``; the second value is what the runner
    charges against the PnL stream.

    ``haircut_usdt`` is the price improvement over the stop level (a resting stop would
    have filled at the level or worse), summed over the closing quantity. Gaps through
    the level are never credited back. ``mean_adverse_gap_pct`` is positive when the
    average fill was worse than the stop level.
    """

    by_symbol: dict[int, list[StopExitEvent]] = defaultdict(list)
    for event in stop_events:
        by_symbol[event.symbol_index].append(event)
    positions: dict[int, float] = defaultdict(float)
    active: dict[int, _ActiveStop] = {}
    resolved: list[tuple[StopExitEvent, float, float, float]] = []
    by_fill: dict[tuple[int, int], float] = {}
    for position_index, interval in enumerate(intervals):
        for symbol, events in by_symbol.items():
            while events and events[0].decision_index <= interval.start_index:
                event = events.pop(0)
                active[symbol] = _ActiveStop(event)
        for fill_index, fill in enumerate(interval.order_events):
            quantity = float(fill.filled_quantity)
            if abs(quantity) <= _TOLERANCE:
                continue
            symbol = int(fill.symbol_index)
            position = positions[symbol]
            stop = active.get(symbol)
            if (
                stop is not None
                and abs(position) > _TOLERANCE
                and (quantity > 0.0) != (position > 0.0)
            ):
                closing = min(abs(position), abs(quantity))
                price = float(fill.execution_price or 0.0)
                stop.closed_quantity += closing
                stop.closed_notional += closing * price
                improvement = (
                    max(0.0, stop.event.side * (price - stop.event.stop_level))
                    * closing
                    * float(multipliers[symbol])
                )
                stop.improvement += improvement
                if improvement > 0.0:
                    key = (position_index, fill_index)
                    by_fill[key] = by_fill.get(key, 0.0) + improvement
            positions[symbol] = position + quantity
            if stop is not None and (
                abs(positions[symbol]) <= _TOLERANCE
                or (positions[symbol] > 0.0) != (position > 0.0)
            ):
                if stop.closed_quantity > 0.0:
                    average = stop.closed_notional / stop.closed_quantity
                    resolved.append(
                        (stop.event, average, stop.closed_quantity, stop.improvement)
                    )
                del active[symbol]
    gaps = [
        100.0 * event.side * (event.stop_level - average) / event.stop_level
        for event, average, _, _ in resolved
    ]
    audit = {
        "n_stop_exits": len(stop_events),
        "n_resolved": len(resolved),
        "n_unresolved": len(stop_events) - len(resolved),
        "n_fill_better_than_stop": sum(
            1
            for event, average, _, _ in resolved
            if event.side * (average - event.stop_level) > 0.0
        ),
        "n_trailing_stops": sum(1 for event in stop_events if event.trailing),
        "mean_adverse_gap_pct": float(np.mean(gaps)) if gaps else None,
        "median_adverse_gap_pct": float(np.median(gaps)) if gaps else None,
        "haircut_usdt": float(sum(by_fill.values())),
    }
    return audit, by_fill


def haircut_adjusted_equity(
    equity: np.ndarray,
    n_intervals: int,
    haircuts_by_fill: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Charge each exit's haircut as a cost on the interval that realizes it.

    ``equity[0]`` is the initial value and ``equity[k]`` the value after interval
    ``k - 1``. The haircut of an interval is taken as a fraction of the engine equity
    at the start of that interval and applied to the adjusted path multiplicatively, so
    it compounds through every later interval (positions are sized in proportion to
    equity). It is a first-order adjustment, not a re-simulation with smaller positions.
    """

    path = np.asarray(equity, dtype=np.float64).reshape(-1)
    if path.size != n_intervals + 1:
        raise ValueError("equity path must hold one value per interval plus the start")
    per_interval = np.zeros(n_intervals, dtype=np.float64)
    for (position, _), amount in haircuts_by_fill.items():
        per_interval[position] += amount
    adjusted = np.empty_like(path)
    adjusted[0] = path[0]
    for step in range(n_intervals):
        engine_return = path[step + 1] / path[step] - 1.0
        cost_fraction = per_interval[step] / path[step]
        adjusted[step + 1] = adjusted[step] * (1.0 + engine_return - cost_fraction)
    return adjusted


# -- gate -----------------------------------------------------------------------------


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def evaluate_configuration(
    base: Mapping[str, Any],
    cost_x2: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    bootstrap: Mapping[str, float],
    validity: bool,
    protocol: TrendStopProtocol,
) -> dict[str, Any]:
    """Apply the preregistered pass / fail rule to one (variant, timeframe) pair.

    Every cell's ``performance``, ``closed_trades`` and the bootstrap input are the
    stop-haircut-adjusted quantities (the engine values are kept beside them as
    ``engine_*``), so no criterion can pass on stop-fill optimism. Every criterion is
    reported with its observed value so a failure is attributable; a missing value
    fails closed.
    """

    per_year = _get(base, "performance", "per_year_return_pct")
    positive_years = sum(
        1 for year in protocol.sub_years if per_year.get(year, 0.0) > 0
    )
    calmar = _get(base, "performance", "calmar")
    benchmark_calmar = _get(benchmark, "performance", "calmar")
    profit_factor = _get(base, "closed_trades", "profit_factor")
    worst_drawdown = max(
        _get(base, "performance", "max_drawdown_pct"),
        _get(cost_x2, "performance", "max_drawdown_pct"),
    )
    net_base = _get(base, "performance", "terminal_net_return_pct")
    net_x2 = _get(cost_x2, "performance", "terminal_net_return_pct")
    closed = _get(base, "closed_trades", "closed_trades")
    criteria: dict[str, dict[str, Any]] = {
        "net_return_positive_base": {"value": net_base, "passed": net_base > 0.0},
        "net_return_positive_cost_x2": {"value": net_x2, "passed": net_x2 > 0.0},
        "positive_sub_years_base": {
            "value": positive_years,
            "required": protocol.min_positive_sub_years,
            "passed": positive_years >= protocol.min_positive_sub_years,
        },
        "profit_factor_base": {
            "value": profit_factor,
            "required": protocol.min_profit_factor,
            "passed": profit_factor is not None
            and profit_factor >= protocol.min_profit_factor,
        },
        "max_drawdown_within_limit_all_scenarios": {
            "value": worst_drawdown,
            "required": 100.0 * protocol.max_drawdown_limit,
            "passed": worst_drawdown <= 100.0 * protocol.max_drawdown_limit,
        },
        "closed_trades_base": {
            "value": closed,
            "required": protocol.min_closed_trades,
            "passed": closed >= protocol.min_closed_trades,
        },
        # Fail closed: a missing strategy or benchmark Calmar never passes.
        "calmar_exceeds_btc_buy_and_hold": {
            "value": calmar,
            "benchmark": benchmark_calmar,
            "passed": calmar is not None
            and benchmark_calmar is not None
            and calmar > benchmark_calmar,
        },
        "bonferroni_bootstrap_lower_bound_positive": {
            "value": bootstrap["bonferroni_ci_low"],
            "passed": bootstrap["bonferroni_ci_low"] > 0.0,
        },
        "validity": {"value": bool(validity), "passed": bool(validity)},
    }
    return {
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria.values()),
        "bootstrap": dict(bootstrap),
    }


def select_label(configurations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``DEVELOPMENT_PROMISING`` iff at least one configuration passes; never a winner."""

    passing = sorted(name for name, item in configurations.items() if item["passed"])
    return {
        "label": LABEL_PROMISING if passing else LABEL_NO_WINNER,
        "passing_configurations": passing,
        "stop_rule_triggered": not passing,
    }


__all__ = [
    "LABEL_NO_WINNER",
    "LABEL_PROMISING",
    "bootstrap_lower_bound",
    "closed_trades_from_ledger",
    "daily_return_series",
    "evaluate_configuration",
    "haircut_adjusted_equity",
    "performance_summary",
    "select_label",
    "stop_exit_audit",
]
