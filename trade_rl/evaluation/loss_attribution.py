"""Read-only attribution of persisted candidate-run losses.

This module never executes a strategy. It decomposes already-persisted realized-path
accounting and raw return evidence. ``residual_pnl`` is only the additive balancing
term after removing persisted explicit execution cost, funding, and borrow from net
PnL; it is not a counterfactual zero-cost backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

_SCHEMA_VERSION = "deterministic_loss_attribution_v1"
_TOLERANCE = 1e-12


def _finite_float(value: object, *, field: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if non_negative and resolved < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return resolved


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _compound(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("raw returns must be one-dimensional and finite")
    return float(np.prod(1.0 + data, dtype=np.float64) - 1.0)


def _locate_timestamp(dataset: MarketDataset, value: object, *, field: str) -> int:
    text = _non_empty_string(value, field=field)
    try:
        timestamp = np.datetime64(text, "ns")
    except ValueError as error:
        raise ValueError(f"{field} must be a datetime") from error
    if np.isnat(timestamp):
        raise ValueError(f"{field} must not be NaT")
    values = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    matches = np.flatnonzero(values == timestamp)
    if matches.size != 1:
        raise ValueError(f"{field} must identify exactly one dataset timestamp")
    return int(matches[0])


def _strategy_entry(
    run: LoadedCandidateRun,
    *,
    symbol_index: int,
    symbol: str,
    strategy: str,
) -> dict[str, object]:
    by_symbol = _array(run.summary.get("by_symbol"), field="candidate by_symbol")
    matches: list[dict[str, object]] = []
    for raw_symbol in by_symbol:
        item = _object(raw_symbol, field="candidate symbol entry")
        if item.get("symbol_index") != symbol_index or item.get("symbol") != symbol:
            continue
        strategies = _array(item.get("strategies"), field="candidate strategies")
        for raw_strategy in strategies:
            entry = _object(raw_strategy, field="candidate strategy entry")
            if entry.get("name") == strategy:
                matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            f"candidate strategy roster mismatch for {symbol}/{strategy}: {len(matches)}"
        )
    return matches[0]


def _metric_diagnostic_values(entry: dict[str, object]) -> tuple[dict[str, float | int], dict[str, float | int]]:
    metrics = _object(entry.get("metrics"), field="candidate metrics")
    diagnostics = _object(entry.get("diagnostics"), field="candidate diagnostics")
    metric_values: dict[str, float | int] = {
        "total_return": _finite_float(metrics.get("total_return"), field="total return"),
        "turnover_total": _finite_float(
            metrics.get("turnover_total"), field="turnover_total", non_negative=True
        ),
        "total_cost": _finite_float(
            metrics.get("total_cost"), field="total_cost", non_negative=True
        ),
        "funding_pnl": _finite_float(metrics.get("funding_pnl"), field="funding_pnl"),
        "borrow_cost": _finite_float(
            metrics.get("borrow_cost"), field="borrow_cost", non_negative=True
        ),
        "n_trades": _non_negative_int(metrics.get("n_trades"), field="n_trades"),
        "rebalance_events": _non_negative_int(
            metrics.get("rebalance_events"), field="rebalance_events"
        ),
        "n_periods": _non_negative_int(metrics.get("n_periods"), field="n_periods"),
    }
    diagnostic_values: dict[str, float | int] = {
        "turnover_total": _finite_float(
            diagnostics.get("turnover_total"),
            field="diagnostic turnover_total",
            non_negative=True,
        ),
        "total_cost": _finite_float(
            diagnostics.get("total_cost"),
            field="diagnostic total_cost",
            non_negative=True,
        ),
        "funding_pnl": _finite_float(
            diagnostics.get("funding_pnl"), field="diagnostic funding_pnl"
        ),
        "borrow_cost": _finite_float(
            diagnostics.get("borrow_cost"),
            field="diagnostic borrow_cost",
            non_negative=True,
        ),
        "n_trades": _non_negative_int(
            diagnostics.get("n_trades"), field="diagnostic n_trades"
        ),
        "rebalance_events": _non_negative_int(
            diagnostics.get("rebalance_events"), field="diagnostic rebalance_events"
        ),
    }
    for key, value in diagnostic_values.items():
        if metric_values[key] != value:
            raise ValueError(f"candidate metric/diagnostic mismatch for {key}")
    return metric_values, diagnostic_values


def _raw_returns(entry: dict[str, object], run: LoadedCandidateRun) -> np.ndarray:
    key = _non_empty_string(entry.get("return_key"), field="return_key")
    try:
        raw = run.returns[key]
    except KeyError as error:
        raise ValueError("candidate return_key is missing from raw returns") from error
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("candidate raw returns must be one-dimensional and finite")
    return values


def _period_returns(
    returns: np.ndarray,
    owners: np.ndarray,
    *,
    unit: str,
) -> tuple[tuple[int, float], ...]:
    periods = np.asarray(owners).astype(f"datetime64[{unit}]")
    unique = np.unique(periods)
    result: list[tuple[int, float]] = []
    offset = 1970 if unit == "Y" else 0
    for period in unique:
        mask = periods == period
        label = int(period.astype(np.int64)) + offset
        result.append((label, _compound(returns[mask])))
    return tuple(result)


def _month_positive_count(returns: np.ndarray, owners: np.ndarray) -> tuple[int, int]:
    periods = np.asarray(owners).astype("datetime64[M]")
    unique = np.unique(periods)
    positives = sum(_compound(returns[periods == period]) > 0.0 for period in unique)
    return int(positives), int(unique.size)


def _market_sensitivity(
    returns: np.ndarray,
    mark_returns: np.ndarray,
) -> tuple[float | None, float | None]:
    strategy = np.asarray(returns, dtype=np.float64)
    market = np.asarray(mark_returns, dtype=np.float64)
    if strategy.shape != market.shape or strategy.ndim != 1:
        raise ValueError("strategy and mark returns must have identical one-dimensional shape")
    if not np.isfinite(strategy).all() or not np.isfinite(market).all():
        raise ValueError("strategy and mark returns must be finite")
    strategy_centered = strategy - float(np.mean(strategy))
    market_centered = market - float(np.mean(market))
    strategy_ss = float(np.dot(strategy_centered, strategy_centered))
    market_ss = float(np.dot(market_centered, market_centered))
    if strategy_ss <= _TOLERANCE or market_ss <= _TOLERANCE:
        return None, None
    cross = float(np.dot(strategy_centered, market_centered))
    correlation = cross / math.sqrt(strategy_ss * market_ss)
    beta = cross / market_ss
    return float(correlation), float(beta)


@dataclass(frozen=True, slots=True)
class LossAttributionCell:
    """One deterministic strategy/symbol realized-path attribution cell."""

    symbol: str
    strategy: str
    net_total_return: float
    reconstructed_total_return: float
    initial_capital: float
    net_pnl: float
    explicit_execution_cost: float
    funding_pnl: float
    borrow_cost: float
    residual_pnl: float
    turnover_total: float
    n_trades: int
    rebalance_events: int
    explicit_cost_flip_on_realized_path: bool
    year_returns: tuple[tuple[int, float], ...]
    positive_months: int
    month_count: int
    underlying_mark_correlation: float | None
    underlying_mark_beta: float | None

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "net_total_return": self.net_total_return,
            "reconstructed_total_return": self.reconstructed_total_return,
            "initial_capital": self.initial_capital,
            "net_pnl": self.net_pnl,
            "explicit_execution_cost": self.explicit_execution_cost,
            "funding_pnl": self.funding_pnl,
            "borrow_cost": self.borrow_cost,
            "residual_pnl": self.residual_pnl,
            "residual_semantics": "realized_path_balancing_term_not_zero_cost_backtest",
            "turnover_total": self.turnover_total,
            "n_trades": self.n_trades,
            "rebalance_events": self.rebalance_events,
            "explicit_cost_flip_on_realized_path": self.explicit_cost_flip_on_realized_path,
            "year_returns": [
                {"year": year, "total_return": value}
                for year, value in self.year_returns
            ],
            "positive_months": self.positive_months,
            "month_count": self.month_count,
            "underlying_mark_correlation": self.underlying_mark_correlation,
            "underlying_mark_beta": self.underlying_mark_beta,
        }


@dataclass(frozen=True, slots=True)
class DeterministicLossAttributionReport:
    """Content-addressable read-only loss-attribution report."""

    dataset_id: str
    symbols: tuple[str, ...]
    strategies: tuple[str, ...]
    seed_count: int
    evaluation_start: str
    evaluation_stop_exclusive: str
    cells: tuple[LossAttributionCell, ...]
    schema_version: str = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "strategies": list(self.strategies),
            "seed_count": self.seed_count,
            "evaluation_start": self.evaluation_start,
            "evaluation_stop_exclusive": self.evaluation_stop_exclusive,
            "cells": [cell.to_payload() for cell in self.cells],
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def analyze_deterministic_loss_attribution(
    runs: tuple[LoadedCandidateRun, ...],
    dataset: MarketDataset,
    *,
    deterministic_strategies: tuple[str, ...],
) -> DeterministicLossAttributionReport:
    """Analyze persisted deterministic strategy evidence without replaying it."""

    if not runs:
        raise ValueError("candidate runs must not be empty")
    strategies = tuple(deterministic_strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(
        not item for item in strategies
    ):
        raise ValueError("deterministic_strategies must be unique and non-empty")

    symbols = tuple(dataset.symbols)
    reference_summary = runs[0].summary
    reference_symbols = tuple(
        _non_empty_string(item, field="candidate symbol")
        for item in _array(reference_summary.get("symbols"), field="candidate symbols")
    )
    if reference_symbols != symbols:
        raise ValueError("candidate symbol roster does not match dataset")

    reference_eval = _object(reference_summary.get("evaluation"), field="evaluation")
    start_text = _non_empty_string(reference_eval.get("start"), field="evaluation start")
    stop_text = _non_empty_string(
        reference_eval.get("stop_exclusive"), field="evaluation stop_exclusive"
    )
    initial_capital = _finite_float(
        reference_eval.get("initial_capital"),
        field="initial_capital",
    )
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive")
    start_index = _locate_timestamp(dataset, start_text, field="evaluation start")
    stop_index = _locate_timestamp(dataset, stop_text, field="evaluation stop_exclusive")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("evaluation clock does not define a valid replay interval")
    expected_length = stop_index - start_index
    interval_owners = np.asarray(dataset.timestamps[start_index:stop_index])
    if interval_owners.shape != (expected_length,):
        raise ValueError("evaluation timestamp alignment is invalid")

    mark = np.asarray(dataset.resolved_array("mark_price"), dtype=np.float64)
    if mark.shape != (dataset.n_bars, dataset.n_symbols):
        raise ValueError("dataset mark_price shape is invalid")
    mark_returns = mark[start_index + 1 : stop_index + 1] / mark[
        start_index:stop_index
    ] - 1.0
    if mark_returns.shape != (expected_length, dataset.n_symbols) or not np.isfinite(
        mark_returns
    ).all():
        raise ValueError("dataset mark-return alignment is invalid")

    seen_seeds: set[int] = set()
    for run in runs:
        summary = run.summary
        if summary.get("dataset_id") != dataset.dataset_id:
            raise ValueError("candidate dataset_id does not match dataset")
        run_symbols = tuple(
            _non_empty_string(item, field="candidate symbol")
            for item in _array(summary.get("symbols"), field="candidate symbols")
        )
        if run_symbols != symbols:
            raise ValueError("candidate symbol roster does not match dataset")
        evaluation = _object(summary.get("evaluation"), field="evaluation")
        for key in ("start", "stop_exclusive", "initial_capital", "execution_overlay"):
            if evaluation.get(key) != reference_eval.get(key):
                raise ValueError(f"candidate evaluation field drifted across seeds: {key}")
        config = _object(summary.get("candidate_config"), field="candidate_config")
        seed = _non_negative_int(config.get("ppo_seed"), field="ppo_seed")
        if seed in seen_seeds:
            raise ValueError("candidate ppo_seed values must be unique")
        seen_seeds.add(seed)

    cells: list[LossAttributionCell] = []
    for symbol_index, symbol in enumerate(symbols):
        for strategy in strategies:
            representative_entry: dict[str, object] | None = None
            representative_returns: np.ndarray | None = None
            representative_metrics: dict[str, float | int] | None = None
            for run_index, run in enumerate(runs):
                entry = _strategy_entry(
                    run,
                    symbol_index=symbol_index,
                    symbol=symbol,
                    strategy=strategy,
                )
                values = _raw_returns(entry, run)
                if values.shape != (expected_length,):
                    raise ValueError(
                        f"candidate return length mismatch for {symbol}/{strategy}"
                    )
                metrics, _ = _metric_diagnostic_values(entry)
                reconstructed = _compound(values)
                if not math.isclose(
                    reconstructed,
                    float(metrics["total_return"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"persisted total return does not match raw returns for {symbol}/{strategy}"
                    )
                if int(metrics["n_periods"]) != expected_length:
                    raise ValueError(
                        f"persisted period count does not match return length for {symbol}/{strategy}"
                    )
                if run_index == 0:
                    representative_entry = entry
                    representative_returns = values.copy()
                    representative_metrics = dict(metrics)
                    continue
                assert representative_returns is not None
                assert representative_metrics is not None
                if not np.array_equal(values, representative_returns):
                    raise ValueError(
                        f"deterministic strategy is not seed invariant for {symbol}/{strategy}"
                    )
                if metrics != representative_metrics:
                    raise ValueError(
                        f"deterministic strategy metrics are not seed invariant for {symbol}/{strategy}"
                    )

            assert representative_entry is not None
            assert representative_returns is not None
            assert representative_metrics is not None
            total_return = float(representative_metrics["total_return"])
            net_pnl = initial_capital * total_return
            total_cost = float(representative_metrics["total_cost"])
            funding_pnl = float(representative_metrics["funding_pnl"])
            borrow_cost = float(representative_metrics["borrow_cost"])
            residual_pnl = net_pnl + total_cost - funding_pnl + borrow_cost
            recomposed = residual_pnl - total_cost + funding_pnl - borrow_cost
            if not math.isclose(recomposed, net_pnl, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError("realized-path accounting decomposition does not balance")

            year_returns = _period_returns(
                representative_returns,
                interval_owners,
                unit="Y",
            )
            if not math.isclose(
                _compound(np.asarray([value for _, value in year_returns])),
                total_return,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("calendar-year return decomposition does not recompose")
            positive_months, month_count = _month_positive_count(
                representative_returns,
                interval_owners,
            )
            correlation, beta = _market_sensitivity(
                representative_returns,
                mark_returns[:, symbol_index],
            )
            cells.append(
                LossAttributionCell(
                    symbol=symbol,
                    strategy=strategy,
                    net_total_return=total_return,
                    reconstructed_total_return=_compound(representative_returns),
                    initial_capital=initial_capital,
                    net_pnl=net_pnl,
                    explicit_execution_cost=total_cost,
                    funding_pnl=funding_pnl,
                    borrow_cost=borrow_cost,
                    residual_pnl=residual_pnl,
                    turnover_total=float(representative_metrics["turnover_total"]),
                    n_trades=int(representative_metrics["n_trades"]),
                    rebalance_events=int(representative_metrics["rebalance_events"]),
                    explicit_cost_flip_on_realized_path=(
                        net_pnl <= 0.0 and net_pnl + total_cost > 0.0
                    ),
                    year_returns=year_returns,
                    positive_months=positive_months,
                    month_count=month_count,
                    underlying_mark_correlation=correlation,
                    underlying_mark_beta=beta,
                )
            )

    return DeterministicLossAttributionReport(
        dataset_id=dataset.dataset_id,
        symbols=symbols,
        strategies=strategies,
        seed_count=len(runs),
        evaluation_start=start_text,
        evaluation_stop_exclusive=stop_text,
        cells=tuple(cells),
    )


__all__ = [
    "DeterministicLossAttributionReport",
    "LossAttributionCell",
    "analyze_deterministic_loss_attribution",
]
