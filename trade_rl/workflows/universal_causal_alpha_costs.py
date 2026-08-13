"""Causal one-way execution-cost estimates for Universal teacher targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.simulation.execution import ExecutionCostConfig


def _single_symbol_cost_array(dataset: Any, name: str) -> np.ndarray:
    values = np.asarray(dataset.resolved_array(name), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(
            f"causal alpha dataset {name} must be a finite single-symbol array"
        )
    return values


def causal_alpha_one_way_cost_rates(
    dataset: Any,
    execution_cost: ExecutionCostConfig,
    *,
    decision_indices: object,
    signal_delay_decisions: int,
    decision_bars: int,
) -> np.ndarray:
    if not isinstance(execution_cost, ExecutionCostConfig):
        raise TypeError("causal alpha cost rates require ExecutionCostConfig")
    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    if decisions.size == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
        raise ValueError("causal alpha cost decisions must be strictly increasing")
    if (
        isinstance(signal_delay_decisions, bool)
        or not isinstance(signal_delay_decisions, int)
        or signal_delay_decisions < 0
    ):
        raise ValueError("signal_delay_decisions must be a non-negative integer")
    if (
        isinstance(decision_bars, bool)
        or not isinstance(decision_bars, int)
        or decision_bars <= 0
    ):
        raise ValueError("decision_bars must be a positive integer")
    execution_indices = decisions + 1 + signal_delay_decisions * decision_bars
    n_bars = getattr(dataset, "n_bars", None)
    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars <= 0:
        raise ValueError("causal alpha cost dataset n_bars is invalid")
    if np.any(execution_indices >= n_bars):
        raise ValueError("causal alpha first executable cost row is unavailable")

    fee = _single_symbol_cost_array(dataset, "fee_rate")
    maker = _single_symbol_cost_array(dataset, "maker_fee_rate")
    taker = _single_symbol_cost_array(dataset, "taker_fee_rate")
    spread = _single_symbol_cost_array(dataset, "spread_rate")
    participation = _single_symbol_cost_array(dataset, "max_participation_rate")
    if any(
        values.size != n_bars
        for values in (fee, maker, taker, spread, participation)
    ):
        raise ValueError("causal alpha dataset cost arrays must align with n_bars")
    if np.any(fee < 0.0) or np.any(maker < 0.0) or np.any(taker < 0.0):
        raise ValueError("causal alpha dataset fee rates must be non-negative")
    if np.any(spread < 0.0) or np.any(
        (participation <= 0.0) | (participation > 1.0)
    ):
        raise ValueError("causal alpha dataset spread/participation rates are invalid")

    selected_participation = np.minimum(
        execution_cost.max_participation_rate, participation[execution_indices]
    )
    limit = execution_cost.order_type == "limit"
    venue_fee = (
        execution_cost.maker_fee_rate + maker[execution_indices]
        if limit
        else execution_cost.taker_fee_rate + taker[execution_indices]
    )
    spread_multiplier = 0.5 if limit else 1.0
    rates = execution_cost.multiplier * (
        execution_cost.fee_rate
        + fee[execution_indices]
        + venue_fee
        + spread_multiplier * (execution_cost.spread_rate + spread[execution_indices])
        + execution_cost.impact_rate * np.sqrt(selected_participation)
    )
    if not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("causal alpha one-way cost rates are invalid")
    return rates.astype(np.float64, copy=False)


__all__ = ["causal_alpha_one_way_cost_rates"]
