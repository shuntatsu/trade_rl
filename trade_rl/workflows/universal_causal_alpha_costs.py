"""Causal one-way execution-cost estimates for Universal teacher targets."""

from __future__ import annotations

import math
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
    """Estimate one-way costs using only state available at each decision close."""

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
        values.size != n_bars for values in (fee, maker, taker, spread, participation)
    ):
        raise ValueError("causal alpha dataset cost arrays must align with n_bars")
    if np.any(fee < 0.0) or np.any(maker < 0.0) or np.any(taker < 0.0):
        raise ValueError("causal alpha dataset fee rates must be non-negative")
    if np.any(spread < 0.0) or np.any((participation <= 0.0) | (participation > 1.0)):
        raise ValueError("causal alpha dataset spread/participation rates are invalid")

    # These arrays are historical bar observations. The future execution row is
    # deliberately used only for the executable-range check above; pricing a
    # target with execution-row spread or participation would leak information
    # unavailable when the target is chosen.
    selected_participation = np.minimum(
        execution_cost.max_participation_rate, participation[decisions]
    )
    limit = execution_cost.order_type == "limit"
    venue_fee = (
        execution_cost.maker_fee_rate + maker[decisions]
        if limit
        else execution_cost.taker_fee_rate + taker[decisions]
    )
    spread_multiplier = 0.5 if limit else 1.0
    rates = execution_cost.multiplier * (
        execution_cost.fee_rate
        + fee[decisions]
        + venue_fee
        + spread_multiplier * (execution_cost.spread_rate + spread[decisions])
        + execution_cost.impact_rate * np.sqrt(selected_participation)
    )
    if not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("causal alpha one-way cost rates are invalid")
    return rates.astype(np.float64, copy=False)


def causal_alpha_liquidity_weight_caps(
    dataset: Any,
    *,
    decision_indices: object,
    reference_portfolio_value: float,
    max_position_to_market_notional: float,
    lookback_decisions: int,
    lower_quantile: float,
    safety_multiplier: float,
) -> np.ndarray:
    """Estimate conservative executable weights from strictly prior liquidity."""

    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    if decisions.size == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
        raise ValueError("causal alpha liquidity decisions must be strictly increasing")
    if not math.isfinite(reference_portfolio_value) or reference_portfolio_value <= 0.0:
        raise ValueError("reference_portfolio_value must be finite and positive")
    if (
        not math.isfinite(max_position_to_market_notional)
        or max_position_to_market_notional <= 0.0
    ):
        raise ValueError("max_position_to_market_notional must be finite and positive")
    if (
        isinstance(lookback_decisions, bool)
        or not isinstance(lookback_decisions, int)
        or lookback_decisions <= 0
    ):
        raise ValueError("lookback_decisions must be a positive integer")
    if not math.isfinite(lower_quantile) or not 0.0 <= lower_quantile <= 0.5:
        raise ValueError("lower_quantile must be finite and within [0, 0.5]")
    if not math.isfinite(safety_multiplier) or not 0.0 < safety_multiplier <= 1.0:
        raise ValueError("safety_multiplier must be finite and within (0, 1]")
    market_notional = getattr(dataset, "market_notional", None)
    if not callable(market_notional):
        raise TypeError("causal alpha liquidity dataset must expose market_notional")

    first_history_index = int(decisions[0]) - lookback_decisions
    if first_history_index < 0:
        raise ValueError("causal alpha liquidity history is incomplete")
    notionals = np.asarray(
        [
            np.asarray(market_notional(index), dtype=np.float64).reshape(-1)
            for index in range(first_history_index, int(decisions[-1]))
        ],
        dtype=np.float64,
    )
    if notionals.ndim != 2 or notionals.shape[1] != 1:
        raise ValueError("causal alpha liquidity requires single-symbol history")
    if not np.isfinite(notionals).all() or np.any(notionals < 0.0):
        raise ValueError(
            "causal alpha liquidity history must be finite and non-negative"
        )

    caps = np.empty(decisions.size, dtype=np.float64)
    for offset, decision in enumerate(decisions):
        stop = int(decision) - first_history_index
        start = stop - lookback_decisions
        history = notionals[start:stop]
        if history.shape != (lookback_decisions, 1):
            raise ValueError("causal alpha liquidity history is incomplete")
        conservative_notional = float(
            np.quantile(history[:, 0], lower_quantile, method="linear")
        )
        caps[offset] = (
            conservative_notional
            * max_position_to_market_notional
            * safety_multiplier
            / reference_portfolio_value
        )
    return caps


__all__ = [
    "causal_alpha_liquidity_weight_caps",
    "causal_alpha_one_way_cost_rates",
]
