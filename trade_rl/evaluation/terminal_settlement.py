"""Shared observer-only terminal residual classification and equity valuation.

This module reads native replay evidence. It does not place orders, amend the
ledger, waive exchange minimums or authorize a study.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import cast

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashLedgerIntervalEvidence,
    SharedCashReplayLedgerEvidence,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.capacity_reference import _capacity_reference_for_symbol
from trade_rl.simulation.quantities import exact_quantity


def terminal_dust(
    ledger: SharedCashReplayLedgerEvidence,
    dataset: MarketDataset,
    execution_cost: ExecutionCostConfig | None,
    final_interval: SharedCashLedgerIntervalEvidence,
) -> tuple[list[dict[str, object]], list[str], bool]:
    """Classify the terminal residual positions of a ledger.

    The engine enforces the venue minimum notional everywhere (at the processing
    bar's open), so a holding that decayed below it cannot be closed and stays a
    position. A residual is *dust* only if the engine itself rejected its terminal
    close for ``below_minimum_notional``: every non-submission order event of that
    symbol on the final interval is a rejection / no-fill with exactly that reason.
    A residual left open for any other reason (no liquidity, a participation cap, a
    partial fill, a latency wait, ...) is a genuine stuck position, whatever its
    value at the mark. Dust also needs a deterministic exit valuation, so a
    stochastic slippage model or a missing ``execution_cost`` classifies nothing as
    dust. Returns ``(dust, other_symbols, valued)``.

    The valuation is conservative: mark value minus the exit cost, notional x
    multiplier x (fee + venue taker fee + spread + impact x sqrt(participation)),
    re-derived from the Dataset with the executor's market-order rates; the
    participation uses the executor's capacity-volume basis (the default previous
    completed bar, or the processing bar when ``processing_bar_volume_capacity`` is
    explicitly on). The open-segment fraction limits both quote and native
    quantity pools; impact uses the larger of their participation rates.
    """

    final_row = int(final_interval.next_index)
    multipliers = dataset.resolved_array("contract_multipliers")
    marks = dataset.resolved_array("mark_price")[final_row]
    opens = dataset.open[final_row]
    minimum = dataset.resolved_array("minimum_notional")[final_row]
    symbol_participation_limits = dataset.resolved_array("max_participation_rate")[
        final_row
    ]
    valued = execution_cost is not None and execution_cost.slippage_std == 0.0
    reasons: dict[int, set[str | None]] = {}
    for event in final_interval.order_events:
        if event.event_type == "submitted":
            continue
        rejected = event.event_type in {"rejected", "no_fill"}
        reasons.setdefault(event.symbol_index, set()).add(
            event.reason if rejected else f"event:{event.event_type}"
        )
    dust: list[dict[str, object]] = []
    other: list[str] = []
    for index, value in enumerate(ledger.terminal_exact_quantities):
        quantity = Fraction(value)
        if quantity == 0:
            continue
        symbol = dataset.symbols[index]
        if (
            execution_cost is None
            or not valued
            or reasons.get(index) != {"below_minimum_notional"}
        ):
            other.append(symbol)
            continue
        multiplier = float(multipliers[index])
        notional = abs(float(quantity)) * float(marks[index]) * multiplier
        open_notional = abs(float(quantity)) * float(opens[index]) * multiplier
        _, capacity_notional, _, native_capacity = _capacity_reference_for_symbol(
            dataset,
            processing_index=final_row,
            processing_open_prices=opens,
            symbol=index,
            processing_bar_volume_capacity=(
                execution_cost.processing_bar_volume_capacity
            ),
        )
        # A zero-capacity bar or an overflowing participation ratio has no
        # finite, defensible hypothetical exit-cost value. Keep the holding
        # non-dust so the ledger validity gate fails closed.
        if (
            not math.isfinite(notional)
            or not math.isfinite(open_notional)
            or not math.isfinite(capacity_notional)
            or capacity_notional <= 0.0
        ):
            other.append(symbol)
            continue
        participation = open_notional / capacity_notional
        symbol_limit = float(symbol_participation_limits[index])
        # A minimum-notional rejection is not a waiver of either capacity pool.
        limit = min(execution_cost.max_participation_rate, symbol_limit)
        fraction = execution_cost.trigger_volume_fractions[0]
        if (
            not math.isfinite(symbol_limit)
            or limit <= 0.0
            or fraction <= 1e-12
            or capacity_notional * limit * fraction <= 1e-12
            or not math.isfinite(participation)
            or participation > limit * fraction
        ):
            other.append(symbol)
            continue
        if native_capacity is not None:
            native_limit = native_capacity * limit * fraction
            if (
                not math.isfinite(native_limit)
                or native_limit <= 1e-12
                or abs(quantity) > exact_quantity(native_limit)
            ):
                other.append(symbol)
                continue
            participation = max(participation, abs(float(quantity)) / native_capacity)
        if not math.isfinite(participation) or participation > limit * fraction:
            other.append(symbol)
            continue
        unit_cost = execution_cost.multiplier * (
            execution_cost.fee_rate
            + float(dataset.resolved_array("fee_rate")[final_row, index])
            + execution_cost.taker_fee_rate
            + float(dataset.resolved_array("taker_fee_rate")[final_row, index])
            + execution_cost.spread_rate
            + float(dataset.resolved_array("spread_rate")[final_row, index])
            + execution_cost.impact_rate * math.sqrt(participation)
        )
        mark_value = float(quantity) * float(marks[index]) * multiplier
        exit_cost = notional * unit_cost
        # A hypothetical exit costing more than the entire marked position
        # cannot be conservatively treated as valued terminal dust.
        if (
            not math.isfinite(mark_value)
            or not math.isfinite(exit_cost)
            or exit_cost > notional
        ):
            other.append(symbol)
            continue
        dust.append(
            {
                "symbol": symbol,
                "quantity": str(quantity),
                "notional_usdt": notional,
                "open_notional_usdt": open_notional,
                "minimum_notional_usdt": max(
                    float(minimum[index]), execution_cost.minimum_notional
                ),
                "reason": "below_minimum_notional",
                "mark_value_usdt": mark_value,
                "exit_cost_usdt": exit_cost,
            }
        )
    return dust, other, valued


def terminal_dust_adjusted_equity(
    equity: np.ndarray, reconciliation: Mapping[str, object]
) -> np.ndarray:
    """Return the equity path with terminal dust valued net of its exit cost.

    The ledger marks a residual at the last price; an exit that could not happen is
    never credited, so the final point is reduced by the modelled exit cost. A flat
    terminal book returns an unchanged copy.
    """

    adjusted = np.array(equity, dtype=np.float64, copy=True)
    adjusted[-1] -= float(
        cast(float, reconciliation.get("terminal_dust_exit_cost_usdt", 0.0))
    )
    return adjusted


__all__ = ["terminal_dust", "terminal_dust_adjusted_equity"]
