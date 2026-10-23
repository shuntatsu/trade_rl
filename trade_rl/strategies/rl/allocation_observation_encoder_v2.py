"""Opt-in exact account/order projection using the immutable v2 declaration.

This partial state omits native ID priority, holding ages and cash reservations.
Declared snapshot consistency does not authenticate an account or market source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from numbers import Real
from typing import Any, cast

import numpy as np

from trade_rl.strategies.allocation_action import AllocationDecision
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot
from trade_rl.strategies.rl.allocation_observation_v2 import (
    FORECAST_FIELDS,
    ORDER_FIELDS,
    AllocationObservationSchema,
)

_TIFS = ("ioc", "day", "gtc")
_STATUSES = ("submitted", "latency_wait", "eligible", "triggered", "partially_filled")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _scalar(
    value: object, name: str, *, minimum: int | None = None, maximum: int | None = None
) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-boolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside its finite reporting range") from error
    if not finite:
        raise ValueError(f"{name} must be finite")
    result = Fraction(str(value))
    if (minimum is not None and result < minimum) or (
        maximum is not None and result > maximum
    ):
        raise ValueError(f"{name} is outside its range")
    return result


def _quantity(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("exact quantity must be a canonical rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "exact quantity must be a canonical rational string"
        ) from error
    if str(result) != value:
        raise ValueError("exact quantity must be a canonical rational string")
    return result


def _project(value: Fraction) -> np.float32:
    if abs(value) > Fraction.from_float(float(np.finfo(np.float32).max)):
        raise ValueError("observation overflows finite float32")
    with np.errstate(under="ignore"):
        projected = np.float32(float(value))
        if abs(Fraction.from_float(float(projected))) > abs(value):
            projected = np.nextafter(projected, np.float32(0))
    if value != 0 and projected == 0:
        raise ValueError("nonzero observation underflows float32")
    return projected


def _raw_features(values: tuple[float, ...]) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.array(values, dtype=np.float32)
    if not np.isfinite(result).all() or any(
        value != 0 and projected == 0
        for value, projected in zip(values, result, strict=True)
    ):
        raise ValueError("raw features must remain finite and nonzero when nonzero")
    return result


def _order_values(
    order: Mapping[str, Any],
    *,
    mark: Fraction,
    multiplier: Fraction,
    capital: Fraction,
    decision_index: int,
    episode_steps: int,
) -> tuple[Fraction, ...]:
    intent = order["intent"]

    def offset(index: int) -> Fraction:
        return Fraction(index - decision_index, episode_steps)

    clocks: list[Fraction] = []
    for index in (
        intent["expiry_index"],
        order["trigger_index"],
        order["last_processed_index"],
    ):
        clocks.extend(
            (
                Fraction(index is not None),
                Fraction(0) if index is None else offset(index),
            )
        )
    quantities = tuple(
        _quantity(order[f"exact_{name}_quantity"]) * mark * multiplier / capital
        for name in ("requested", "cumulative_filled", "remaining")
    )
    return (
        Fraction(1),
        *quantities,
        _scalar(order["cumulative_filled_notional"], "filled notional", minimum=0)
        / capital,
        _scalar(intent["submission_reference_price"], "submission reference") / mark,
        _scalar(intent["decision_equity"], "order decision equity") / capital,
        offset(intent["submit_index"]),
        offset(intent["eligible_index"]),
        *clocks,
        Fraction(intent["reduce_only"]),
        *(Fraction(intent["time_in_force"] == tif) for tif in _TIFS),
        *(Fraction(order["status"] == status) for status in _STATUSES),
    )


def encode_allocation_observation_v2(
    snapshot: AllocationAccountSnapshot,
    decision: AllocationDecision,
    *,
    schema: AllocationObservationSchema,
    episode_steps: int,
) -> np.ndarray:
    """Fresh F+9+13+24K float32 values from matching causal declarations.

    Exact economic facts determine order-slot sorting; IDs and global indices
    remain evidence only. This sort does not reproduce native capacity priority.
    v1 pending aggregates are reporting summaries, not exact slot quantities.
    """
    if (
        not isinstance(snapshot, AllocationAccountSnapshot)
        or not isinstance(decision, AllocationDecision)
        or not isinstance(schema, AllocationObservationSchema)
    ):
        raise ValueError("v2 requires the declared snapshot, decision and schema types")
    if (
        _integer(episode_steps, "episode_steps") != schema.episode_steps
        or episode_steps < decision.remaining_steps
        or schema.initial_capital != decision.initial_capital
        or schema.feature_names != decision.feature_names
    ):
        raise ValueError("decision features, capital and horizon must match the schema")
    if len(snapshot.active_orders) > schema.max_active_orders:
        raise ValueError("active orders exceed the declared order slots")
    book = cast(Mapping[str, Any], snapshot.book_facts)
    context, inputs = decision.baseline.context, decision.baseline.inputs
    mark = _scalar(book["mark_prices"][0], "mark")
    multiplier = _scalar(book["contract_multipliers"][0], "multiplier")
    position = _quantity(book["exact_quantities"][0])
    cash = _scalar(book["cash"], "cash")
    equity = _scalar(book["equity"], "equity")
    peak = _scalar(book["peak_value"], "peak")
    maximum_dd = _scalar(book["max_drawdown"], "maximum drawdown", minimum=0, maximum=1)
    current_dd = _scalar(
        book["current_drawdown"], "current drawdown", minimum=0, maximum=1
    )
    current_weight = (
        float(book["quantities"][0]) * float(mark) * float(multiplier) / float(equity)
    )
    expected_context = {
        "account_id": snapshot.account_id,
        "symbol": snapshot.symbol,
        "state_digest": snapshot.source_state_digest,
        "decision_time": np.datetime64(snapshot.decision_time, "ns"),
        "quantity": str(position),
        "cash": book["cash"],
        "equity": book["equity"],
        "current_weight": current_weight,
    }
    if (
        any(getattr(context, name) != value for name, value in expected_context.items())
        or _scalar(decision.max_drawdown, "decision drawdown") != maximum_dd
        or decision.pending_count != len(snapshot.active_orders)
        or inputs.symbol != snapshot.symbol
        or inputs.decision_time != context.decision_time
    ):
        raise ValueError(
            "snapshot account/source/time/position facts must match the decision"
        )
    capital = _scalar(schema.initial_capital, "initial_capital")
    values = tuple(
        _scalar(getattr(inputs, name), name) for name in FORECAST_FIELDS[:-1]
    ) + (
        _scalar(decision.baseline.target_weight, "baseline target"),
        position * mark * multiplier / capital,
        _scalar(context.current_weight, "current weight"),
        cash / capital,
        equity / capital,
        peak / capital,
        current_dd,
        maximum_dd,
        _scalar(book["margin_used"], "margin_used", minimum=0) / capital,
        _scalar(book["maintenance_margin"], "maintenance_margin", minimum=0, maximum=1),
        _scalar(book["maintenance_requirement"], "maintenance_requirement", minimum=0)
        / capital,
        _scalar(book["margin_deficit"], "margin_deficit", minimum=0) / capital,
        Fraction(snapshot.observable_tradable),
        Fraction(decision.remaining_steps, episode_steps),
    )
    rows = sorted(
        _order_values(
            cast(Mapping[str, Any], order),
            mark=mark,
            multiplier=multiplier,
            capital=capital,
            decision_index=snapshot.decision_index,
            episode_steps=episode_steps,
        )
        for order in snapshot.active_orders
    )
    values += tuple(value for row in rows for value in row)
    values += (Fraction(0),) * (
        len(ORDER_FIELDS) * (schema.max_active_orders - len(rows))
    )
    return np.concatenate(
        (
            _raw_features(decision.feature_values),
            np.array([_project(value) for value in values], dtype=np.float32),
        )
    )


__all__ = ["encode_allocation_observation_v2"]
