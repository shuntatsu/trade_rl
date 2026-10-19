"""Read the clock and event projection of one parsed H receipt row."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.market_replay_oracle.events import (
    EventBatchPin,
    OrderEventContext,
    OrderEventEvidence,
    read_pinned_order_events,
)
from tools.market_replay_oracle.identity import canonical


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value) is not dict:
        raise ValueError("native event row must be a parsed JSON object")
    return value


def _matches(value: object, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise ValueError("native event row differs from the bound integer clock/symbol")


def _execution(native: dict[str, Any], context: OrderEventContext) -> None:
    execution = _object(native["execution"])
    _matches(execution["next_index"], context.processing_index)
    _matches(execution["bars_advanced"], 1)


def _batch(
    native: dict[str, Any], context: OrderEventContext
) -> tuple[OrderEventEvidence, ...]:
    values = native["order_events"]
    if type(values) is not list or len(values) > 4096:
        raise ValueError("native order_events must be a present bounded array")
    try:
        raw = canonical(values)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("native order_events is not finite JSON") from error
    # These bytes are a derived child projection, never the original gzip bytes.
    pin = EventBatchPin(len(raw), hashlib.sha256(raw).hexdigest())
    return read_pinned_order_events(raw, pin, context)


def actor_events(
    value: object, context: OrderEventContext
) -> tuple[tuple[OrderEventEvidence, ...] | None, bool]:
    """Retain an explicit missing native tail; reject missing keys and bad clocks."""
    row = _object(value)
    try:
        _matches(row["decision_index"], context.decision_index)
        _matches(row["decision_time_ns"], context.decision_time_ns)
        failure = row["native_validation_failure"]
        if failure is not None and (
            type(failure) is not dict
            or set(failure) != {"error_type", "message"}
            or type(failure["error_type"]) is not str
            or not failure["error_type"]
            or type(failure["message"]) is not str
        ):
            raise ValueError("native validation failure must retain its error shape")
        if row["native"] is None:
            if failure is not None:
                raise ValueError("rejected native facts cannot be absent")
            return None, False
        native = _object(row["native"])
        for key, expected in (
            ("decision_index", context.decision_index),
            ("decision_time_ns", context.decision_time_ns),
            ("processing_index", context.processing_index),
            ("processing_time_ns", context.processing_time_ns),
            ("symbol_index", context.symbol_index),
        ):
            _matches(native[key], expected)
        _execution(native, context)
        return _batch(native, context), failure is not None
    except KeyError as error:
        raise ValueError("actor event row is missing a required field") from error


def reserve_events(
    value: object, context: OrderEventContext
) -> tuple[tuple[OrderEventEvidence, ...], bool]:
    """Use reserve wrapper clocks and the embedded one-bar execution clock."""
    row = _object(value)
    try:
        _matches(row["start_index"], context.decision_index)
        _matches(row["processing_index"], context.processing_index)
        _matches(row["processing_time_ns"], context.processing_time_ns)
        if type(row["clock_valid"]) is not bool:
            raise ValueError("reserve clock_valid must be a boolean")
        native = _object(row["native"])
        _execution(native, context)
        return _batch(native, context), not row["clock_valid"]
    except KeyError as error:
        raise ValueError("reserve event row is missing a required field") from error
