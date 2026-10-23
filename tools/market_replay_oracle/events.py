"""Read pinned single-symbol event arrays, without authenticating their origin."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tools.market_replay_oracle.identity import (
    _reject_constant,
    _unique_pairs,
    canonical,
    json_object,
)

_MAX_BYTES = 1024 * 1024
_MAX_EVENTS = 4096
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "order_id",
        "replaced_order_id",
        "dataset_id",
        "execution_policy_digest",
        "symbol_index",
        "event_type",
        "processing_index",
        "timestamp_ns",
        "previous_status",
        "new_status",
        "requested_quantity",
        "remaining_quantity",
        "filled_quantity",
        "execution_price",
        "filled_notional",
        "capacity_before",
        "capacity_after",
        "participation_rate",
        "trigger_segment",
        "available_volume_fraction",
        "reason",
        "path_mode",
        "path_points",
    }
)
_STATUSES = frozenset(
    {
        "submitted",
        "latency_wait",
        "eligible",
        "triggered",
        "partially_filled",
        "filled",
        "rejected",
        "expired",
        "cancelled",
    }
)
_EVENT_TYPES = (_STATUSES - {"partially_filled"}) | {"partial_fill", "no_fill"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: object) -> None:
    _require(
        isinstance(value, str) and _DIGEST.fullmatch(value) is not None,
        "invalid digest",
    )


def _integer(value: object, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value  # type: ignore[return-value]


def _number(value: object, name: str) -> float:
    _require(type(value) in (int, float), f"{name} must be numeric, excluding bool")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} exceeds finite float range") from error
    _require(math.isfinite(result), f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class EventBatchPin:
    """Caller-supplied byte identity, not a source or runtime credential."""

    raw_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.raw_bytes) is int and 0 < self.raw_bytes <= _MAX_BYTES,
            "event pin size is outside the 1 MiB bound",
        )
        _digest(self.sha256)


@dataclass(frozen=True, slots=True)
class OrderEventContext:
    """Expected facts for one symbol/advance; the caller must bind their source."""

    dataset_id: str
    execution_policy_digest: str
    symbol_index: int
    decision_index: int
    decision_time_ns: int
    processing_index: int
    processing_time_ns: int
    effective_lot_size: float

    def __post_init__(self) -> None:
        _digest(self.dataset_id)
        _digest(self.execution_policy_digest)
        for name in (
            "symbol_index",
            "decision_index",
            "decision_time_ns",
            "processing_index",
            "processing_time_ns",
        ):
            _integer(getattr(self, name), name)
        _require(
            self.processing_index == self.decision_index + 1
            and self.processing_time_ns > self.decision_time_ns,
            "context must describe one ordered advance",
        )
        lot = _number(self.effective_lot_size, "effective lot")
        _require(lot >= 0.0, "effective lot must be known and nonnegative")


@dataclass(frozen=True, slots=True)
class OrderEventEvidence:
    """Immutable parsed values only; constructing this type grants no authority."""

    _payload: bytes = field(repr=False)
    exact_filled_quantity: Fraction

    def payload(self) -> dict[str, Any]:
        """Return a detached comparison projection of the entire closed event."""
        return json_object(self._payload)


def _project_inward(exact: Fraction) -> float:
    try:
        result = float(exact)
    except OverflowError as error:
        raise ValueError("fill projection exceeds finite float range") from error
    _require(math.isfinite(result), "fill projection must be finite")
    while abs(Fraction(str(result))) > abs(exact):
        result = math.nextafter(result, 0.0)
    return result


def _quantity(event: dict[str, Any], context: OrderEventContext) -> Fraction:
    quantity = _number(event["filled_quantity"], "filled quantity")
    filled = event["event_type"] in {"filled", "partial_fill"}
    if not filled:
        _require(
            event["schema_version"] == "order_event_v1", "non-fill events must use v1"
        )
        _require(
            quantity == 0.0
            and event["execution_price"] is None
            and event["filled_notional"] == 0,
            "non-fill carries fill values",
        )
        return Fraction(0)
    requested = _number(event["requested_quantity"], "requested quantity")
    _require(
        event["processing_index"] == context.processing_index,
        "fill must occur at the processing clock",
    )
    _require(
        quantity != 0 and (quantity > 0) == (requested > 0),
        "fill quantity must have the requested sign",
    )
    _require(event["execution_price"] is not None, "fill requires its execution price")
    if event["schema_version"] == "order_event_v1":
        _require(
            context.effective_lot_size == 0.0, "positive lot fill requires v2 witness"
        )
        return Fraction(str(quantity))
    count = event["filled_lot_count"]
    _require(
        type(count) is int and count != 0, "lot count must be a nonzero signed integer"
    )
    step = _number(event["lot_size"], "lot size")
    _require(
        step > 0.0 and step == context.effective_lot_size,
        "event lot differs from expected effective lot",
    )
    exact = count * Fraction(str(step))
    _require(
        _project_inward(exact) == quantity, "fill projection differs from lot witness"
    )
    return exact


def _event(
    value: object, context: OrderEventContext, sequence: int
) -> OrderEventEvidence:
    _require(type(value) is dict, "event must be an object")
    event: dict[str, Any] = value  # type: ignore[assignment]
    schema = event.get("schema_version")
    _require(schema in ("order_event_v1", "order_event_v2"), "unsupported event schema")
    fields = _FIELDS | (
        {"filled_lot_count", "lot_size"} if schema == "order_event_v2" else set()
    )
    _require(
        set(event) in (fields, fields | {"reduce_only"}), "event field closure mismatch"
    )
    _require(
        "reduce_only" not in event or event["reduce_only"] is True,
        "canonical reduce_only must be omitted when false or be true",
    )
    _require(
        _integer(event["sequence"], "sequence") == sequence,
        "event sequence must be complete within its advance",
    )
    _require(
        _integer(event["symbol_index"], "symbol_index") == context.symbol_index,
        "event symbol_index differs from context",
    )
    index = _integer(event["processing_index"], "processing_index")
    _require(
        index in (context.decision_index, context.processing_index),
        "event processing_index differs from context",
    )
    expected_time = (
        context.decision_time_ns
        if index == context.decision_index
        else context.processing_time_ns
    )
    _require(
        _integer(event["timestamp_ns"], "timestamp_ns") == expected_time,
        "event timestamp differs from its bound index",
    )
    for name in ("dataset_id", "execution_policy_digest"):
        _require(
            event[name] == getattr(context, name), f"event {name} differs from context"
        )
    _digest(event["order_id"])
    if event["replaced_order_id"] is not None:
        _digest(event["replaced_order_id"])
    for name, allowed in (
        ("event_type", _EVENT_TYPES),
        ("previous_status", _STATUSES),
        ("new_status", _STATUSES),
        ("path_mode", {"optimistic", "neutral", "conservative"}),
    ):
        _require(
            isinstance(event[name], str) and event[name] in allowed,
            f"unsupported {name}",
        )
    for name in ("reason", "trigger_segment"):
        _require(
            event[name] is None or isinstance(event[name], str),
            f"{name} must be string or null",
        )
    kind = (
        "partially_filled"
        if event["event_type"] == "partial_fill"
        else event["event_type"]
    )
    _require(
        event["new_status"]
        == (event["previous_status"] if kind == "no_fill" else kind),
        "event kind differs from status",
    )
    for name in ("requested_quantity", "remaining_quantity", "filled_quantity"):
        _number(event[name], name)
    _require(
        abs(_number(event["requested_quantity"], "requested quantity")) > 1e-12,
        "requested quantity must be nonzero",
    )
    for name in ("filled_notional", "capacity_before", "capacity_after"):
        _require(_number(event[name], name) >= 0.0, f"{name} must be nonnegative")
    for name in ("participation_rate", "available_volume_fraction"):
        _require(
            0.0 <= _number(event[name], name) <= 1.0, f"{name} must be within [0, 1]"
        )
    if event["execution_price"] is not None:
        _require(
            _number(event["execution_price"], "execution price") > 0.0,
            "execution price must be positive",
        )
    points = event["path_points"]
    _require(
        type(points) is list and len(points) in (0, 4),
        "path must be empty or contain four prices",
    )
    for point in points:
        _require(_number(point, "path price") > 0.0, "path prices must be positive")
    return OrderEventEvidence(canonical(event), _quantity(event, context))


def read_pinned_order_events(
    raw: bytes, pin: EventBatchPin, context: OrderEventContext
) -> tuple[OrderEventEvidence, ...]:
    """Check bytes and per-advance evidence; no order admission or ledger replay.

    Input is one native ``order_events`` array serialized as UTF-8 JSON. Each
    batch starts its own sequence at zero. Raw pins and context are supplied by
    the caller; this function does not bind an outer receipt, attempt or source.
    """
    _require(
        type(pin) is EventBatchPin and type(context) is OrderEventContext,
        "event pin and context types are required",
    )
    pin.__post_init__()
    context.__post_init__()
    _require(
        type(raw) is bytes
        and len(raw) == pin.raw_bytes
        and hashlib.sha256(raw).hexdigest() == pin.sha256,
        "event byte pin mismatch",
    )
    try:
        values = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        canonical(values)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid event JSON") from error
    _require(
        type(values) is list and len(values) <= _MAX_EVENTS,
        "event batch must be an array of at most 4096 events",
    )
    return tuple(
        _event(value, context, sequence) for sequence, value in enumerate(values)
    )
