"""Manufactured event evidence; expected quantities come from hand rationals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any

import pytest

import tools.market_replay_oracle as oracle


def _api() -> tuple[Any, Any, Any]:
    names = ("EventBatchPin", "OrderEventContext", "read_pinned_order_events")
    assert all(hasattr(oracle, name) for name in names), (
        "strict event reader is missing"
    )
    return tuple(getattr(oracle, name) for name in names)  # type: ignore[return-value]


def _event(**changes: Any) -> dict[str, Any]:
    event = {
        "schema_version": "order_event_v2",
        "sequence": 0,
        "order_id": "c" * 64,
        "replaced_order_id": None,
        "dataset_id": "a" * 64,
        "execution_policy_digest": "b" * 64,
        "symbol_index": 1,
        "event_type": "partial_fill",
        "processing_index": 7,
        "timestamp_ns": 1_700_000_000_000_000_000,
        "previous_status": "eligible",
        "new_status": "partially_filled",
        "requested_quantity": 1.0,
        "remaining_quantity": 0.7,
        "filled_quantity": 0.3,
        "execution_price": 100.0,
        "filled_notional": 30.0,
        "capacity_before": 100.0,
        "capacity_after": 70.0,
        "participation_rate": 0.3,
        "trigger_segment": None,
        "available_volume_fraction": 1.0,
        "reason": None,
        "path_mode": "conservative",
        "path_points": [],
        "filled_lot_count": 10,
        "lot_size": 0.03,
    }
    event.update(changes)
    return event


def _legacy(*, fill: bool = True, **changes: Any) -> dict[str, Any]:
    event = _event(schema_version="order_event_v1")
    event.pop("filled_lot_count")
    event.pop("lot_size")
    if not fill:
        event.update(
            event_type="no_fill",
            new_status="eligible",
            remaining_quantity=1.0,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            capacity_after=100.0,
            participation_rate=0.0,
        )
    event.update(changes)
    return event


def _context(**changes: Any) -> Any:
    _, context, _ = _api()
    values = dict(
        dataset_id="a" * 64,
        execution_policy_digest="b" * 64,
        symbol_index=1,
        decision_index=6,
        decision_time_ns=1_699_996_400_000_000_000,
        processing_index=7,
        processing_time_ns=1_700_000_000_000_000_000,
        effective_lot_size=0.03,
    )
    values.update(changes)
    return context(**values)


def _read(events: list[dict[str, Any]], **context: Any) -> Any:
    pin_type, _, reader = _api()
    raw = json.dumps(events, allow_nan=False).encode()
    pin = pin_type(raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return reader(raw, pin, _context(**context))


def test_quantized_fill_retains_the_exact_hand_rational() -> None:
    events = _read([_event()])
    assert isinstance(events, tuple) and len(events) == 1
    assert events[0].exact_filled_quantity == Fraction(3, 10)
    payload = events[0].payload()
    payload["filled_lot_count"] = 999
    assert events[0].payload()["filled_lot_count"] == 10
    with pytest.raises(FrozenInstanceError):
        events[0].exact_filled_quantity = Fraction(9)


def test_zero_lot_legacy_fill_uses_its_decimal_float_quantity() -> None:
    event = _legacy(
        filled_quantity=-0.1, requested_quantity=-1.0, remaining_quantity=-0.9
    )
    assert _read([event], effective_lot_size=0.0)[0].exact_filled_quantity == Fraction(
        -1, 10
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_float_projection_is_conservative_beyond_integer_precision(sign: int) -> None:
    # 2**53 + 3 rounds outward to 2**53 + 4; the native contract projects inward.
    count = sign * 9_007_199_254_740_995
    inward = sign * 9_007_199_254_740_994.0
    event = _event(
        filled_lot_count=count,
        lot_size=1.0,
        filled_quantity=inward,
        requested_quantity=sign * 1e17,
    )
    assert _read([event], effective_lot_size=1.0)[0].exact_filled_quantity == Fraction(
        count
    )
    event["filled_quantity"] = float(count)
    with pytest.raises(ValueError, match="projection"):
        _read([event], effective_lot_size=1.0)


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_id": "d" * 64},
        {"execution_policy_digest": "d" * 64},
        {"symbol_index": 0},
        {"processing_index": 8},
        {"processing_time_ns": 1_700_000_000_000_000_001},
        {"effective_lot_size": 0.02},
    ],
)
def test_valid_event_from_a_different_bound_row_is_rejected(
    change: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _read([_event()], **change)


def test_v1_fills_need_zero_rule_but_nonfills_remain_compatible() -> None:
    with pytest.raises(ValueError, match="lot"):
        _read([_legacy()])
    assert _read([_legacy(fill=False)])[0].exact_filled_quantity == 0
    with pytest.raises(ValueError):
        _read([_legacy(fill=False, symbol_index=0)])
    with pytest.raises(ValueError, match="lot"):
        _read([_event()], effective_lot_size=0.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"filled_lot_count": True},
        {"filled_lot_count": 0},
        {"filled_lot_count": 10.0},
        {"filled_lot_count": 10**400},
        {"filled_lot_count": -10},
        {"lot_size": True},
        {"lot_size": 0.0},
        {"lot_size": -0.03},
        {"lot_size": 0.06},
        {"filled_quantity": 0.0},
        {"filled_quantity": 0.31},
        {"requested_quantity": -1.0},
        {"event_type": "no_fill"},
        {"new_status": "filled"},
        {"execution_price": None},
    ],
)
def test_invalid_fill_witnesses_cannot_be_reconstructed(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _read([_event(**changes)])


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": True},
        {"symbol_index": True},
        {"processing_index": -1},
        {"timestamp_ns": 1.7e18},
        {"order_id": "not-a-digest"},
        {"replaced_order_id": 3},
        {"previous_status": "unknown"},
        {"event_type": "unknown"},
        {"reason": 4},
        {"trigger_segment": False},
        {"reduce_only": 1},
        {"reduce_only": False},
        {"requested_quantity": True},
        {"requested_quantity": 0.0},
        {"remaining_quantity": "0.7"},
        {"capacity_before": -1.0},
        {"capacity_after": -1.0},
        {"filled_notional": -1.0},
        {"participation_rate": 1.1},
        {"available_volume_fraction": -0.1},
        {"path_mode": "unknown"},
        {"path_points": [1.0, 2.0]},
        {"path_points": [1.0, 2.0, 0.0, 3.0]},
        {"extra": "ignored?"},
    ],
)
def test_closed_event_fields_and_types(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _read([_event(**changes)])


def test_missing_fields_and_forged_nonfill_quantity_reject() -> None:
    event = _event()
    event.pop("reason")
    with pytest.raises(ValueError):
        _read([event])
    with pytest.raises(ValueError):
        _read([_legacy(fill=False, filled_quantity=0.3)])


def test_sequence_is_complete_within_each_batch_and_restarts_on_next_advance() -> None:
    assert len(_read([_event(), _event(sequence=1)])) == 2
    assert len(_read([_event()], processing_index=7)) == 1
    assert (
        len(_read([_event(processing_index=8)], decision_index=7, processing_index=8))
        == 1
    )
    assert _read([]) == ()
    for events in (
        [_event(sequence=1)],
        [_event(), _event()],
        [_event(), _event(sequence=2)],
    ):
        with pytest.raises(ValueError, match="sequence"):
            _read(events)


def test_one_advance_contains_decision_submission_and_processing_fill() -> None:
    submission = _legacy(
        fill=False,
        event_type="submitted",
        previous_status="submitted",
        new_status="submitted",
        processing_index=6,
        timestamp_ns=1_699_996_400_000_000_000,
    )
    parsed = _read([submission, _event(sequence=1)])
    assert [event.exact_filled_quantity for event in parsed] == [
        Fraction(0),
        Fraction(3, 10),
    ]


def test_changed_processing_row_requires_its_own_lot_rule() -> None:
    event = _event(
        processing_index=8,
        timestamp_ns=1_700_003_600_000_000_000,
        lot_size=0.02,
        filled_lot_count=15,
    )
    context = dict(
        decision_index=7,
        decision_time_ns=1_700_000_000_000_000_000,
        processing_index=8,
        processing_time_ns=1_700_003_600_000_000_000,
    )
    assert _read([event], effective_lot_size=0.02, **context)[
        0
    ].exact_filled_quantity == Fraction(3, 10)
    with pytest.raises(ValueError, match="lot"):
        _read([event], **context)


@pytest.mark.parametrize(
    "context",
    [
        {"decision_index": -1},
        {"decision_index": True},
        {"processing_index": 9},
        {"decision_time_ns": True},
        {"processing_time_ns": 1_699_996_400_000_000_000},
    ],
)
def test_empty_batches_cannot_hide_invalid_interval_clocks(
    context: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _read([], **context)


@pytest.mark.parametrize(
    "event",
    [
        _event(processing_index=6, timestamp_ns=1_699_996_400_000_000_000),
        _legacy(fill=False, processing_index=6),
        _legacy(fill=False, processing_index=5),
        _legacy(fill=False, event_type="cancelled", new_status="eligible"),
        _legacy(fill=False, new_status="partially_filled"),
    ],
)
def test_interval_clock_pairs_and_event_statuses_cannot_be_swapped(
    event: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _read([event])


@pytest.mark.parametrize("lot", [None, True, -1.0, float("nan"), float("inf")])
def test_empty_batches_still_require_a_known_valid_lot_rule(lot: Any) -> None:
    with pytest.raises(ValueError):
        _read([], effective_lot_size=lot)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[null]",
        b"[",
        b"[NaN]",
        b"[1e999]",
        b"[\xff]",
        b'[{"sequence":0,"sequence":1}]',
    ],
)
def test_untrusted_json_is_strict_even_with_matching_raw_pin(raw: bytes) -> None:
    pin_type, _, reader = _api()
    pin = pin_type(raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        reader(raw, pin, _context())


def test_byte_pins_and_input_bounds_are_checked_before_parsing() -> None:
    pin_type, _, reader = _api()
    raw = b"invalid JSON"
    with pytest.raises(ValueError, match="pin"):
        reader(raw, pin_type(raw_bytes=len(raw), sha256="0" * 64), _context())
    with pytest.raises(ValueError, match="pin"):
        reader(
            raw,
            pin_type(raw_bytes=2, sha256=hashlib.sha256(raw).hexdigest()),
            _context(),
        )
    with pytest.raises(ValueError):
        pin_type(raw_bytes=1024 * 1024 + 1, sha256="0" * 64)
    with pytest.raises(ValueError):
        pin_type(raw_bytes=True, sha256="0" * 64)
    with pytest.raises(ValueError):
        _read([{}] * 4097)


def test_manufactured_native_canonical_events_remain_readable() -> None:
    from trade_rl.simulation.orders.model import OrderEvent

    for event, lot, exact in (
        (_event(reduce_only=True), 0.03, Fraction(3, 10)),
        (_legacy(), 0.0, Fraction(3, 10)),
        (_legacy(fill=False), 0.03, Fraction(0)),
    ):
        native = OrderEvent.from_mapping(event).canonical_payload()
        assert _read([native], effective_lot_size=lot)[0].exact_filled_quantity == exact
