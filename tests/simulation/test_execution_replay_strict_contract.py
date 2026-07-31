from __future__ import annotations

from copy import deepcopy

import pytest

from trade_rl.simulation.execution_replay import validate_order_event_stream
from trade_rl.simulation.orders import OrderDomainError, OrderEvent, OrderStatus


def _event(
    *,
    sequence: int = 0,
    event_type: str = "submitted",
    previous_status: OrderStatus = OrderStatus.SUBMITTED,
    new_status: OrderStatus = OrderStatus.SUBMITTED,
    requested_quantity: float = 10.0,
    remaining_quantity: float = 10.0,
    filled_quantity: float = 0.0,
    filled_notional: float = 0.0,
    execution_price: float | None = None,
    processing_index: int = 0,
    reason: str | None = None,
) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=sequence,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id="d" * 64,
        execution_policy_digest="e" * 64,
        symbol_index=0,
        event_type=event_type,
        processing_index=processing_index,
        timestamp_ns=processing_index + 1,
        previous_status=previous_status,
        new_status=new_status,
        requested_quantity=requested_quantity,
        remaining_quantity=remaining_quantity,
        filled_quantity=filled_quantity,
        execution_price=execution_price,
        filled_notional=filled_notional,
        capacity_before=1000.0 if filled_quantity else 0.0,
        capacity_after=(
            1000.0 - filled_notional if filled_quantity else 0.0
        ),
        participation_rate=0.1 if filled_quantity else 0.0,
        trigger_segment="open" if filled_quantity else None,
        available_volume_fraction=1.0 if filled_quantity else 0.0,
        reason=reason,
        path_mode="conservative",
        path_points=(100.0, 101.0, 99.0, 100.5) if filled_quantity else (),
    )


def test_order_event_from_mapping_requires_exact_domain_fields() -> None:
    payload = {
        "schema_version": "order_event_v1",
        "dataset_id": "d" * 64,
        "execution_policy_digest": "e" * 64,
    }

    with pytest.raises(OrderDomainError, match="field closure"):
        OrderEvent.from_mapping(payload)


def test_order_event_from_mapping_round_trips_canonical_payload() -> None:
    event = _event()

    restored = OrderEvent.from_mapping(event.canonical_payload())

    assert restored == event
    assert restored.canonical_payload() == event.canonical_payload()


def test_order_event_from_mapping_rejects_boolean_integer() -> None:
    payload = deepcopy(_event().canonical_payload())
    payload["sequence"] = True

    with pytest.raises(OrderDomainError, match="sequence"):
        OrderEvent.from_mapping(payload)


def test_order_event_stream_requires_contiguous_global_sequence() -> None:
    events = (
        _event(sequence=0),
        _event(
            sequence=2,
            event_type="eligible",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            processing_index=1,
        ),
    )

    with pytest.raises(ValueError, match="contiguous"):
        validate_order_event_stream(events)


def test_order_event_stream_rejects_event_after_terminal_state() -> None:
    events = (
        _event(sequence=0),
        _event(
            sequence=1,
            event_type="eligible",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            processing_index=1,
        ),
        _event(
            sequence=2,
            event_type="filled",
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.FILLED,
            remaining_quantity=0.0,
            filled_quantity=10.0,
            filled_notional=1000.0,
            execution_price=100.0,
            processing_index=2,
            reason="filled",
        ),
        _event(
            sequence=3,
            event_type="no_fill",
            previous_status=OrderStatus.FILLED,
            new_status=OrderStatus.FILLED,
            remaining_quantity=0.0,
            processing_index=3,
        ),
    )

    with pytest.raises(ValueError, match="terminal"):
        validate_order_event_stream(events)


def test_order_event_stream_rejects_inconsistent_fill_arithmetic() -> None:
    events = (
        _event(sequence=0),
        _event(
            sequence=1,
            event_type="eligible",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            processing_index=1,
        ),
        _event(
            sequence=2,
            event_type="partial_fill",
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.PARTIALLY_FILLED,
            remaining_quantity=8.0,
            filled_quantity=3.0,
            filled_notional=300.0,
            execution_price=100.0,
            processing_index=2,
        ),
    )

    with pytest.raises(ValueError, match="remaining quantity"):
        validate_order_event_stream(events)
