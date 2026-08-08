from __future__ import annotations

import pytest

from trade_rl.simulation.execution_replay import validate_order_event_stream
from trade_rl.simulation.orders import OrderEvent, OrderStatus


def _event(
    *,
    sequence: int,
    event_type: str,
    previous_status: OrderStatus,
    new_status: OrderStatus,
    remaining_quantity: float,
    filled_quantity: float = 0.0,
    filled_notional: float = 0.0,
    execution_price: float | None = None,
    processing_index: int,
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
        requested_quantity=10.0,
        remaining_quantity=remaining_quantity,
        filled_quantity=filled_quantity,
        execution_price=execution_price,
        filled_notional=filled_notional,
        capacity_before=1_000.0 if filled_quantity else 0.0,
        capacity_after=50.0 if filled_quantity else 0.0,
        participation_rate=0.1 if filled_quantity else 0.0,
        trigger_segment="open" if filled_quantity else None,
        available_volume_fraction=1.0 if filled_quantity else 0.0,
        reason=reason,
        path_mode="conservative",
        path_points=(100.0, 101.0, 99.0, 100.5) if filled_quantity else (),
    )


def test_order_event_stream_rejects_rejection_after_partial_fill() -> None:
    events = (
        _event(
            sequence=0,
            event_type="submitted",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.SUBMITTED,
            remaining_quantity=10.0,
            processing_index=0,
        ),
        _event(
            sequence=1,
            event_type="eligible",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            remaining_quantity=10.0,
            processing_index=1,
        ),
        _event(
            sequence=2,
            event_type="partial_fill",
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.PARTIALLY_FILLED,
            remaining_quantity=0.5,
            filled_quantity=9.5,
            filled_notional=950.0,
            execution_price=100.0,
            processing_index=1,
        ),
        _event(
            sequence=3,
            event_type="rejected",
            previous_status=OrderStatus.PARTIALLY_FILLED,
            new_status=OrderStatus.REJECTED,
            remaining_quantity=0.5,
            processing_index=2,
            reason="below_lot_size",
        ),
    )

    with pytest.raises(ValueError, match="event transition"):
        validate_order_event_stream(events)
