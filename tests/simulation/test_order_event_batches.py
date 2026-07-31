from __future__ import annotations

import pytest

from trade_rl.simulation.order_event_batches import merge_order_event_batches
from trade_rl.simulation.orders import OrderEvent, OrderStatus


def _submitted_event(*, order_id: str, sequence: int) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=sequence,
        order_id=order_id,
        replaced_order_id=None,
        dataset_id="d" * 64,
        execution_policy_digest="e" * 64,
        symbol_index=0,
        event_type="submitted",
        processing_index=0,
        timestamp_ns=1,
        previous_status=OrderStatus.SUBMITTED,
        new_status=OrderStatus.SUBMITTED,
        requested_quantity=1.0,
        remaining_quantity=1.0,
        filled_quantity=0.0,
        execution_price=None,
        filled_notional=0.0,
        capacity_before=0.0,
        capacity_after=0.0,
        participation_rate=0.0,
        trigger_segment=None,
        available_volume_fraction=0.0,
        reason=None,
        path_mode="conservative",
        path_points=(),
    )


def test_merge_order_event_batches_resequences_invocation_local_events() -> None:
    first = (_submitted_event(order_id="a" * 64, sequence=0),)
    second = (_submitted_event(order_id="b" * 64, sequence=0),)

    merged = merge_order_event_batches((first, second))

    assert tuple(event.sequence for event in merged) == (0, 1)
    assert tuple(event.order_id for event in merged) == ("a" * 64, "b" * 64)


def test_merge_order_event_batches_rejects_noncanonical_local_sequence() -> None:
    invalid = (_submitted_event(order_id="a" * 64, sequence=1),)

    with pytest.raises(ValueError, match="batch 0 sequence"):
        merge_order_event_batches((invalid,))


def test_merge_order_event_batches_accepts_no_invocations() -> None:
    assert merge_order_event_batches(()) == ()
