"""Explicit aggregation boundary for invocation-local order-event batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from trade_rl.simulation.execution_replay import validate_order_event_stream
from trade_rl.simulation.orders import OrderEvent


def merge_order_event_batches(
    batches: Sequence[Sequence[OrderEvent]],
) -> tuple[OrderEvent, ...]:
    """Validate local sequences, resequence globally, and validate the full history."""

    merged: list[OrderEvent] = []
    for batch_index, batch in enumerate(batches):
        normalized_batch = tuple(batch)
        local_sequence = tuple(event.sequence for event in normalized_batch)
        if local_sequence != tuple(range(len(normalized_batch))):
            raise ValueError(
                f"order event batch {batch_index} sequence must be contiguous from zero"
            )
        base_sequence = len(merged)
        merged.extend(
            replace(event, sequence=base_sequence + offset)
            for offset, event in enumerate(normalized_batch)
        )
    if not merged:
        return ()
    return validate_order_event_stream(merged)


__all__ = ["merge_order_event_batches"]
