from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_stateful_execution import (
    _executor,
    _intent,
    _market,
    _zero_book,
)
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderDomainError,
    OrderEvent,
)


@pytest.mark.parametrize("direction", [1, -1])
def test_partial_fills_publish_the_accepted_signed_lots(direction: int) -> None:
    dataset = _market(
        lot_size=np.full((6, 1), 0.1),
        volume=np.array([[1.0], [0.1], [0.2], [1.0], [1.0], [1.0]]),
    )
    executor = _executor(dataset, max_participation_rate=1.0)
    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, direction * 0.3),),
        start_index=0,
        bars=2,
    )
    fills = [
        event
        for event in result.order_events
        if event.event_type in {"partial_fill", "filled"}
    ]
    assert [event.processing_index for event in fills] == [1, 2]
    assert result.book.exact_quantities == (direction * Fraction(3, 10),)
    assert [getattr(event, "filled_lot_count", None) for event in fills] == [
        direction,
        direction * 2,
    ]
    payloads = [event.canonical_payload() for event in fills]
    assert all(payload["schema_version"] == "order_event_v2" for payload in payloads)
    assert (
        sum(
            Fraction(str(payload["lot_size"])) * payload["filled_lot_count"]
            for payload in payloads
        )
        == result.book.exact_quantities[0]
    )
    assert [
        OrderEvent.from_mapping(payload).canonical_payload() for payload in payloads
    ] == payloads


def _legacy_payload() -> dict[str, object]:
    return {
        "schema_version": "order_event_v1",
        "sequence": 0,
        "order_id": "a" * 64,
        "replaced_order_id": None,
        "dataset_id": "d" * 64,
        "execution_policy_digest": "e" * 64,
        "symbol_index": 0,
        "event_type": "filled",
        "processing_index": 1,
        "timestamp_ns": 1,
        "previous_status": "eligible",
        "new_status": "filled",
        "requested_quantity": 0.3,
        "remaining_quantity": 0.0,
        "filled_quantity": 0.3,
        "execution_price": 100.0,
        "filled_notional": 30.0,
        "capacity_before": 100.0,
        "capacity_after": 70.0,
        "participation_rate": 0.3,
        "trigger_segment": "open",
        "available_volume_fraction": 1.0,
        "reason": "filled",
        "path_mode": "conservative",
        "path_points": [100.0, 110.0, 90.0, 100.0],
    }


def test_legacy_order_event_mapping_remains_byte_semantically_unchanged() -> None:
    legacy = _legacy_payload()
    assert OrderEvent.from_mapping(legacy).canonical_payload() == legacy


@pytest.mark.parametrize(
    "mutation",
    [
        {"filled_lot_count": True},
        {"filled_lot_count": 3.0},
        {"filled_lot_count": 0},
        {"filled_lot_count": -3},
        {"filled_lot_count": 4},
        {"lot_size": 0.0},
        {"lot_size": True},
        {"lot_size": float("nan")},
        {"event_type": "submitted"},
    ],
)
def test_v2_rejects_invalid_exact_lot_evidence(mutation: dict[str, object]) -> None:
    valid = {
        **_legacy_payload(),
        "schema_version": "order_event_v2",
        "filled_lot_count": 3,
        "lot_size": 0.1,
    }
    assert OrderEvent.from_mapping(valid).canonical_payload() == valid
    with pytest.raises(ValueError):
        OrderEvent.from_mapping({**valid, **mutation})


@pytest.mark.parametrize("missing", ["filled_lot_count", "lot_size"])
def test_v2_requires_both_exact_lot_fields(missing: str) -> None:
    valid = {
        **_legacy_payload(),
        "schema_version": "order_event_v2",
        "filled_lot_count": 3,
        "lot_size": 0.1,
    }
    assert OrderEvent.from_mapping(valid).canonical_payload() == valid
    del valid[missing]
    with pytest.raises(ValueError):
        OrderEvent.from_mapping(valid)


@pytest.mark.parametrize("direction", [1, -1])
def test_v2_preserves_distinct_lots_with_the_same_float_projection(
    direction: int,
) -> None:
    counts = [direction * 10**18, direction * (10**18 + 1)]
    events = [
        OrderEvent.from_mapping(
            {
                **_legacy_payload(),
                "schema_version": "order_event_v2",
                "requested_quantity": direction * 2.0,
                "filled_quantity": direction * 1.0,
                "filled_lot_count": count,
                "lot_size": 1e-18,
            }
        )
        for count in counts
    ]
    assert events[0].filled_quantity == events[1].filled_quantity
    assert [event.canonical_payload()["filled_lot_count"] for event in events] == counts
    assert events[0].canonical_payload() != events[1].canonical_payload()


def test_unquantized_execution_retains_the_legacy_event_shape() -> None:
    dataset = _market()
    executor = _executor(dataset)
    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 0.3),),
        start_index=0,
        bars=1,
    )
    assert any(event.event_type == "filled" for event in result.order_events)
    for event in result.order_events:
        payload = event.canonical_payload()
        assert payload["schema_version"] == "order_event_v1"
        assert "filled_lot_count" not in payload and "lot_size" not in payload
        assert OrderEvent.from_mapping(payload).canonical_payload() == payload


@pytest.mark.parametrize("metadata", [{"filled_lot_count": 3}, {"lot_size": 0.1}])
def test_legacy_event_rejects_unversioned_lot_evidence(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(OrderDomainError):
        OrderEvent.from_mapping({**_legacy_payload(), **metadata})
    with pytest.raises(OrderDomainError):
        replace(OrderEvent.from_mapping(_legacy_payload()), **metadata)
