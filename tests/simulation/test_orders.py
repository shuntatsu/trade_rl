from __future__ import annotations

import hashlib

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderDomainError,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
    execution_policy_digest,
)


def _intent(**overrides: object) -> OrderIntent:
    values: dict[str, object] = {
        "dataset_id": "d" * 64,
        "target_identity": "target-1",
        "execution_policy_digest": "e" * 64,
        "symbol_index": 0,
        "requested_quantity": 10.0,
        "order_type": OrderType.LIMIT,
        "time_in_force": TimeInForce.GTC,
        "limit_price": 99.0,
        "stop_price": None,
        "submit_index": 4,
        "eligible_index": 5,
        "expiry_index": None,
        "submission_reference_price": 100.0,
        "decision_equity": 1_000.0,
        "replaced_order_id": None,
    }
    values.update(overrides)
    return OrderIntent.create(**values)


def test_order_intent_id_is_deterministic_and_identity_bound() -> None:
    first = _intent()
    second = _intent()
    changed = _intent(target_identity="target-2")

    assert first.order_id == second.order_id
    assert first.order_id != changed.order_id
    assert len(first.order_id) == 64


def test_order_intent_rejects_invalid_type_specific_prices() -> None:
    with pytest.raises(OrderDomainError, match="limit_price"):
        _intent(limit_price=None)
    with pytest.raises(OrderDomainError, match="stop_price"):
        _intent(order_type=OrderType.STOP_MARKET, limit_price=None, stop_price=None)
    with pytest.raises(OrderDomainError, match="market orders"):
        _intent(order_type=OrderType.MARKET, limit_price=99.0)


def test_order_intent_rejects_noncausal_or_zero_inputs() -> None:
    with pytest.raises(OrderDomainError, match="non-zero"):
        _intent(requested_quantity=0.0)
    with pytest.raises(OrderDomainError, match="eligible_index"):
        _intent(eligible_index=3)
    with pytest.raises(OrderDomainError, match="expiry_index"):
        _intent(expiry_index=4)
    with pytest.raises(OrderDomainError, match="decision_equity"):
        _intent(decision_equity=float("nan"))


def test_pending_order_preserves_quantity_identity_across_partial_and_full_fill() -> None:
    pending = PendingOrder.from_intent(_intent())
    partial = pending.apply_fill(quantity=4.0, notional=396.0, processing_index=5)

    assert partial.remaining_quantity == pytest.approx(6.0)
    assert partial.cumulative_filled_quantity == pytest.approx(4.0)
    assert partial.cumulative_filled_notional == pytest.approx(396.0)
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    assert partial.evidence_version == 1

    filled = partial.apply_fill(quantity=6.0, notional=594.0, processing_index=6)
    assert filled.remaining_quantity == pytest.approx(0.0)
    assert filled.cumulative_filled_quantity == pytest.approx(10.0)
    assert filled.status is OrderStatus.FILLED
    assert filled.terminal_reason == "filled"


def test_pending_order_rejects_overfill_wrong_direction_and_terminal_mutation() -> None:
    pending = PendingOrder.from_intent(_intent())
    with pytest.raises(OrderDomainError, match="direction"):
        pending.apply_fill(quantity=-1.0, notional=99.0, processing_index=5)
    with pytest.raises(OrderDomainError, match="remaining"):
        pending.apply_fill(quantity=11.0, notional=1_089.0, processing_index=5)

    filled = pending.apply_fill(quantity=10.0, notional=990.0, processing_index=5)
    with pytest.raises(OrderDomainError, match="terminal"):
        filled.cancel(processing_index=6, reason="replace")


def test_pending_order_explicit_transitions_are_monotonic() -> None:
    pending = PendingOrder.from_intent(_intent(eligible_index=7))
    waiting = pending.mark_latency_wait(processing_index=5)
    eligible = waiting.mark_eligible(processing_index=7)
    triggered = eligible.mark_triggered(processing_index=7)
    expired = triggered.expire(processing_index=8, reason="day_expired")

    assert waiting.status is OrderStatus.LATENCY_WAIT
    assert eligible.status is OrderStatus.ELIGIBLE
    assert triggered.status is OrderStatus.TRIGGERED
    assert triggered.trigger_index == 7
    assert expired.status is OrderStatus.EXPIRED
    assert expired.terminal_reason == "day_expired"
    assert expired.evidence_version == 4


def test_order_book_rejects_duplicate_active_ids_and_aggregates_residuals() -> None:
    first = PendingOrder.from_intent(_intent())
    second = PendingOrder.from_intent(
        _intent(symbol_index=1, requested_quantity=-3.0, target_identity="target-2")
    )
    state = OrderBookState(active_orders=(first, second), terminal_orders=())

    assert state.active_for_symbol(0) == (first,)
    np.testing.assert_allclose(state.active_remaining_quantities(2), [10.0, -3.0])

    with pytest.raises(OrderDomainError, match="duplicate"):
        OrderBookState(active_orders=(first, first), terminal_orders=())


def test_order_book_replace_moves_terminal_orders_and_rejects_unknown_id() -> None:
    first = PendingOrder.from_intent(_intent())
    state = OrderBookState(active_orders=(first,), terminal_orders=())
    replaced = state.replace(first.cancel(processing_index=6, reason="superseded"))

    assert replaced.active_orders == ()
    assert replaced.terminal_orders[0].status is OrderStatus.CANCELLED
    with pytest.raises(OrderDomainError, match="unknown"):
        replaced.replace(first)


def test_order_event_payload_is_canonical_and_digest_helper_matches_codec() -> None:
    event = OrderEvent(
        schema_version="order_event_v1",
        sequence=2,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id="d" * 64,
        execution_policy_digest="e" * 64,
        symbol_index=0,
        event_type="partial_fill",
        processing_index=5,
        timestamp_ns=123,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=OrderStatus.PARTIALLY_FILLED,
        requested_quantity=10.0,
        remaining_quantity=6.0,
        filled_quantity=4.0,
        execution_price=99.0,
        filled_notional=396.0,
        capacity_before=500.0,
        capacity_after=104.0,
        participation_rate=0.4,
        trigger_segment="first_extreme",
        available_volume_fraction=0.5,
        reason=None,
        path_mode="conservative",
        path_points=(100.0, 105.0, 95.0, 101.0),
    )
    payload = event.canonical_payload()

    assert payload["previous_status"] == "eligible"
    assert payload["path_points"] == [100.0, 105.0, 95.0, 101.0]
    expected = hashlib.sha256(canonical_json_bytes({"a": 1, "b": [2, 3]})).hexdigest()
    assert execution_policy_digest({"b": [2, 3], "a": 1}) == expected
