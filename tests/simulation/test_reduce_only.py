from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_order_admission import _book, _evaluate, _intent
from tests.simulation.test_stateful_execution import _executor, _market
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.simulation.liquidity import (
    LiquidityAllocationError,
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderDomainError,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
)


def _closing(quantity: float = -1.0, **overrides: object) -> OrderIntent:
    values = json.loads(canonical_json_bytes(_intent(quantity)))
    values.pop("order_id")
    values["order_type"] = OrderType.MARKET
    values["time_in_force"] = _intent(quantity).time_in_force
    values["reduce_only"] = True
    values.update(overrides)
    return OrderIntent.create(**values)


def test_reduce_only_identity_rejects_flag_tampering_and_reads_legacy() -> None:
    ordinary = _intent(-1.0)
    assert ordinary.order_id == (
        "e4403fef74bae1af0476d56358bb699e25b3f9d37eed320a57d81a3f13782c03"
    )
    closing = _closing()
    assert closing.order_id != ordinary.order_id
    legacy = json.loads(canonical_json_bytes(ordinary))
    legacy.pop("reduce_only", None)
    assert OrderIntent.from_mapping(legacy).reduce_only is False
    payload = json.loads(canonical_json_bytes(closing))
    assert OrderIntent.from_mapping(payload) == closing
    payload["reduce_only"] = False
    with pytest.raises(OrderDomainError, match="identity"):
        OrderIntent.from_mapping(payload)
    legacy["reduce_only"] = True
    with pytest.raises(OrderDomainError, match="identity"):
        OrderIntent.from_mapping(legacy)


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_reduce_only_flag_is_strict_boolean(invalid: object) -> None:
    with pytest.raises(OrderDomainError, match="boolean"):
        _closing(reduce_only=invalid)
    with pytest.raises(OrderDomainError, match="boolean"):
        replace(_intent(), reduce_only=invalid)
    payload = json.loads(canonical_json_bytes(_intent()))
    payload["reduce_only"] = invalid
    with pytest.raises(OrderDomainError, match="boolean"):
        OrderIntent.from_mapping(payload)


@pytest.mark.parametrize(
    "kind,prices",
    [
        (OrderType.LIMIT, {"limit_price": 100.0}),
        (OrderType.STOP_MARKET, {"stop_price": 100.0}),
    ],
)
def test_reduce_only_unsupported_types_fail_closed(
    kind: OrderType, prices: dict
) -> None:
    with pytest.raises(OrderDomainError, match="MARKET"):
        _closing(order_type=kind, **prices)


@pytest.mark.parametrize(
    "position,quantity,reason",
    [
        (0.0, -1.0, "reduce_only_no_position"),
        (1.0, 1.0, "reduce_only_wrong_direction"),
        (-1.0, -1.0, "reduce_only_wrong_direction"),
        (1.0, -1.0000000000001, "reduce_only_exceeds_position"),
        (-1.0, 1.1, "reduce_only_exceeds_position"),
        (1.0, -1.0, None),
        (-1.0, 1.0, None),
    ],
)
def test_reduce_only_admission_uses_strict_inventory_bounds(
    position: float, quantity: float, reason: str | None
) -> None:
    decision = _evaluate(_closing(quantity), book=_book(quantity=position))
    assert decision.accepted is (reason is None)
    assert decision.reason == reason


def test_reduce_only_retains_notional_and_side_restrictions() -> None:
    intent = _closing(-0.4)
    assert _evaluate(intent, book=_book(quantity=0.4), minimum_notional=50).reason == (
        "below_minimum_notional"
    )
    assert _evaluate(intent, book=_book(quantity=0.4), sell_allowed=False).reason == (
        "sell_disabled"
    )


@pytest.mark.parametrize(
    "position,quantity,reason",
    [
        (0.0, -0.5, "reduce_only_no_position"),
        (1.0, -1.5, "reduce_only_exceeds_position"),
    ],
)
def test_pending_opening_is_not_actual_inventory_for_admission(
    position: float, quantity: float, reason: str
) -> None:
    executor = _executor(_market(), max_participation_rate=1.0, lot_size=0.1)
    common = {"execution_policy_digest": executor.execution_policy_digest}
    opening = _closing(
        1.0,
        reduce_only=False,
        order_type=OrderType.LIMIT,
        limit_price=80.0,
        eligible_index=0,
        **common,
    )
    closing = _closing(quantity, **common)
    result = executor.execute_orders(
        _book(quantity=position),
        OrderBookState.empty(),
        (opening, closing),
        start_index=0,
        bars=1,
    )
    assert result.fill_count == 0
    assert result.book.quantities.tolist() == [position]
    closed = next(
        order
        for order in result.order_book.terminal_orders
        if order.order_id == closing.order_id
    )
    assert closed.status is OrderStatus.REJECTED
    assert closed.terminal_reason == reason


@pytest.mark.parametrize("volume,remaining", [(1000.0, 99950.0), (0.5, 0.0)])
def test_margin_flatten_between_allocations_cannot_reopen_with_reduce_only(
    volume: float, remaining: float
) -> None:
    executor = _executor(
        _market(volume=np.full((6, 1), volume)),
        max_participation_rate=1.0,
        lot_size=0.1,
        fee_rate=0.001,
        maintenance_margin_rate=0.25,
        collateral_haircut=0.1,
    )
    common = {"execution_policy_digest": executor.execution_policy_digest}
    opening = _closing(0.5, reduce_only=False, eligible_index=0, **common)
    closing = _closing(-0.5, **common)
    result = executor.execute_orders(
        _book(quantity=0.5, capital=100.0),
        OrderBookState.empty(),
        (opening, closing),
        start_index=0,
        bars=1,
    )
    assert result.book.insolvent
    assert result.book.exact_quantities == (Fraction(0),)
    assert result.fill_count == 1
    assert result.interval_cost == pytest.approx(0.05)
    assert not result.order_book.active_orders
    assert all(
        event.filled_quantity == 0 for event in result.order_events if event.reduce_only
    )
    no_fill = next(
        event for event in result.order_events if event.event_type == "no_fill"
    )
    assert no_fill.reason == "reduce_only_exhausted"
    assert no_fill.capacity_before == no_fill.capacity_after == remaining
    assert result.capacity_evidence[0].consumed_capacity_notional == 50.0
    assert result.capacity_evidence[0].remaining_capacity_notional == remaining


def _request(
    name: str,
    quantity: float,
    *,
    reduce_only: bool = True,
    priority: LiquidityPriority = LiquidityPriority.MARKET,
) -> LiquidityRequest:
    return LiquidityRequest(
        order_id=name * 64,
        remaining_quantity=quantity,
        execution_price=100.0,
        available_volume_fraction=1.0,
        priority=priority,
        eligible_index=1,
        reduce_only=reduce_only,
    )


def _allocate(requests: tuple, position: Fraction | None, **overrides: object):
    values = dict(
        processing_volume=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.1,
        minimum_notional=0.0,
        initial_position=position,
    )
    values.update(overrides)
    return allocate_symbol_capacity(requests, **values)


def test_reduce_only_allocator_requires_exact_inventory() -> None:
    with pytest.raises(LiquidityAllocationError, match="initial_position"):
        _allocate((_request("a", -1.0),), None)


@pytest.mark.parametrize("direction", [-1, 1])
def test_actual_priority_clips_competing_closes_and_preserves_capacity(
    direction: int,
) -> None:
    requests = (
        _request("a", direction * 2.0),
        _request("b", direction * 2.0),
        _request(
            "c",
            -direction * 1.0,
            reduce_only=False,
            priority=LiquidityPriority.OLDER_LIMIT,
        ),
    )
    fills, capacity = _allocate(tuple(reversed(requests)), Fraction(-direction))
    assert [fill.filled_quantity for fill in fills] == [
        direction * 1.0,
        0.0,
        -direction * 1.0,
    ]
    assert fills[0].filled_lot_count == direction * 10
    assert fills[0].reduce_only_exhausted
    assert fills[1].no_fill_reason == "reduce_only_exhausted"
    assert capacity.consumed_capacity_notional == 200.0
    assert capacity.remaining_capacity_notional == 800.0


def test_ordinary_fill_before_closing_is_part_of_inventory_authority() -> None:
    fills, _ = _allocate(
        (
            _request(
                "a",
                -2.0,
                reduce_only=False,
                priority=LiquidityPriority.PREVIOUSLY_TRIGGERED_STOP,
            ),
            _request("b", -1.0),
        ),
        Fraction(1),
    )
    assert [fill.filled_quantity for fill in fills] == [-2.0, 0.0]
    assert fills[1].no_fill_reason == "reduce_only_exhausted"


@pytest.mark.parametrize(
    "position,lot,expected,count",
    [
        (Fraction("0.19999999999999"), 0.1, -0.1, -1),
        (Fraction("0.09999999999999"), 0.1, 0.0, None),
        (Fraction("0.3"), 0.1, -0.3, -3),
        (Fraction("0.3"), 0.0, -0.3, None),
    ],
)
def test_closing_clip_does_not_promote_true_sublots(
    position: Fraction, lot: float, expected: float, count: int | None
) -> None:
    fills, _ = _allocate((_request("a", -1.0),), position, lot_size=lot)
    assert fills[0].filled_quantity == expected
    assert fills[0].filled_lot_count == count


def test_clipped_exit_still_requires_minimum_notional_at_allocation() -> None:
    fills, capacity = _allocate(
        (_request("a", -2.0),), Fraction("0.4"), minimum_notional=50.0
    )
    assert fills[0].no_fill_reason == "below_minimum_notional"
    assert capacity.consumed_capacity_notional == 0.0


def test_stateful_close_expires_excess_before_later_ordinary_fill() -> None:
    executor = _executor(
        _market(), max_participation_rate=1.0, lot_size=0.1, fee_rate=0.001
    )
    closing = _closing(-2.0, execution_policy_digest=executor.execution_policy_digest)
    opening = _closing(
        1.0,
        reduce_only=False,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        eligible_index=0,
        execution_policy_digest=executor.execution_policy_digest,
    )
    earlier_sell = _closing(
        -1.0,
        reduce_only=False,
        eligible_index=0,
        execution_policy_digest=executor.execution_policy_digest,
    )
    result = executor.execute_orders(
        _book(quantity=2.0),
        OrderBookState.empty(),
        (closing, opening, earlier_sell),
        start_index=0,
        bars=1,
    )
    assert result.book.exact_quantities == (Fraction(1),)
    assert result.book.cash == pytest.approx(899.7)
    assert result.interval_cost == pytest.approx(0.3)
    assert result.fill_count == 3
    assert result.expired_count == 1
    assert not result.order_book.active_orders
    closed = next(
        order
        for order in result.order_book.terminal_orders
        if order.order_id == closing.order_id
    )
    assert closed.status is OrderStatus.EXPIRED
    assert closed.remaining_quantity == -1.0
    assert closed.terminal_reason == "reduce_only_exhausted"
    for event in result.order_events:
        assert event.reduce_only is (event.order_id == closing.order_id)
        restored = OrderEvent.from_mapping(
            json.loads(canonical_json_bytes(event.canonical_payload()))
        )
        assert restored == event
    assert result.capacity_evidence[0].consumed_capacity_notional == 300.0


def test_competing_stateful_closes_expire_without_charging_unfilled_quantity() -> None:
    executor = _executor(
        _market(), max_participation_rate=1.0, lot_size=0.1, fee_rate=0.001
    )
    common = {"execution_policy_digest": executor.execution_policy_digest}
    closing = tuple(
        _closing(-1.0, target_identity=name, **common) for name in ("a", "b")
    )
    opening = _closing(
        2.0,
        reduce_only=False,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        eligible_index=0,
        **common,
    )
    result = executor.execute_orders(
        _book(quantity=1.0),
        OrderBookState.empty(),
        (*closing, opening),
        start_index=0,
        bars=1,
    )
    assert result.book.exact_quantities == (Fraction(2),)
    assert result.book.cash == pytest.approx(799.7)
    assert result.interval_cost == pytest.approx(0.3)
    assert result.fill_count == 2
    assert result.expired_count == 1
    assert not result.order_book.active_orders
    closing_events = [event for event in result.order_events if event.reduce_only]
    assert sum(event.filled_quantity for event in closing_events) == -1.0
    no_fill = next(event for event in closing_events if event.event_type == "no_fill")
    assert no_fill.reason == "reduce_only_exhausted"
    assert no_fill.capacity_before == no_fill.capacity_after == 99900.0
    assert result.capacity_evidence[0].consumed_capacity_notional == 300.0


@pytest.mark.parametrize("direction", [-1, 1])
def test_partial_reduce_only_restart_closes_without_reopening(direction: int) -> None:
    executor = _executor(
        _market(volume=np.ones((6, 1))), max_participation_rate=1.0, lot_size=0.1
    )
    intent = _closing(
        direction * 2.0, execution_policy_digest=executor.execution_policy_digest
    )
    first = executor.execute_orders(
        _book(quantity=-direction * 2.0),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    pending = PendingOrder.from_mapping(
        json.loads(canonical_json_bytes(first.order_book.active_orders[0]))
    )
    assert pending.intent.reduce_only
    second = executor.execute_orders(
        first.book.clone(), OrderBookState((pending,), ()), (), start_index=1, bars=1
    )
    assert second.book.exact_quantities == (Fraction(0),)
    assert not second.order_book.active_orders
    assert second.order_book.terminal_orders[-1].status is OrderStatus.FILLED
    # Another actor has already closed the position before a resumed partial.
    stale = executor.execute_orders(
        _book(), OrderBookState((pending,), ()), (), start_index=1, bars=1
    )
    assert stale.fill_count == 0
    assert stale.book.exact_quantities == (Fraction(0),)
    assert stale.order_book.terminal_orders[-1].status is OrderStatus.EXPIRED
