from __future__ import annotations

import pytest

from trade_rl.simulation.liquidity import (
    LiquidityAllocationError,
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)


def _request(
    order_id: str,
    quantity: float,
    *,
    price: float = 100.0,
    fraction: float = 1.0,
    priority: LiquidityPriority = LiquidityPriority.MARKET,
    eligible_index: int = 1,
) -> LiquidityRequest:
    return LiquidityRequest(
        order_id=order_id if len(order_id) == 64 else order_id * 64,
        remaining_quantity=quantity,
        execution_price=price,
        available_volume_fraction=fraction,
        priority=priority,
        eligible_index=eligible_index,
    )


def test_allocator_uses_processing_bar_volume_and_shared_pool() -> None:
    allocations, evidence = allocate_symbol_capacity(
        requests=(
            _request("a", 8.0),
            _request(
                "b",
                8.0,
                fraction=0.5,
                priority=LiquidityPriority.OLDER_LIMIT,
            ),
        ),
        processing_volume=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=0.5,
        lot_size=1.0,
        minimum_notional=0.0,
    )

    assert evidence.initial_capacity_notional == pytest.approx(500.0)
    assert sum(item.filled_notional for item in allocations) <= 500.0
    assert evidence.consumed_capacity_notional == pytest.approx(500.0)
    assert evidence.remaining_capacity_notional == pytest.approx(0.0)
    assert allocations[0].order_id == "a" * 64
    assert allocations[0].filled_quantity == pytest.approx(5.0)
    assert allocations[1].filled_quantity == pytest.approx(0.0)


def test_trigger_fraction_caps_access_against_initial_symbol_pool() -> None:
    allocations, evidence = allocate_symbol_capacity(
        requests=(
            _request(
                "a",
                10.0,
                fraction=0.25,
                priority=LiquidityPriority.NEWLY_TRIGGERED_STOP,
            ),
        ),
        processing_volume=20.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=0.5,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert evidence.initial_capacity_notional == pytest.approx(1_000.0)
    assert allocations[0].accessible_capacity_notional == pytest.approx(250.0)
    assert allocations[0].filled_quantity == pytest.approx(2.5)
    assert allocations[0].filled_notional == pytest.approx(250.0)
    assert evidence.remaining_capacity_notional == pytest.approx(750.0)


def test_priority_is_deterministic_independent_of_input_order() -> None:
    newer_limit = _request(
        "f",
        3.0,
        priority=LiquidityPriority.NEWER_LIMIT,
        eligible_index=3,
    )
    triggered_stop = _request(
        "a",
        3.0,
        priority=LiquidityPriority.PREVIOUSLY_TRIGGERED_STOP,
        eligible_index=2,
    )
    market = _request(
        "b",
        3.0,
        priority=LiquidityPriority.MARKET,
        eligible_index=1,
    )

    allocations, _ = allocate_symbol_capacity(
        requests=(newer_limit, market, triggered_stop),
        processing_volume=4.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=1.0,
        minimum_notional=0.0,
    )

    assert [item.order_id for item in allocations] == [
        "a" * 64,
        "b" * 64,
        "f" * 64,
    ]
    assert [item.filled_quantity for item in allocations] == pytest.approx(
        [3.0, 1.0, 0.0]
    )


def test_same_priority_uses_eligible_index_then_order_id() -> None:
    allocations, _ = allocate_symbol_capacity(
        requests=(
            _request(
                "c",
                1.0,
                priority=LiquidityPriority.OLDER_LIMIT,
                eligible_index=2,
            ),
            _request(
                "b",
                1.0,
                priority=LiquidityPriority.OLDER_LIMIT,
                eligible_index=1,
            ),
            _request(
                "a",
                1.0,
                priority=LiquidityPriority.OLDER_LIMIT,
                eligible_index=1,
            ),
        ),
        processing_volume=3.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert [item.order_id for item in allocations] == ["a" * 64, "b" * 64, "c" * 64]


def test_lot_rounding_recomputes_exact_notional_and_preserves_direction() -> None:
    allocations, evidence = allocate_symbol_capacity(
        requests=(_request("a", -3.7, price=80.0),),
        processing_volume=10.0,
        price=100.0,
        contract_multiplier=2.0,
        participation_limit=0.5,
        lot_size=1.0,
        minimum_notional=0.0,
    )

    allocation = allocations[0]
    assert allocation.filled_quantity == pytest.approx(-3.0)
    assert allocation.filled_notional == pytest.approx(480.0)
    assert evidence.consumed_capacity_notional == pytest.approx(480.0)
    assert evidence.remaining_capacity_notional == pytest.approx(520.0)


def test_minimum_notional_and_zero_fraction_return_explicit_no_fill() -> None:
    allocations, _ = allocate_symbol_capacity(
        requests=(
            _request("a", 0.4, price=100.0),
            _request("b", 2.0, fraction=0.0),
        ),
        processing_volume=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.0,
        minimum_notional=50.0,
    )

    assert allocations[0].filled_quantity == 0.0
    assert allocations[0].no_fill_reason == "below_minimum_notional"
    assert allocations[1].filled_quantity == 0.0
    assert allocations[1].no_fill_reason == "zero_volume_fraction"


def test_zero_processing_volume_returns_no_capacity_without_error() -> None:
    allocations, evidence = allocate_symbol_capacity(
        requests=(_request("a", 2.0),),
        processing_volume=0.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=0.5,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert evidence.initial_capacity_notional == 0.0
    assert allocations[0].no_fill_reason == "no_capacity"
    assert allocations[0].filled_notional == 0.0


def test_allocator_never_overallocates_across_many_request_shapes() -> None:
    for count in range(1, 12):
        requests = tuple(
            _request(
                f"{index:064x}",
                (-1.0 if index % 2 else 1.0) * (index + 1.25),
                price=80.0 + index,
                fraction=(1.0, 0.5, 0.25, 0.0)[index % 4],
                priority=LiquidityPriority(index % len(LiquidityPriority)),
                eligible_index=index % 3,
            )
            for index in range(count)
        )
        allocations, evidence = allocate_symbol_capacity(
            requests=requests,
            processing_volume=17.0,
            price=100.0,
            contract_multiplier=1.5,
            participation_limit=0.2,
            lot_size=0.1,
            minimum_notional=5.0,
        )

        total = sum(item.filled_notional for item in allocations)
        assert total <= evidence.initial_capacity_notional + 1e-12
        assert evidence.consumed_capacity_notional == pytest.approx(total)
        assert evidence.remaining_capacity_notional >= -1e-12


def test_large_capacity_tolerates_subtraction_roundoff() -> None:
    price = 103.66558464909237
    allocations, evidence = allocate_symbol_capacity(
        requests=(_request("a", 0.1, price=price),),
        processing_volume=1_000_000.0,
        processing_market_notional=103_665_584.64909236,
        price=price,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert allocations[0].filled_quantity == pytest.approx(0.1)
    assert evidence.consumed_capacity_notional == pytest.approx(
        allocations[0].filled_notional
    )
    assert evidence.remaining_capacity_notional >= 0.0


def test_invalid_requests_and_capacity_inputs_fail_closed() -> None:
    with pytest.raises(LiquidityAllocationError, match="duplicate"):
        allocate_symbol_capacity(
            requests=(_request("a", 1.0), _request("a", 2.0)),
            processing_volume=1.0,
            price=100.0,
            contract_multiplier=1.0,
            participation_limit=1.0,
            lot_size=0.0,
            minimum_notional=0.0,
        )
    with pytest.raises(LiquidityAllocationError, match="fraction"):
        _request("a", 1.0, fraction=1.1)
    with pytest.raises(LiquidityAllocationError, match="participation_limit"):
        allocate_symbol_capacity(
            requests=(),
            processing_volume=1.0,
            price=100.0,
            contract_multiplier=1.0,
            participation_limit=0.0,
            lot_size=0.0,
            minimum_notional=0.0,
        )
