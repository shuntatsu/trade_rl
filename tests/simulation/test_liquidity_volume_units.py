from __future__ import annotations

import pytest

from trade_rl.simulation.liquidity import (
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)


def test_explicit_processing_market_notional_preserves_quote_volume_semantics() -> None:
    request = LiquidityRequest(
        order_id="a" * 64,
        remaining_quantity=10.0,
        execution_price=100.0,
        available_volume_fraction=1.0,
        priority=LiquidityPriority.MARKET,
        eligible_index=1,
    )
    allocations, evidence = allocate_symbol_capacity(
        requests=(request,),
        processing_volume=1_000.0,
        processing_market_notional=1_000.0,
        price=100.0,
        contract_multiplier=5.0,
        participation_limit=0.1,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert evidence.market_notional == pytest.approx(1_000.0)
    assert evidence.initial_capacity_notional == pytest.approx(100.0)
    assert allocations[0].filled_notional == pytest.approx(100.0)

def test_native_quantity_pool_binds_fill_and_participation_at_lower_fill_price() -> None:
    request = LiquidityRequest(
        order_id="b" * 64,
        remaining_quantity=20.0,
        execution_price=90.0,
        available_volume_fraction=1.0,
        priority=LiquidityPriority.MARKET,
        eligible_index=1,
    )
    allocations, evidence = allocate_symbol_capacity(
        requests=(request,),
        processing_volume=10.0,
        processing_market_notional=1_000.0,
        processing_quantity_capacity=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert evidence.initial_capacity_notional == pytest.approx(1_000.0)
    assert allocations[0].filled_quantity == pytest.approx(10.0)
    assert allocations[0].filled_notional == pytest.approx(900.0)
    assert allocations[0].participation_rate == pytest.approx(1.0)


def test_trigger_fraction_and_shared_pool_constrain_native_quantity_capacity() -> None:
    requests = (
        LiquidityRequest(
            order_id="c" * 64,
            remaining_quantity=20.0,
            execution_price=90.0,
            available_volume_fraction=0.25,
            priority=LiquidityPriority.MARKET,
            eligible_index=1,
        ),
        LiquidityRequest(
            order_id="d" * 64,
            remaining_quantity=20.0,
            execution_price=90.0,
            available_volume_fraction=1.0,
            priority=LiquidityPriority.OLDER_LIMIT,
            eligible_index=1,
        ),
    )
    allocations, evidence = allocate_symbol_capacity(
        requests=requests,
        processing_volume=10.0,
        processing_market_notional=1_000.0,
        processing_quantity_capacity=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=1.0,
        lot_size=0.0,
        minimum_notional=0.0,
    )

    assert allocations[0].filled_quantity == pytest.approx(2.5)
    assert allocations[0].participation_rate == pytest.approx(0.25)
    assert allocations[1].filled_quantity == pytest.approx(7.5)
    assert allocations[1].participation_rate == pytest.approx(0.75)
    assert sum(abs(item.filled_quantity) for item in allocations) == pytest.approx(10.0)
    assert evidence.remaining_capacity_notional == pytest.approx(100.0)

