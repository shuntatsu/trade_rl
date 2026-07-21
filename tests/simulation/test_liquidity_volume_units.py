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
