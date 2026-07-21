from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.order_admission import OrderAdmissionPolicy
from trade_rl.simulation.orders import OrderIntent, OrderType, TimeInForce


def test_partial_order_admission_projects_only_remaining_quantity() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="target-10",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=10.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    book = BookState(
        quantities=np.array([2.0]),
        cash=800.0,
        mark_prices=np.array([100.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1.0]),
    )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    )

    decision = policy.evaluate(
        intent,
        remaining_quantity=8.0,
        book=book,
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.01,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([100.0]),
    )

    assert decision.accepted
    assert decision.admitted_quantity == pytest.approx(8.0)
