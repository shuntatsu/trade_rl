from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.conformance_probe import (
    run_child_order_sequence_execution_probe,
)
from trade_rl.simulation.execution_canonicalization import compare_dual_shadow_execution
from trade_rl.simulation.legacy_execution_probe import (
    run_legacy_child_order_sequence_probe,
)
from trade_rl.simulation.target_exposure_controller import (
    TargetExposureChildOrder,
    TargetExposureController,
    TargetExposureInput,
)


def _same_side_target_change_orders() -> tuple[TargetExposureChildOrder, ...]:
    controller = TargetExposureController(no_trade_band=0.0)
    realized_quantity = 0.0
    child_orders: list[TargetExposureChildOrder] = []
    for target_exposure in (0.1, 0.2, 0.05, 0.0):
        plan = controller.plan(
            TargetExposureInput(
                target_exposure=target_exposure,
                allocated_equity=1_000.0,
                reference_price=100.0,
                contract_multiplier=1.0,
                realized_quantity=realized_quantity,
                working_remaining_quantities=(),
            )
        )
        assert plan.cancel_working_orders is False
        assert plan.child_order is not None
        child_orders.append(plan.child_order)
        realized_quantity += plan.child_order.quantity
    assert realized_quantity == pytest.approx(0.0)
    return tuple(child_orders)


@pytest.mark.nautilus
def test_same_side_target_changes_have_exact_dual_shadow_parity() -> None:
    child_orders = _same_side_target_change_orders()
    assert [order.quantity for order in child_orders] == pytest.approx(
        [1.0, 1.0, -1.5, -0.5]
    )
    assert [order.reduce_only for order in child_orders] == [False, False, True, True]

    legacy = run_legacy_child_order_sequence_probe(child_orders)
    candidate = run_child_order_sequence_execution_probe(
        child_orders,
        starting_balance=Decimal("1000"),
    )

    assert [fill.quantity_lots for fill in candidate.fills] == [1000, 1000, -1500, -500]
    assert [fill.position_lots for fill in candidate.fills] == [1000, 2000, 500, 0]

    report = compare_dual_shadow_execution(
        legacy_fills=legacy.fills,
        candidate_fills=candidate.fills,
        legacy_economics=legacy.economics,
        candidate_economics=candidate.economics,
    )
    assert report.fill_parity is True, (
        report.mismatches,
        legacy.fills,
        candidate.fills,
    )
    economics = (
        f"fee={legacy.economics.fee_minor}/{candidate.economics.fee_minor} "
        f"pnl={legacy.economics.realized_pnl_minor}/"
        f"{candidate.economics.realized_pnl_minor} "
        f"equity={legacy.economics.final_equity_minor}/"
        f"{candidate.economics.final_equity_minor}"
    )
    assert report.economic_parity is True, economics
    assert report.exact_parity is True, report.mismatches
