"""Map framework-neutral target-exposure plans into Nautilus order commands."""

from __future__ import annotations

from typing import Any

from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime
from trade_rl.simulation.target_exposure_controller import TargetExposurePlan


def submit_target_exposure_plan(
    *,
    strategy: Any,
    instrument: Any,
    plan: TargetExposurePlan,
) -> Any | None:
    """Submit at most one next child order for a controller plan.

    Cancellation and submission are deliberately separate phases. A stale working
    generation is cancelled first; the caller must observe terminal state and call
    the controller again before any replacement is submitted.
    """

    require_nautilus_runtime()
    if plan.cancel_working_orders:
        if plan.child_order is not None:
            raise ValueError("cancel plan must not submit a replacement child order")
        strategy.cancel_all_orders(instrument.id)
        return None

    child = plan.child_order
    if child is None:
        return None
    if child.quantity == 0.0:
        raise ValueError("child order quantity must be non-zero")

    from nautilus_trader.model.enums import OrderSide, TimeInForce

    side = OrderSide.BUY if child.quantity > 0.0 else OrderSide.SELL
    order = strategy.order_factory.market(
        instrument_id=instrument.id,
        order_side=side,
        quantity=instrument.make_qty(abs(child.quantity)),
        time_in_force=TimeInForce.IOC,
        reduce_only=child.reduce_only,
    )
    strategy.submit_order(order)
    return order


__all__ = ["submit_target_exposure_plan"]
