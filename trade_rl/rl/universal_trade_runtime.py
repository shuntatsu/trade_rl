"""Read-only Universal Trade RL runtime state for the U1 policy contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UniversalTradeRuntimeSnapshot:
    """Scalar causal state projected from the maintained single-symbol runtime."""

    policy_requested_weight: float
    pending_target_weight: float
    pending_target_active: bool
    risk_projected_weight: float
    current_weight: float
    previous_action: float
    fill_ratio: float
    unfilled_turnover_ratio: float
    participation_ratio: float
    execution_cost_rate: float
    position_age_hours: float
    pending_notional_ratio: float
    pending_order_type_code: float
    pending_order_status_code: float
    pending_order_age_hours: float
    pending_order_eligible_delay_hours: float
    pending_order_triggered: bool
    pending_order_expiry_distance_hours: float
    asset_active: bool
    tradable: bool
    borrow_available: bool
    borrow_rate: float
    mark_index_basis: float
    current_drawdown: float
    current_gross_exposure: float
    current_net_exposure: float
    cash_weight: float
    risk_scale: float
    margin_utilization: float


__all__ = ["UniversalTradeRuntimeSnapshot"]
