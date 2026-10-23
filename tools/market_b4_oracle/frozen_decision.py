"""Frozen after-cost, action, and risk projection primitives.

This module composes caller-owned assumptions and grants no identity, account,
execution, admission, or research authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.market_b4_oracle.allocation_choice import (
    AllocationAssumptions,
    choose_after_cost_weight,
)

_SCOPE = "SINGLE_ACTIVE_SYMBOL_FROZEN_H_RISK_ARITHMETIC_ONLY"
_ORDER_TOLERANCE = 1e-12
_FAIL_CLOSED_TOLERANCE = 1e-10
_MAX_GROSS = 1.0
_MAX_ABS_WEIGHT = 1.0
_DRAWDOWN_START = 0.10
_DRAWDOWN_STOP = 0.20


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite ordinary float")
    return value


def frozen_risk_scale(drawdown: float) -> float:
    drawdown = _finite_float(drawdown, "drawdown")
    if not 0.0 <= drawdown <= 1.0:
        raise ValueError("drawdown must be within [0, 1]")
    if drawdown <= _DRAWDOWN_START:
        return 1.0
    if drawdown >= _DRAWDOWN_STOP:
        return 0.0
    return 1.0 - (drawdown - _DRAWDOWN_START) / (_DRAWDOWN_STOP - _DRAWDOWN_START)


@dataclass(frozen=True, slots=True)
class ScalarRiskProjection:
    weights: float
    requested_turnover: float
    constrained_turnover: float
    was_constrained: bool
    reasons: tuple[str, ...]
    risk_scale: float
    projection_l1: float
    turnover_overridden: bool
    proposal_weights: float
    pretrade_weights: float
    max_gross: float
    drawdown_budget: float
    max_abs_weight: float
    fail_closed_tolerance: float

    @property
    def scope(self) -> str:
        return _SCOPE


def project_frozen_single_symbol(
    proposal_weight: float,
    *,
    current_weight: float,
    drawdown: float,
    emergency_flatten: bool = False,
    reduce_only: bool = False,
) -> ScalarRiskProjection:
    proposal = _finite_float(proposal_weight, "proposal_weight")
    current = _finite_float(current_weight, "current_weight")
    if type(emergency_flatten) is not bool:
        raise ValueError("emergency_flatten must be an exact bool")
    if type(reduce_only) is not bool:
        raise ValueError("reduce_only must be an exact bool")
    scale = frozen_risk_scale(drawdown)

    reasons: list[str] = []
    if reduce_only:
        reasons.append("reduce_only")
    requested_turnover = abs(proposal - current)

    controlled = proposal
    if emergency_flatten:
        controlled = 0.0
        reasons.append("emergency_flatten")

    reduce_only_satisfied = False
    if not emergency_flatten and reduce_only:
        if abs(current) <= _ORDER_TOLERANCE:
            controlled = 0.0
            reduce_only_satisfied = True
        elif controlled * current < -_ORDER_TOLERANCE:
            raise ValueError("reduce-only target cannot change sign")
        elif abs(controlled) >= abs(current) - _ORDER_TOLERANCE:
            controlled = current
            reduce_only_satisfied = True
    if reduce_only_satisfied:
        reasons.append("reduce_only_satisfied")

    weight = controlled
    if weight > _MAX_ABS_WEIGHT:
        weight = _MAX_ABS_WEIGHT
        reasons.append("max_abs_weight")
    elif weight < -_MAX_ABS_WEIGHT:
        weight = -_MAX_ABS_WEIGHT
        reasons.append("max_abs_weight")

    gross = abs(weight)
    if gross > _MAX_GROSS:
        weight *= _MAX_GROSS / gross
        reasons.append("max_gross")
    if scale < 1.0:
        weight *= scale
        reasons.append("drawdown_deleveraging")

    constrained_turnover = abs(weight - current)
    emergency_turnover = abs(weight - current) if emergency_flatten else 0.0
    turnover_overridden = emergency_turnover > _ORDER_TOLERANCE
    if turnover_overridden:
        reasons.append("emergency_turnover_override")

    if not math.isfinite(weight):
        raise RuntimeError("risk projection produced a non-finite weight")
    if abs(weight) > _MAX_ABS_WEIGHT * scale + _FAIL_CLOSED_TOLERANCE:
        raise RuntimeError("risk projection violated max_abs_weight")
    if abs(weight) > _MAX_GROSS * scale + _FAIL_CLOSED_TOLERANCE:
        raise RuntimeError("risk projection violated max_gross")
    if scale == 0.0 and abs(weight) > _ORDER_TOLERANCE:
        raise RuntimeError("emergency drawdown stop did not flatten target")

    return ScalarRiskProjection(
        weights=weight,
        requested_turnover=requested_turnover,
        constrained_turnover=constrained_turnover,
        was_constrained=bool(reasons),
        reasons=tuple(dict.fromkeys(reasons)),
        risk_scale=scale,
        projection_l1=abs(proposal - weight),
        turnover_overridden=turnover_overridden,
        proposal_weights=proposal,
        pretrade_weights=weight,
        max_gross=_MAX_GROSS,
        drawdown_budget=_DRAWDOWN_START,
        max_abs_weight=_MAX_ABS_WEIGHT,
        fail_closed_tolerance=_FAIL_CLOSED_TOLERANCE,
    )


@dataclass(frozen=True, slots=True)
class FrozenAllocationTarget:
    """One-symbol expected target stages, not an account or execution record."""

    role: str
    baseline_weight: float
    baseline_objective: float
    action_mode: str
    action_code: int
    action_label: str
    raw_action_target_weight: float
    action_target_weight: float
    is_hold: bool
    risk: ScalarRiskProjection

    @property
    def final_target_weight(self) -> float:
        return self.risk.weights

    @property
    def scope(self) -> str:
        return "FROZEN_SINGLE_SYMBOL_AFTER_COST_ACTION_RISK_TARGET_ONLY"


def compose_frozen_allocation_target(
    assumptions: AllocationAssumptions,
    *,
    current_weight: float,
    lower_weight: float,
    upper_weight: float,
    drawdown: float,
    role: str,
    emergency_flatten: bool = False,
    reduce_only: bool = False,
) -> FrozenAllocationTarget:
    """Compose supplied baseline arithmetic, fixed H role action, then H risk.

    Current weight, bounds, drawdown, and horizon assumptions are explicit
    caller inputs. This function does not authenticate them or read an account.
    The frozen NONRL role uses residual action code 2; CASH uses direct HOLD
    code 0. Both actions pass through the final single-symbol risk projection.
    """
    current = _finite_float(current_weight, "current_weight")
    if role == "NONRL":
        mode, code, label = "residual", 2, "baseline"
    elif role == "CASH":
        mode, code, label = "direct", 0, "quantity_hold"
    else:
        raise ValueError("role must be exactly NONRL or CASH")

    baseline = choose_after_cost_weight(
        assumptions,
        current_weight=current,
        lower_weight=lower_weight,
        upper_weight=upper_weight,
        risk_aversion=0.0,
        max_turnover=None,
    )
    if role == "NONRL":
        raw_action_target = baseline.target_weight
        action_target = raw_action_target
    else:
        # H direct action code 0 is an explicit hold and bypasses allocator
        # bounds. The hard risk projection still runs after the action.
        raw_action_target = current
        action_target = current

    risk = project_frozen_single_symbol(
        action_target,
        current_weight=current,
        drawdown=drawdown,
        emergency_flatten=emergency_flatten,
        reduce_only=reduce_only,
    )
    return FrozenAllocationTarget(
        role=role,
        baseline_weight=baseline.target_weight,
        baseline_objective=baseline.objective_value,
        action_mode=mode,
        action_code=code,
        action_label=label,
        raw_action_target_weight=raw_action_target,
        action_target_weight=action_target,
        is_hold=action_target == current,
        risk=risk,
    )


__all__ = [
    "FrozenAllocationTarget",
    "ScalarRiskProjection",
    "compose_frozen_allocation_target",
    "frozen_risk_scale",
    "project_frozen_single_symbol",
]
