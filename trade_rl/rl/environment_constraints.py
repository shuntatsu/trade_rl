"""Causal action-path diagnostics and constraint-cost contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.domain.constraint_contracts import CONSTRAINT_COST_NAMES

_TOLERANCE = 1e-12
_FORCED_LIQUIDATION_REASONS = frozenset(
    {
        "minimum_equity",
        "execution_cost_exhaustion",
        "margin_call",
        "liquidation",
        "insolvency",
    }
)


def _readonly_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    vector.setflags(write=False)
    return vector


def _l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum())


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right), initial=0.0))


def _changed(left: np.ndarray, right: np.ndarray) -> bool:
    return not np.allclose(left, right, atol=_TOLERANCE, rtol=0.0)


def _termination_value(reason: object | None) -> str | None:
    if reason is None:
        return None
    raw = getattr(reason, "value", reason)
    return str(raw)


@dataclass(frozen=True, slots=True)
class ActionPathDiagnostics:
    """Immutable distances across the maintained target-to-fill pipeline."""

    policy_target: np.ndarray
    pretrade_target: np.ndarray
    feasible_target: np.ndarray
    submitted_order_target: np.ndarray
    filled_weight: np.ndarray
    execution_intent_target: np.ndarray | None = None
    policy_to_execution_intent_l1: float = field(init=False)
    execution_intent_to_pretrade_l1: float = field(init=False)
    policy_to_pretrade_l1: float = field(init=False)
    pretrade_to_feasible_l1: float = field(init=False)
    feasible_to_submitted_l1: float = field(init=False)
    submitted_to_filled_l1: float = field(init=False)
    execution_intent_to_filled_l1: float = field(init=False)
    policy_to_filled_l1: float = field(init=False)
    policy_to_execution_intent_max_abs: float = field(init=False)
    execution_intent_to_pretrade_max_abs: float = field(init=False)
    policy_to_pretrade_max_abs: float = field(init=False)
    pretrade_to_feasible_max_abs: float = field(init=False)
    feasible_to_submitted_max_abs: float = field(init=False)
    submitted_to_filled_max_abs: float = field(init=False)
    policy_changed_by_execution_delay: bool = field(init=False)
    execution_intent_changed_by_pretrade: bool = field(init=False)
    policy_changed_by_pretrade: bool = field(init=False)
    pretrade_changed_by_feasibility: bool = field(init=False)
    feasible_changed_before_submission: bool = field(init=False)
    submission_changed_by_fill: bool = field(init=False)

    def __post_init__(self) -> None:
        policy = _readonly_vector(self.policy_target, field_name="policy_target")
        execution_intent = _readonly_vector(
            policy
            if self.execution_intent_target is None
            else self.execution_intent_target,
            field_name="execution_intent_target",
        )
        names = (
            "pretrade_target",
            "feasible_target",
            "submitted_order_target",
            "filled_weight",
        )
        remaining = tuple(
            _readonly_vector(getattr(self, name), field_name=name) for name in names
        )
        vectors = (policy, execution_intent, *remaining)
        shape = policy.shape
        if any(vector.shape != shape for vector in vectors[1:]):
            raise ValueError("all action-path vectors must have the same shape")
        object.__setattr__(self, "policy_target", policy)
        object.__setattr__(self, "execution_intent_target", execution_intent)
        for name, vector in zip(names, remaining, strict=True):
            object.__setattr__(self, name, vector)

        pretrade, feasible, submitted, filled = remaining
        object.__setattr__(
            self,
            "policy_to_execution_intent_l1",
            _l1(policy, execution_intent),
        )
        object.__setattr__(
            self,
            "execution_intent_to_pretrade_l1",
            _l1(execution_intent, pretrade),
        )
        object.__setattr__(self, "policy_to_pretrade_l1", _l1(policy, pretrade))
        object.__setattr__(self, "pretrade_to_feasible_l1", _l1(pretrade, feasible))
        object.__setattr__(
            self,
            "feasible_to_submitted_l1",
            _l1(feasible, submitted),
        )
        object.__setattr__(
            self,
            "submitted_to_filled_l1",
            _l1(submitted, filled),
        )
        object.__setattr__(
            self,
            "execution_intent_to_filled_l1",
            _l1(execution_intent, filled),
        )
        object.__setattr__(self, "policy_to_filled_l1", _l1(policy, filled))
        object.__setattr__(
            self,
            "policy_to_execution_intent_max_abs",
            _max_abs(policy, execution_intent),
        )
        object.__setattr__(
            self,
            "execution_intent_to_pretrade_max_abs",
            _max_abs(execution_intent, pretrade),
        )
        object.__setattr__(
            self,
            "policy_to_pretrade_max_abs",
            _max_abs(policy, pretrade),
        )
        object.__setattr__(
            self,
            "pretrade_to_feasible_max_abs",
            _max_abs(pretrade, feasible),
        )
        object.__setattr__(
            self,
            "feasible_to_submitted_max_abs",
            _max_abs(feasible, submitted),
        )
        object.__setattr__(
            self,
            "submitted_to_filled_max_abs",
            _max_abs(submitted, filled),
        )
        object.__setattr__(
            self,
            "policy_changed_by_execution_delay",
            _changed(policy, execution_intent),
        )
        object.__setattr__(
            self,
            "execution_intent_changed_by_pretrade",
            _changed(execution_intent, pretrade),
        )
        object.__setattr__(
            self,
            "policy_changed_by_pretrade",
            _changed(policy, pretrade),
        )
        object.__setattr__(
            self,
            "pretrade_changed_by_feasibility",
            _changed(pretrade, feasible),
        )
        object.__setattr__(
            self,
            "feasible_changed_before_submission",
            _changed(feasible, submitted),
        )
        object.__setattr__(
            self,
            "submission_changed_by_fill",
            _changed(submitted, filled),
        )

    @classmethod
    def from_stages(
        cls,
        *,
        policy_target: np.ndarray,
        pretrade_target: np.ndarray,
        feasible_target: np.ndarray,
        submitted_order_target: np.ndarray,
        filled_weight: np.ndarray,
        execution_intent_target: np.ndarray | None = None,
    ) -> ActionPathDiagnostics:
        return cls(
            policy_target=policy_target,
            execution_intent_target=execution_intent_target,
            pretrade_target=pretrade_target,
            feasible_target=feasible_target,
            submitted_order_target=submitted_order_target,
            filled_weight=filled_weight,
        )


@dataclass(frozen=True, slots=True)
class ConstraintCostRequest:
    """Causal inputs required to calculate one transition's constraint costs."""

    policy_target: np.ndarray
    max_gross: float
    decision_hours: float
    drawdown: float
    drawdown_budget: float
    margin_deficit: float
    initial_capital: float
    previous_equity: float
    filled_turnover: float
    interval_cost: float
    interval_funding: float
    interval_borrow_cost: float
    termination_reason: object | None
    emergency_deleverage: bool
    liquidation_terminal: bool
    liquidation_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_target",
            _readonly_vector(self.policy_target, field_name="policy_target"),
        )
        for field_name in (
            "max_gross",
            "decision_hours",
            "drawdown",
            "drawdown_budget",
            "margin_deficit",
            "initial_capital",
            "previous_equity",
            "filled_turnover",
            "interval_cost",
            "interval_funding",
            "interval_borrow_cost",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.max_gross <= 0.0:
            raise ValueError("max_gross must be positive")
        if self.decision_hours <= 0.0:
            raise ValueError("decision_hours must be positive")
        if not 0.0 <= self.drawdown <= 1.0:
            raise ValueError("drawdown must be within [0, 1]")
        if not 0.0 <= self.drawdown_budget <= 1.0:
            raise ValueError("drawdown_budget must be within [0, 1]")
        if self.margin_deficit < 0.0:
            raise ValueError("margin_deficit must be non-negative")
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if self.previous_equity <= 0.0:
            raise ValueError("previous_equity must be positive")
        if self.filled_turnover < 0.0:
            raise ValueError("filled_turnover must be non-negative")
        if self.interval_borrow_cost < 0.0:
            raise ValueError("interval_borrow_cost must be non-negative")
        for field_name in (
            "emergency_deleverage",
            "liquidation_terminal",
            "liquidation_complete",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ConstraintCostVector:
    """Seven constraint costs plus non-constraint transition diagnostics."""

    drawdown_excess: float
    drawdown_stop_event: float
    margin_deficit_fraction: float
    forced_liquidation_event: float
    gross_exposure_request_excess: float
    daily_turnover: float
    execution_cost_fraction: float
    funding_credit_fraction: float
    transition_elapsed_hours: float | None = None

    def __post_init__(self) -> None:
        for field_name in (*CONSTRAINT_COST_NAMES, "funding_credit_fraction"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if self.transition_elapsed_hours is not None:
            elapsed = float(self.transition_elapsed_hours)
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError("transition_elapsed_hours must be finite and positive")
            object.__setattr__(self, "transition_elapsed_hours", elapsed)

    def constraint_dict(self) -> dict[str, float]:
        """Return only the seven values eligible for constraint optimization."""

        return {name: float(getattr(self, name)) for name in CONSTRAINT_COST_NAMES}

    def as_dict(self) -> dict[str, float]:
        """Return complete scalar telemetry that is present on this transition."""

        payload = {
            **self.constraint_dict(),
            "funding_credit_fraction": self.funding_credit_fraction,
        }
        if self.transition_elapsed_hours is not None:
            payload["transition_elapsed_hours"] = self.transition_elapsed_hours
        return payload


def calculate_constraint_costs(request: ConstraintCostRequest) -> ConstraintCostVector:
    """Calculate independent causal constraint costs without shaping reward."""

    margin_denominator = max(
        request.initial_capital,
        float(np.finfo(np.float64).eps),
    )
    equity_denominator = max(
        request.previous_equity,
        float(np.finfo(np.float64).eps),
    )
    reason = _termination_value(request.termination_reason)
    drawdown_stop_event = float(reason == "drawdown_stop")
    forced_liquidation = bool(
        not request.liquidation_terminal
        and (
            reason in _FORCED_LIQUIDATION_REASONS
            or (request.emergency_deleverage and not request.liquidation_complete)
        )
    )
    execution_cost = (
        max(0.0, request.interval_cost)
        + max(0.0, -request.interval_funding)
        + request.interval_borrow_cost
    )
    return ConstraintCostVector(
        drawdown_excess=max(0.0, request.drawdown - request.drawdown_budget),
        drawdown_stop_event=drawdown_stop_event,
        margin_deficit_fraction=request.margin_deficit / margin_denominator,
        forced_liquidation_event=float(forced_liquidation),
        gross_exposure_request_excess=max(
            0.0,
            float(np.abs(request.policy_target).sum()) - request.max_gross,
        ),
        daily_turnover=request.filled_turnover * 24.0 / request.decision_hours,
        execution_cost_fraction=execution_cost / equity_denominator,
        funding_credit_fraction=max(0.0, request.interval_funding) / equity_denominator,
        transition_elapsed_hours=request.decision_hours,
    )


__all__ = [
    "ActionPathDiagnostics",
    "CONSTRAINT_COST_NAMES",
    "ConstraintCostRequest",
    "ConstraintCostVector",
    "calculate_constraint_costs",
]
