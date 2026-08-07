"""Framework-neutral target exposure to child-order state machine.

The policy chooses a target exposure. This controller deliberately does not create
venue orders and does not depend on NautilusTrader. It converts the target into the
next safe child-order instruction using only state observable at activation time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

_QUANTITY_TOLERANCE = 1e-12
_EXPOSURE_TOLERANCE = 1e-12


class ControllerPhase(str, Enum):
    """Observable phase of one target-exposure reconciliation generation."""

    IDLE = "idle"
    CANCELING_STALE = "canceling_stale"
    REDUCING = "reducing"
    OPENING = "opening"
    TRACKING = "tracking"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class TargetExposureInput:
    """State available when a queued target is activated."""

    target_exposure: float
    allocated_equity: float
    reference_price: float
    contract_multiplier: float
    realized_quantity: float
    working_remaining_quantities: tuple[float, ...]
    emergency_flatten: bool = False
    halted: bool = False


@dataclass(frozen=True, slots=True)
class TargetExposureChildOrder:
    """Framework-neutral instruction for exactly one next child order."""

    quantity: float
    reduce_only: bool


@dataclass(frozen=True, slots=True)
class TargetExposurePlan:
    """Deterministic next action for the target-exposure controller."""

    phase: ControllerPhase
    raw_target_exposure: float
    effective_target_exposure: float
    desired_quantity: float
    committed_quantity: float
    cancel_working_orders: bool
    child_order: TargetExposureChildOrder | None
    deferred_target_quantity: float | None = None


class TargetExposureController:
    """Reconcile one signed exposure target without crossing through flat in one order."""

    def __init__(self, *, no_trade_band: float = 0.05) -> None:
        if not math.isfinite(no_trade_band) or not 0.0 <= no_trade_band < 1.0:
            raise ValueError("no_trade_band must be finite and within [0, 1)")
        self._no_trade_band = float(no_trade_band)

    @property
    def no_trade_band(self) -> float:
        return self._no_trade_band

    def plan(self, state: TargetExposureInput) -> TargetExposurePlan:
        """Return only the next safe controller action for ``state``.

        Working residual quantity is part of committed exposure. If a new target no
        longer matches that commitment, cancellation is always a separate phase. A
        sign reversal is also split: first reduce the realized position to flat,
        then a later invocation may open the opposite side after terminal evidence.
        """

        self._validate_input(state)

        raw_target = float(state.target_exposure)
        committed = float(state.realized_quantity) + math.fsum(
            state.working_remaining_quantities
        )
        quantity_scale = (
            float(state.reference_price) * float(state.contract_multiplier)
        )
        committed_exposure = committed * quantity_scale / float(state.allocated_equity)

        if state.halted:
            return TargetExposurePlan(
                phase=ControllerPhase.HALTED,
                raw_target_exposure=raw_target,
                effective_target_exposure=committed_exposure,
                desired_quantity=committed,
                committed_quantity=committed,
                cancel_working_orders=False,
                child_order=None,
            )

        effective_target = 0.0 if state.emergency_flatten else raw_target
        if (
            not state.emergency_flatten
            and abs(effective_target - committed_exposure)
            <= self._no_trade_band + _EXPOSURE_TOLERANCE
        ):
            effective_target = committed_exposure

        desired = (
            effective_target
            * float(state.allocated_equity)
            / quantity_scale
        )

        if state.working_remaining_quantities:
            if math.isclose(
                desired,
                committed,
                rel_tol=0.0,
                abs_tol=_QUANTITY_TOLERANCE,
            ):
                return TargetExposurePlan(
                    phase=ControllerPhase.TRACKING,
                    raw_target_exposure=raw_target,
                    effective_target_exposure=effective_target,
                    desired_quantity=desired,
                    committed_quantity=committed,
                    cancel_working_orders=False,
                    child_order=None,
                )
            return TargetExposurePlan(
                phase=ControllerPhase.CANCELING_STALE,
                raw_target_exposure=raw_target,
                effective_target_exposure=effective_target,
                desired_quantity=desired,
                committed_quantity=committed,
                cancel_working_orders=True,
                child_order=None,
            )

        realized = float(state.realized_quantity)
        if math.isclose(
            desired,
            realized,
            rel_tol=0.0,
            abs_tol=_QUANTITY_TOLERANCE,
        ):
            return TargetExposurePlan(
                phase=ControllerPhase.IDLE,
                raw_target_exposure=raw_target,
                effective_target_exposure=effective_target,
                desired_quantity=desired,
                committed_quantity=committed,
                cancel_working_orders=False,
                child_order=None,
            )

        if self._opposite_sign(realized, desired):
            return TargetExposurePlan(
                phase=ControllerPhase.REDUCING,
                raw_target_exposure=raw_target,
                effective_target_exposure=effective_target,
                desired_quantity=desired,
                committed_quantity=committed,
                cancel_working_orders=False,
                child_order=TargetExposureChildOrder(
                    quantity=-realized,
                    reduce_only=True,
                ),
                deferred_target_quantity=desired,
            )

        delta = desired - realized
        reducing = (
            not math.isclose(realized, 0.0, abs_tol=_QUANTITY_TOLERANCE)
            and abs(desired) < abs(realized) - _QUANTITY_TOLERANCE
        )
        return TargetExposurePlan(
            phase=ControllerPhase.REDUCING if reducing else ControllerPhase.OPENING,
            raw_target_exposure=raw_target,
            effective_target_exposure=effective_target,
            desired_quantity=desired,
            committed_quantity=committed,
            cancel_working_orders=False,
            child_order=TargetExposureChildOrder(
                quantity=delta,
                reduce_only=reducing,
            ),
        )

    @staticmethod
    def _opposite_sign(left: float, right: float) -> bool:
        if abs(left) <= _QUANTITY_TOLERANCE or abs(right) <= _QUANTITY_TOLERANCE:
            return False
        return math.copysign(1.0, left) != math.copysign(1.0, right)

    @staticmethod
    def _validate_input(state: TargetExposureInput) -> None:
        if (
            not math.isfinite(state.target_exposure)
            or not -1.0 <= state.target_exposure <= 1.0
        ):
            raise ValueError("target_exposure must be finite and within [-1, 1]")
        if not math.isfinite(state.allocated_equity) or state.allocated_equity <= 0.0:
            raise ValueError("allocated_equity must be finite and positive")
        if not math.isfinite(state.reference_price) or state.reference_price <= 0.0:
            raise ValueError("reference_price must be finite and positive")
        if (
            not math.isfinite(state.contract_multiplier)
            or state.contract_multiplier <= 0.0
        ):
            raise ValueError("contract_multiplier must be finite and positive")
        if not math.isfinite(state.realized_quantity):
            raise ValueError("realized_quantity must be finite")
        if any(not math.isfinite(value) for value in state.working_remaining_quantities):
            raise ValueError("working_remaining_quantities must be finite")


__all__ = [
    "ControllerPhase",
    "TargetExposureChildOrder",
    "TargetExposureController",
    "TargetExposureInput",
    "TargetExposurePlan",
]
