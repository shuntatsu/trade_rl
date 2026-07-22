"""Immutable action and execution-delay planning for one environment decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    BaselineResidualComposer,
    ResidualAction,
    ResidualActionV2,
    ResidualComposition,
    TargetWeightAction,
)
from trade_rl.strategies.trend import TrendTargets


class DecisionCalendar(Protocol):
    @property
    def regular_cadence(self) -> bool: ...

    def bars_for_hours(self, hours: float) -> int: ...

    def bars_until(
        self,
        start: int,
        hours: float,
        *,
        maximum_index: int,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class EnvironmentDecisionRequest:
    action: np.ndarray
    trends: TrendTargets
    alpha: np.ndarray
    factor_basis: np.ndarray
    hybrid_weights: np.ndarray
    shadow_weights: np.ndarray
    pending_hybrid_target: np.ndarray | None
    pending_shadow_target: np.ndarray | None
    current_index: int
    end_index: int


@dataclass(frozen=True, slots=True)
class EnvironmentDecisionPlan:
    parsed_action: ResidualAction | ResidualActionV2 | TargetWeightAction
    maintained_action: np.ndarray
    saturated_count: int
    raw_max_abs: float
    composition: ResidualComposition
    submitted_hybrid_target: np.ndarray
    submitted_shadow_target: np.ndarray
    executed_hybrid_target: np.ndarray
    executed_shadow_target: np.ndarray
    next_pending_hybrid_target: np.ndarray | None
    next_pending_shadow_target: np.ndarray | None
    execution_delay_warmup: bool
    bars: int


class EnvironmentDecisionPlanner:
    """Parse actions and resolve causal targets without mutating environment state."""

    def __init__(
        self,
        dataset: DecisionCalendar,
        *,
        action_spec: ActionSpec,
        composer: BaselineResidualComposer,
        max_gross: float,
        alpha_enabled: bool,
        accept_legacy_actions: bool,
        signal_delay_decisions: int,
        decision_every: int | None,
        decision_hours: float,
    ) -> None:
        if not np.isfinite(max_gross) or max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")
        if signal_delay_decisions not in (0, 1):
            raise ValueError("signal_delay_decisions must be zero or one")
        if decision_every is not None and decision_every <= 0:
            raise ValueError("decision_every must be positive")
        if not np.isfinite(decision_hours) or decision_hours <= 0.0:
            raise ValueError("decision_hours must be finite and positive")
        self.dataset = dataset
        self.action_spec = action_spec
        self.composer = composer
        self.max_gross = float(max_gross)
        self.alpha_enabled = bool(alpha_enabled)
        self.accept_legacy_actions = bool(accept_legacy_actions)
        self.signal_delay_decisions = signal_delay_decisions
        self.decision_every = decision_every
        self.decision_hours = float(decision_hours)

    def parse_action(
        self,
        value: np.ndarray,
    ) -> tuple[
        ResidualAction | ResidualActionV2 | TargetWeightAction,
        np.ndarray,
        int,
        float,
    ]:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if (
            self.action_spec.mode is ActionMode.RESIDUAL
            and vector.shape == (2,)
            and self.accept_legacy_actions
        ):
            legacy = ResidualAction.from_array(vector)
            migrated = np.zeros(self.action_spec.size, dtype=np.float32)
            if legacy.trend_mix >= 0.0:
                migrated[0] = legacy.trend_mix
            else:
                migrated[1] = -legacy.trend_mix
            if self.alpha_enabled:
                migrated[self.action_spec.names.index("alpha_scale")] = (
                    legacy.alpha_budget
                )
            return (
                legacy,
                migrated,
                int(np.count_nonzero(np.abs(vector) > 1.0)),
                float(np.max(np.abs(vector), initial=0.0)),
            )
        parsed = self.action_spec.parse(value)
        if isinstance(parsed, TargetWeightAction):
            maintained = parsed.as_array()
        else:
            maintained = parsed.as_array(
                alpha_enabled=self.alpha_enabled,
                risk_tilt_enabled=self.action_spec.risk_tilt_enabled,
            )
        return (
            parsed,
            maintained,
            parsed.saturated_count,
            parsed.raw_max_abs,
        )

    def decision_bar_count(self, *, current_index: int, end_index: int) -> int:
        remaining = end_index - current_index
        if remaining <= 0:
            raise RuntimeError("step called after the episode ended")
        if self.decision_every is not None:
            return min(self.decision_every, remaining)
        if self.dataset.regular_cadence:
            return min(self.dataset.bars_for_hours(self.decision_hours), remaining)
        return self.dataset.bars_until(
            current_index,
            self.decision_hours,
            maximum_index=end_index,
        )

    def plan(self, request: EnvironmentDecisionRequest) -> EnvironmentDecisionPlan:
        parsed, maintained, saturated_count, raw_max_abs = self.parse_action(
            request.action
        )
        composition = self.composer.compose(
            parsed,
            request.trends,
            request.alpha,
            alpha_enabled=self.alpha_enabled,
            factor_basis=request.factor_basis,
            max_gross=self.max_gross,
        )
        submitted_hybrid = (
            np.asarray(composition.proposal, dtype=np.float64).reshape(-1).copy()
        )
        submitted_shadow = (
            np.asarray(request.trends.base, dtype=np.float64).reshape(-1).copy()
        )
        execution_delay_warmup = False
        next_pending_hybrid: np.ndarray | None = None
        next_pending_shadow: np.ndarray | None = None
        if self.signal_delay_decisions == 0:
            executed_hybrid = submitted_hybrid.copy()
            executed_shadow = submitted_shadow.copy()
        else:
            execution_delay_warmup = request.pending_hybrid_target is None
            executed_hybrid = (
                np.asarray(request.hybrid_weights, dtype=np.float64).reshape(-1).copy()
                if request.pending_hybrid_target is None
                else np.asarray(request.pending_hybrid_target, dtype=np.float64)
                .reshape(-1)
                .copy()
            )
            executed_shadow = (
                np.asarray(request.shadow_weights, dtype=np.float64).reshape(-1).copy()
                if request.pending_shadow_target is None
                else np.asarray(request.pending_shadow_target, dtype=np.float64)
                .reshape(-1)
                .copy()
            )
            next_pending_hybrid = submitted_hybrid.copy()
            next_pending_shadow = submitted_shadow.copy()
        return EnvironmentDecisionPlan(
            parsed_action=parsed,
            maintained_action=np.asarray(maintained, dtype=np.float32).copy(),
            saturated_count=saturated_count,
            raw_max_abs=raw_max_abs,
            composition=composition,
            submitted_hybrid_target=submitted_hybrid,
            submitted_shadow_target=submitted_shadow,
            executed_hybrid_target=executed_hybrid,
            executed_shadow_target=executed_shadow,
            next_pending_hybrid_target=next_pending_hybrid,
            next_pending_shadow_target=next_pending_shadow,
            execution_delay_warmup=execution_delay_warmup,
            bars=self.decision_bar_count(
                current_index=request.current_index,
                end_index=request.end_index,
            ),
        )


__all__ = [
    "EnvironmentDecisionPlan",
    "EnvironmentDecisionPlanner",
    "EnvironmentDecisionRequest",
]
