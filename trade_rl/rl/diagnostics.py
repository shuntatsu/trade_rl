"""Auditable action and constraint diagnostics for policy evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ActionDiagnostics:
    n_steps: int
    n_values: int
    saturated_values: int
    constrained_steps: int
    turnover_override_steps: int
    mean_abs_action: float
    mean_action_delta_l1: float
    mean_projection_l1: float
    maximum_abs_action: float

    @property
    def saturation_rate(self) -> float:
        return 0.0 if self.n_values == 0 else self.saturated_values / self.n_values

    @property
    def constraint_activation_rate(self) -> float:
        return 0.0 if self.n_steps == 0 else self.constrained_steps / self.n_steps


class ActionDiagnosticsAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n_steps = 0
        self.n_values = 0
        self.saturated_values = 0
        self.constrained_steps = 0
        self.turnover_override_steps = 0
        self.absolute_action_sum = 0.0
        self.action_delta_sum = 0.0
        self.projection_sum = 0.0
        self.maximum_abs_action = 0.0

    def update(
        self,
        *,
        action: np.ndarray,
        saturated_count: int,
        action_delta_l1: float,
        projection_l1: float,
        constrained: bool,
        turnover_overridden: bool,
    ) -> None:
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("diagnostic action must be a finite non-empty vector")
        for field_name, value in (
            ("action_delta_l1", action_delta_l1),
            ("projection_l1", projection_l1),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if saturated_count < 0 or saturated_count > vector.size:
            raise ValueError("saturated_count is outside the action vector")
        self.n_steps += 1
        self.n_values += vector.size
        self.saturated_values += saturated_count
        self.constrained_steps += int(constrained)
        self.turnover_override_steps += int(turnover_overridden)
        self.absolute_action_sum += float(np.abs(vector).sum())
        self.action_delta_sum += action_delta_l1
        self.projection_sum += projection_l1
        self.maximum_abs_action = max(
            self.maximum_abs_action,
            float(np.max(np.abs(vector), initial=0.0)),
        )

    def snapshot(self) -> ActionDiagnostics:
        return ActionDiagnostics(
            n_steps=self.n_steps,
            n_values=self.n_values,
            saturated_values=self.saturated_values,
            constrained_steps=self.constrained_steps,
            turnover_override_steps=self.turnover_override_steps,
            mean_abs_action=(
                0.0 if self.n_values == 0 else self.absolute_action_sum / self.n_values
            ),
            mean_action_delta_l1=(
                0.0 if self.n_steps == 0 else self.action_delta_sum / self.n_steps
            ),
            mean_projection_l1=(
                0.0 if self.n_steps == 0 else self.projection_sum / self.n_steps
            ),
            maximum_abs_action=self.maximum_abs_action,
        )
