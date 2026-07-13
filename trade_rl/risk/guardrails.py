"""Operational fail-closed guardrails shared by research and serving."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperationalGuardrailConfig:
    max_data_age_hours: float = 2.0
    max_daily_loss: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_data_age_hours) or self.max_data_age_hours < 0.0:
            raise ValueError("max_data_age_hours must be finite and non-negative")
        if (
            not math.isfinite(self.max_daily_loss)
            or not 0.0 <= self.max_daily_loss < 1.0
        ):
            raise ValueError("max_daily_loss must be within [0, 1)")


@dataclass(frozen=True, slots=True)
class GuardrailTarget:
    weights: np.ndarray
    triggered: tuple[str, ...]


class OperationalGuardrails:
    """Flatten proposals when market state is stale, absent, or loss-limited."""

    def __init__(self, config: OperationalGuardrailConfig | None = None) -> None:
        self.config = config or OperationalGuardrailConfig()

    def apply(
        self,
        target: np.ndarray,
        *,
        portfolio_value: float,
        day_start_value: float,
        data_age_hours: float,
        features_available: bool,
    ) -> GuardrailTarget:
        weights = np.asarray(target, dtype=np.float64).reshape(-1).copy()
        if weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError("guardrail target must be a non-empty finite vector")
        for field_name, value in (
            ("portfolio_value", portfolio_value),
            ("day_start_value", day_start_value),
            ("data_age_hours", data_age_hours),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if portfolio_value <= 0.0 or day_start_value <= 0.0:
            raise ValueError("portfolio and day-start values must be positive")
        if data_age_hours < 0.0:
            raise ValueError("data_age_hours must be non-negative")

        triggered: list[str] = []
        if data_age_hours > self.config.max_data_age_hours:
            triggered.append("stale_data")
        if not features_available:
            triggered.append("features_unavailable")
        daily_loss = 1.0 - portfolio_value / day_start_value
        if daily_loss >= self.config.max_daily_loss:
            triggered.append("daily_loss")
        if triggered:
            weights.fill(0.0)
        return GuardrailTarget(weights=weights, triggered=tuple(triggered))
