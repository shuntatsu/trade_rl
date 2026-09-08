"""Shared forecast-to-position intent controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True, slots=True)
class ForecastIntentConfig:
    """Symmetric hysteresis thresholds in forecast-return units."""

    entry_threshold: float
    exit_threshold: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.entry_threshold) or self.entry_threshold <= 0.0:
            raise ValueError("entry_threshold must be finite and positive")
        if not math.isfinite(self.exit_threshold) or self.exit_threshold < 0.0:
            raise ValueError("exit_threshold must be finite and non-negative")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be smaller than entry_threshold")


class ForecastIntentController:
    """Convert one signed return forecast into LONG/FLAT/SHORT intent."""

    def __init__(self, config: ForecastIntentConfig) -> None:
        self.config = config

    def decide(self, forecast: float, *, current: PositionIntent) -> PositionIntent:
        if not isinstance(current, PositionIntent):
            raise TypeError("current must be a PositionIntent")
        if not math.isfinite(forecast):
            return PositionIntent.FLAT

        entry = self.config.entry_threshold
        exit_threshold = self.config.exit_threshold
        if current is PositionIntent.FLAT:
            if forecast >= entry:
                return PositionIntent.LONG
            if forecast <= -entry:
                return PositionIntent.SHORT
            return PositionIntent.FLAT

        if current is PositionIntent.LONG:
            if forecast <= -entry:
                return PositionIntent.SHORT
            if forecast <= exit_threshold:
                return PositionIntent.FLAT
            return PositionIntent.LONG

        if forecast >= entry:
            return PositionIntent.LONG
        if forecast >= -exit_threshold:
            return PositionIntent.FLAT
        return PositionIntent.SHORT


__all__ = ["ForecastIntentConfig", "ForecastIntentController"]
