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


class CostAwareForecastIntentController:
    """Veto forecast-intent changes whose expected improvement does not pay cost."""

    def __init__(
        self,
        config: ForecastIntentConfig,
        *,
        one_way_switch_cost: float,
    ) -> None:
        if (
            isinstance(one_way_switch_cost, bool)
            or not math.isfinite(one_way_switch_cost)
            or one_way_switch_cost < 0.0
        ):
            raise ValueError("one-way switch cost must be finite and non-negative")
        self.config = config
        self.one_way_switch_cost = float(one_way_switch_cost)
        self._baseline = ForecastIntentController(config)

    def decide(self, forecast: float, *, current: PositionIntent) -> PositionIntent:
        proposed = self._baseline.decide(forecast, current=current)
        if not math.isfinite(forecast):
            return proposed
        if proposed is current:
            return current

        intent_distance = abs(int(proposed) - int(current))
        expected_improvement = float(int(proposed) - int(current)) * forecast
        switching_cost = intent_distance * self.one_way_switch_cost
        return proposed if expected_improvement > switching_cost else current


__all__ = [
    "CostAwareForecastIntentController",
    "ForecastIntentConfig",
    "ForecastIntentController",
]
