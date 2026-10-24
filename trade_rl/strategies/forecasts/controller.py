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
    """Gate log-return forecast intent changes with a simple-return cost proxy."""

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

        intent_delta = int(proposed) - int(current)
        if intent_delta > 0:
            return (
                proposed if forecast > math.log1p(self.one_way_switch_cost) else current
            )
        if self.one_way_switch_cost >= 1.0:
            return current
        return proposed if forecast < math.log1p(-self.one_way_switch_cost) else current

    def decide_for_position(
        self,
        forecast: float,
        *,
        current_weight: float,
        gross_budget: float,
        current_position_quantity: float | None = None,
    ) -> PositionIntent:
        """Choose among discrete targets using the actual filled position."""
        if not math.isfinite(current_weight):
            raise ValueError("current_weight must be finite")
        if current_position_quantity is not None and not math.isfinite(
            current_position_quantity
        ):
            raise ValueError("current_position_quantity must be finite when present")
        if not math.isfinite(gross_budget) or gross_budget <= 0.0 or gross_budget > 1.0:
            raise ValueError("gross_budget must be finite and within (0, 1]")
        if not math.isfinite(forecast):
            return PositionIntent.FLAT

        filled_exposure = (
            current_weight
            if current_position_quantity is None
            else current_position_quantity
        )
        if filled_exposure > 0.0:
            filled_side = PositionIntent.LONG
        elif filled_exposure < 0.0:
            filled_side = PositionIntent.SHORT
        else:
            filled_side = PositionIntent.FLAT

        proposed = self._baseline.decide(forecast, current=filled_side)
        candidates = tuple(dict.fromkeys((filled_side, proposed, PositionIntent.FLAT)))
        relative_weight = _bounded_relative_weight(
            current_weight,
            gross_budget=gross_budget,
        )
        selected = candidates[0]
        for candidate in candidates[1:]:
            if _has_higher_position_utility(
                forecast,
                candidate=candidate,
                current=selected,
                relative_weight=relative_weight,
                one_way_switch_cost=self.one_way_switch_cost,
            ):
                selected = candidate
        return selected


def _bounded_relative_weight(current_weight: float, *, gross_budget: float) -> float:
    if current_weight >= gross_budget:
        return 1.0
    if current_weight <= -gross_budget:
        return -1.0
    return current_weight / gross_budget


def _has_higher_position_utility(
    forecast: float,
    *,
    candidate: PositionIntent,
    current: PositionIntent,
    relative_weight: float,
    one_way_switch_cost: float,
) -> bool:
    """Compare two target utilities without forming potentially huge returns."""
    delta = int(candidate) - int(current)
    if delta == 0:
        return False

    relative_turnover_delta = (
        abs(int(candidate) - relative_weight) - abs(int(current) - relative_weight)
    ) / delta
    relative_turnover_delta = min(1.0, max(-1.0, relative_turnover_delta))
    simple_return_threshold = one_way_switch_cost * relative_turnover_delta
    if simple_return_threshold <= -1.0:
        return delta > 0

    forecast_threshold = math.log1p(simple_return_threshold)
    if delta > 0:
        return forecast > forecast_threshold
    return forecast < forecast_threshold


__all__ = [
    "CostAwareForecastIntentController",
    "ForecastIntentConfig",
    "ForecastIntentController",
]
