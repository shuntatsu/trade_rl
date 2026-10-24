"""Shared forecast-to-position intent controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

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
        current_intent: PositionIntent | None = None,
        same_intent_target_weight: float | None = None,
    ) -> PositionIntent:
        """Choose among discrete intents using filled and cached target weights."""
        if not math.isfinite(current_weight):
            raise ValueError("current_weight must be finite")
        if current_position_quantity is not None and not math.isfinite(
            current_position_quantity
        ):
            raise ValueError("current_position_quantity must be finite when present")
        if not math.isfinite(gross_budget) or gross_budget <= 0.0 or gross_budget > 1.0:
            raise ValueError("gross_budget must be finite and within (0, 1]")
        if current_intent is not None and not isinstance(
            current_intent, PositionIntent
        ):
            raise TypeError("current_intent must be a PositionIntent when provided")
        if same_intent_target_weight is not None:
            if current_intent is None:
                raise ValueError(
                    "current_intent is required with same_intent_target_weight"
                )
            if not math.isfinite(same_intent_target_weight):
                raise ValueError("same_intent_target_weight must be finite")
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
        selected = candidates[0]
        selected_target = _position_target_weight(
            selected,
            gross_budget=gross_budget,
            current_intent=current_intent,
            same_intent_target_weight=same_intent_target_weight,
        )
        for candidate in candidates[1:]:
            candidate_target = _position_target_weight(
                candidate,
                gross_budget=gross_budget,
                current_intent=current_intent,
                same_intent_target_weight=same_intent_target_weight,
            )
            if _has_higher_position_utility(
                forecast,
                candidate_target=candidate_target,
                current_target=selected_target,
                current_weight=current_weight,
                one_way_switch_cost=self.one_way_switch_cost,
            ):
                selected = candidate
                selected_target = candidate_target
        return selected


def _position_target_weight(
    intent: PositionIntent,
    *,
    gross_budget: float,
    current_intent: PositionIntent | None,
    same_intent_target_weight: float | None,
) -> float:
    if same_intent_target_weight is not None and intent is current_intent:
        return same_intent_target_weight
    return int(intent) * gross_budget


def _has_higher_position_utility(
    forecast: float,
    *,
    candidate_target: float,
    current_target: float,
    current_weight: float,
    one_way_switch_cost: float,
) -> bool:
    """Compare target utilities without forming returns or large turnover terms."""
    if candidate_target == current_target:
        return False

    candidate_is_higher = candidate_target > current_target
    relative_turnover_delta = _relative_turnover_slope(
        candidate_target,
        current_target,
        current_weight,
    )
    simple_return_threshold = one_way_switch_cost * relative_turnover_delta
    threshold_gap = 1.0 + simple_return_threshold
    if abs(threshold_gap) <= 1e-8:
        exact_gap = _exact_simple_return_threshold_gap(
            candidate_target=candidate_target,
            current_target=current_target,
            current_weight=current_weight,
            one_way_switch_cost=one_way_switch_cost,
        )
        if exact_gap <= 0:
            return candidate_is_higher
        forecast_threshold = _log_fraction(exact_gap)
    elif threshold_gap <= 0.0:
        return candidate_is_higher
    else:
        forecast_threshold = math.log1p(simple_return_threshold)
    if candidate_is_higher:
        return forecast > forecast_threshold
    return forecast < forecast_threshold


def _relative_turnover_slope(
    candidate_target: float,
    current_target: float,
    current_weight: float,
) -> float:
    """Return the bounded turnover slope for two arbitrary target weights."""
    low = min(candidate_target, current_target)
    high = max(candidate_target, current_target)
    if current_weight <= low:
        return 1.0
    if current_weight >= high:
        return -1.0

    scale = max(abs(low), abs(high), abs(current_weight))
    low_scaled = low / scale
    high_scaled = high / scale
    weight_scaled = current_weight / scale
    width = high_scaled - low_scaled
    slope = (high_scaled - weight_scaled) / width - (weight_scaled - low_scaled) / width
    return min(1.0, max(-1.0, slope))


def _exact_simple_return_threshold_gap(
    *,
    candidate_target: float,
    current_target: float,
    current_weight: float,
    one_way_switch_cost: float,
) -> Fraction:
    """Recover a tiny ``1 + cost * slope`` gap lost to float cancellation."""
    candidate = Fraction.from_float(float(candidate_target))
    current = Fraction.from_float(float(current_target))
    weight = Fraction.from_float(float(current_weight))
    cost = Fraction.from_float(float(one_way_switch_cost))
    slope = (abs(candidate - weight) - abs(current - weight)) / (candidate - current)
    return Fraction(1) + cost * slope


def _log_fraction(value: Fraction) -> float:
    """Take the logarithm of a positive rational without converting it first."""
    return math.log(value.numerator) - math.log(value.denominator)


__all__ = [
    "CostAwareForecastIntentController",
    "ForecastIntentConfig",
    "ForecastIntentController",
]
