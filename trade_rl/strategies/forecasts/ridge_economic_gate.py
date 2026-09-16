"""Economic transition gate around the existing Ridge forecast proposal policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.strategies.forecasts.controller import ForecastIntentConfig
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

_POSITION_VALUE = {
    PositionIntent.LONG: 1,
    PositionIntent.FLAT: 0,
    PositionIntent.SHORT: -1,
}


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateConfig:
    """Frozen wrapper inputs around one existing Ridge proposal policy."""

    model: RidgeForecastModel
    entry_threshold: float
    exit_threshold: float
    one_way_explicit_cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, RidgeForecastModel):
            raise TypeError("model must be a RidgeForecastModel")
        ForecastIntentConfig(
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
        )
        if (
            not math.isfinite(self.one_way_explicit_cost)
            or self.one_way_explicit_cost < 0.0
        ):
            raise ValueError("one_way_explicit_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RidgeEconomicTransitionDecision:
    """Inspectable arithmetic for one voluntary Ridge intent transition."""

    current: PositionIntent
    proposed: PositionIntent
    delta_position: int
    expected_incremental_edge: float
    transition_cost: float
    allowed: bool


def evaluate_ridge_economic_transition(
    *,
    current: PositionIntent,
    proposed: PositionIntent,
    forecast: float,
    one_way_explicit_cost: float,
) -> RidgeEconomicTransitionDecision:
    """Evaluate the preregistered strict net-edge gate in forecast-return units."""

    if not isinstance(current, PositionIntent):
        raise TypeError("current must be a PositionIntent")
    if not isinstance(proposed, PositionIntent):
        raise TypeError("proposed must be a PositionIntent")
    if not math.isfinite(forecast):
        raise ValueError("forecast must be finite")
    if not math.isfinite(one_way_explicit_cost) or one_way_explicit_cost < 0.0:
        raise ValueError("one_way_explicit_cost must be finite and non-negative")

    delta = _POSITION_VALUE[proposed] - _POSITION_VALUE[current]
    expected_edge = float(delta) * forecast
    transition_cost = float(abs(delta)) * one_way_explicit_cost
    return RidgeEconomicTransitionDecision(
        current=current,
        proposed=proposed,
        delta_position=delta,
        expected_incremental_edge=expected_edge,
        transition_cost=transition_cost,
        allowed=expected_edge > transition_cost,
    )


class RidgeEconomicGateStrategy:
    """Gate only voluntary transitions from the existing Ridge proposal policy."""

    def __init__(self, config: RidgeEconomicGateConfig) -> None:
        if not isinstance(config, RidgeEconomicGateConfig):
            raise TypeError("config must be a RidgeEconomicGateConfig")
        self.config = config
        self.model = config.model
        self.proposal = RidgeForecastStrategy(
            config.model,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
        )

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        proposed = self.proposal.decide(observation)
        indices = self.model.feature_indices

        # Preserve the existing proposal policy's fail-safe FLAT behavior.
        # The cost gate must never retain a stale position when required Ridge
        # features are unavailable or non-finite.
        if not bool(np.all(observation.feature_available[list(indices)])):
            return proposed
        selected = np.asarray(observation.features[list(indices)], dtype=np.float64)
        if not np.isfinite(selected).all():
            return proposed

        current = observation.current_intent
        if proposed is current:
            return current
        forecast = self.model.predict(observation.features)
        decision = evaluate_ridge_economic_transition(
            current=current,
            proposed=proposed,
            forecast=forecast,
            one_way_explicit_cost=self.config.one_way_explicit_cost,
        )
        return proposed if decision.allowed else current


__all__ = [
    "RidgeEconomicGateConfig",
    "RidgeEconomicGateStrategy",
    "RidgeEconomicTransitionDecision",
    "evaluate_ridge_economic_transition",
]
