"""Economic action gate around the existing mean-reversion proposal policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)

_POSITION_VALUE = {
    PositionIntent.LONG: 1,
    PositionIntent.FLAT: 0,
    PositionIntent.SHORT: -1,
}


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateConfig:
    """Frozen economic-gate parameters around one proposal configuration."""

    proposal: MeanReversionIntentConfig
    beta_gate: float
    one_way_explicit_cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, MeanReversionIntentConfig):
            raise TypeError("proposal must be a MeanReversionIntentConfig")
        if not math.isfinite(self.beta_gate) or self.beta_gate >= 0.0:
            raise ValueError("beta_gate must be finite and strictly negative")
        if (
            not math.isfinite(self.one_way_explicit_cost)
            or self.one_way_explicit_cost < 0.0
        ):
            raise ValueError("one_way_explicit_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EconomicTransitionDecision:
    """Inspectable arithmetic for one voluntary intent transition."""

    current: PositionIntent
    proposed: PositionIntent
    delta_position: int
    expected_incremental_edge: float
    transition_cost: float
    allowed: bool


def evaluate_economic_transition(
    *,
    current: PositionIntent,
    proposed: PositionIntent,
    signal: float,
    beta_gate: float,
    one_way_explicit_cost: float,
) -> EconomicTransitionDecision:
    """Evaluate the preregistered strict net-edge gate in return units."""

    if not isinstance(current, PositionIntent):
        raise TypeError("current must be a PositionIntent")
    if not isinstance(proposed, PositionIntent):
        raise TypeError("proposed must be a PositionIntent")
    if not math.isfinite(signal):
        raise ValueError("signal must be finite")
    if not math.isfinite(beta_gate) or beta_gate >= 0.0:
        raise ValueError("beta_gate must be finite and strictly negative")
    if not math.isfinite(one_way_explicit_cost) or one_way_explicit_cost < 0.0:
        raise ValueError("one_way_explicit_cost must be finite and non-negative")

    delta = _POSITION_VALUE[proposed] - _POSITION_VALUE[current]
    expected_edge = float(delta) * beta_gate * signal
    transition_cost = float(abs(delta)) * one_way_explicit_cost
    return EconomicTransitionDecision(
        current=current,
        proposed=proposed,
        delta_position=delta,
        expected_incremental_edge=expected_edge,
        transition_cost=transition_cost,
        allowed=expected_edge > transition_cost,
    )


class MeanReversionEconomicGateStrategy:
    """Gate voluntary mean-reversion transitions by sealed economic edge."""

    def __init__(self, config: MeanReversionEconomicGateConfig) -> None:
        if not isinstance(config, MeanReversionEconomicGateConfig):
            raise TypeError("config must be a MeanReversionEconomicGateConfig")
        self.config = config
        self._proposal = MeanReversionIntentStrategy(config.proposal)

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        proposed = self._proposal.decide(observation)
        signal_index = self.config.proposal.signal_index

        # Preserve the proposal policy's safety behavior. Missing/non-finite
        # signal must flatten even when the cost gate would otherwise prefer to
        # keep the current position.
        if not bool(observation.feature_available[signal_index]):
            return proposed
        signal = float(observation.features[signal_index])
        if not math.isfinite(signal):
            return proposed

        current = observation.current_intent
        if proposed is current:
            return current
        decision = evaluate_economic_transition(
            current=current,
            proposed=proposed,
            signal=signal,
            beta_gate=self.config.beta_gate,
            one_way_explicit_cost=self.config.one_way_explicit_cost,
        )
        return proposed if decision.allowed else current


__all__ = [
    "EconomicTransitionDecision",
    "MeanReversionEconomicGateConfig",
    "MeanReversionEconomicGateStrategy",
    "evaluate_economic_transition",
]
