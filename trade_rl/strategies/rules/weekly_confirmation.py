"""Voluntary directional confirmation from completed weekly BB and Ichimoku."""

from __future__ import annotations

import math

from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class WeeklyConfirmationStrategy:
    """Filter an existing strategy without changing its filled-position state.

    A band touch alone never exits or reverses. Weekly trend permission requires
    price beyond the plotted cloud and Tenkan beyond Kijun in the same direction.
    A band-reaching wick plus a close beyond Tenkan against that direction blocks
    it. These are voluntary FLAT intents; ordinary quantity hold still applies.
    """

    def __init__(
        self, strategy: SingleSymbolStrategy, feature_indices: tuple[int, ...]
    ) -> None:
        if (
            len(feature_indices) != 7
            or len(set(feature_indices)) != 7
            or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in feature_indices
            )
        ):
            raise ValueError(
                "weekly context requires seven distinct non-negative feature indices"
            )
        self.strategy = strategy
        self.feature_indices = feature_indices

    @property
    def protective_exit_pending(self) -> bool:
        return bool(getattr(self.strategy, "protective_exit_pending", False))

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        decision = self.strategy.decide(observation)
        if decision is PositionIntent.FLAT:
            return decision
        if any(i >= observation.features.size for i in self.feature_indices):
            raise ValueError("weekly context feature index out of range")
        if not observation.feature_available[list(self.feature_indices)].all():
            return PositionIntent.FLAT
        values = tuple(float(observation.features[i]) for i in self.feature_indices)
        if not all(math.isfinite(value) for value in values):
            return PositionIntent.FLAT
        _, bb_high, bb_low, tenkan, kijun, upper, lower = values
        if decision is PositionIntent.LONG:
            permitted = upper > 0 and tenkan < kijun
            exhausted = bb_high >= 1 and tenkan < 0
        else:
            permitted = lower < 0 and tenkan > kijun
            exhausted = bb_low <= -1 and tenkan > 0
        return decision if permitted and not exhausted else PositionIntent.FLAT
