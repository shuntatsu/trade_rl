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
    it. An explicit short-term index instead combines the same completed-week
    band contact with opposite-side lower-timeframe Tenkan distance, without the
    weekly direction permission. These are separate opt-in packages, not an
    isolated indicator comparison. Both emit voluntary FLAT; quantity hold applies.
    """

    def __init__(
        self,
        strategy: SingleSymbolStrategy,
        feature_indices: tuple[int, ...],
        *,
        short_term_index: int | None = None,
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
        if short_term_index is not None and (
            isinstance(short_term_index, bool)
            or not isinstance(short_term_index, int)
            or short_term_index < 0
            or short_term_index in feature_indices
        ):
            raise ValueError("short-term index must be a distinct non-negative integer")
        self.short_term_index = short_term_index

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
        if self.short_term_index is not None:
            index = self.short_term_index
            if index >= observation.features.size:
                raise ValueError("short-term feature index out of range")
            if not observation.feature_available[index]:
                return PositionIntent.FLAT
            short_term = float(observation.features[index])
            if not math.isfinite(short_term):
                return PositionIntent.FLAT
            exhausted = (
                bb_high >= 1 and short_term < 0
                if decision is PositionIntent.LONG
                else bb_low <= -1 and short_term > 0
            )
            return PositionIntent.FLAT if exhausted else decision
        if decision is PositionIntent.LONG:
            permitted = upper > 0 and tenkan < kijun
            exhausted = bb_high >= 1 and tenkan < 0
        else:
            permitted = lower < 0 and tenkan > kijun
            exhausted = bb_low <= -1 and tenkan > 0
        return decision if permitted and not exhausted else PositionIntent.FLAT
