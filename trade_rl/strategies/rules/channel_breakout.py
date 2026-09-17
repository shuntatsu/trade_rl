"""Sparse directional breakout intent with separate entry and exit channels."""

from __future__ import annotations

import numpy as np

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class ChannelBreakoutStrategy:
    """Enter on the long channel and exit on the shorter channel."""

    def __init__(self, feature_indices: tuple[int, int, int, int]) -> None:
        if (
            len(feature_indices) != 4
            or len(set(feature_indices)) != 4
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in feature_indices
            )
        ):
            raise ValueError(
                "channel feature indices must be four distinct nonnegative integers"
            )
        self.feature_indices = feature_indices

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        if max(self.feature_indices) >= observation.features.size:
            raise ValueError("channel feature index outside observation")
        indices = list(self.feature_indices)
        channels = observation.features[indices]
        if not np.all(observation.feature_available[indices]) or not np.all(
            np.isfinite(channels)
        ):
            return PositionIntent.FLAT
        entry_upper, entry_lower, exit_upper, exit_lower = channels
        if entry_upper > 0:
            return PositionIntent.LONG
        if entry_lower < 0:
            return PositionIntent.SHORT
        if observation.current_intent is PositionIntent.LONG and exit_lower >= 0:
            return PositionIntent.LONG
        if observation.current_intent is PositionIntent.SHORT and exit_upper <= 0:
            return PositionIntent.SHORT
        return PositionIntent.FLAT
