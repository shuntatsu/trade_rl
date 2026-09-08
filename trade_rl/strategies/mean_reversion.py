"""Lean single-signal mean-reversion intent strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True, slots=True)
class MeanReversionIntentConfig:
    """Symmetric hysteresis thresholds for one causal deviation feature."""

    signal_index: int
    entry_threshold: float
    exit_threshold: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.signal_index, bool)
            or not isinstance(self.signal_index, int)
            or self.signal_index < 0
        ):
            raise ValueError("signal_index must be a non-negative integer")
        if not math.isfinite(self.entry_threshold) or self.entry_threshold <= 0.0:
            raise ValueError("entry_threshold must be finite and positive")
        if not math.isfinite(self.exit_threshold) or self.exit_threshold < 0.0:
            raise ValueError("exit_threshold must be finite and non-negative")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be smaller than entry_threshold")


class MeanReversionIntentStrategy:
    """Fade large deviations and exit as the signal mean-reverts toward zero."""

    def __init__(self, config: MeanReversionIntentConfig) -> None:
        self.config = config

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        index = self.config.signal_index
        if index >= observation.features.size:
            raise ValueError("signal_index is outside observation features")
        if not bool(observation.feature_available[index]):
            return PositionIntent.FLAT

        signal = float(observation.features[index])
        if not math.isfinite(signal):
            return PositionIntent.FLAT

        entry = self.config.entry_threshold
        exit_threshold = self.config.exit_threshold
        current = observation.current_intent

        if current is PositionIntent.FLAT:
            if signal >= entry:
                return PositionIntent.SHORT
            if signal <= -entry:
                return PositionIntent.LONG
            return PositionIntent.FLAT

        if current is PositionIntent.LONG:
            if signal >= entry:
                return PositionIntent.SHORT
            if signal >= -exit_threshold:
                return PositionIntent.FLAT
            return PositionIntent.LONG

        if signal <= -entry:
            return PositionIntent.LONG
        if signal <= exit_threshold:
            return PositionIntent.FLAT
        return PositionIntent.SHORT


__all__ = ["MeanReversionIntentConfig", "MeanReversionIntentStrategy"]
