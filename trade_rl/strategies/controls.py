"""Fixed-intent benchmark strategies."""

from __future__ import annotations

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class ConstantIntentStrategy:
    """Return the same position intent for every observation."""

    def __init__(self, intent: PositionIntent) -> None:
        if not isinstance(intent, PositionIntent):
            raise TypeError("intent must be a PositionIntent")
        self.intent = intent

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        del observation
        return self.intent


__all__ = ["ConstantIntentStrategy"]
