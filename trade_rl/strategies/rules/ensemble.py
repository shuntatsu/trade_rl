"""Multi-rule ensemble intent strategy designed for risk-adjusted profit maximization."""

from __future__ import annotations

from collections.abc import Sequence

from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class EnsembleIntentStrategy:
    """Combines multiple strategies by voting/agreement to filter false breakouts.

    Maximizes net profit by requiring agreement between sub-strategies (e.g. trend
    momentum and mean-reversion filters), drastically reducing turnover and trading costs.
    """

    def __init__(
        self,
        strategies: Sequence[SingleSymbolStrategy],
        *,
        min_agreement: int = 1,
    ) -> None:
        if not strategies:
            raise ValueError("strategies sequence must not be empty")
        if type(min_agreement) is not int:
            raise ValueError("min_agreement must be an integer")
        if min_agreement < 1 or min_agreement > len(strategies):
            raise ValueError(
                f"min_agreement must be between 1 and {len(strategies)}, got {min_agreement}"
            )
        self.strategies = tuple(strategies)
        self.min_agreement = min_agreement

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        long_votes = 0
        short_votes = 0

        for strat in self.strategies:
            decision = strat.decide(observation)
            if decision is PositionIntent.LONG:
                long_votes += 1
            elif decision is PositionIntent.SHORT:
                short_votes += 1

        if long_votes >= self.min_agreement and long_votes > short_votes:
            return PositionIntent.LONG
        if short_votes >= self.min_agreement and short_votes > long_votes:
            return PositionIntent.SHORT

        return PositionIntent.FLAT


__all__ = ["EnsembleIntentStrategy"]
