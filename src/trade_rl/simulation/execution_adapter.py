"""Compatibility import for the canonical maintained market executor."""

from trade_rl.simulation.execution import MarketExecutor

StatefulCompatibilityMarketExecutor = MarketExecutor

__all__ = ["StatefulCompatibilityMarketExecutor"]
