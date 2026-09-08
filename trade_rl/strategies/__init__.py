"""Lean single-symbol strategy contracts."""

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent
from trade_rl.strategies.trend import TrendIntentConfig, TrendIntentStrategy

__all__ = [
    "ConstantIntentStrategy",
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "PositionIntent",
    "SingleSymbolStrategy",
    "StrategyObservation",
    "TrendIntentConfig",
    "TrendIntentStrategy",
    "target_weight_for_intent",
]
