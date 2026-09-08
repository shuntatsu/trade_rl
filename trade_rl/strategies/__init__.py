"""Lean single-symbol strategy contracts."""

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecast import ForecastIntentConfig, ForecastIntentController
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent
from trade_rl.strategies.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.trend import TrendIntentConfig, TrendIntentStrategy

__all__ = [
    "ConstantIntentStrategy",
    "ForecastIntentConfig",
    "ForecastIntentController",
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "PositionIntent",
    "RidgeForecastModel",
    "RidgeForecastStrategy",
    "SingleSymbolStrategy",
    "StrategyObservation",
    "TrendIntentConfig",
    "TrendIntentStrategy",
    "fit_ridge_forecast",
    "target_weight_for_intent",
]
