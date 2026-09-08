"""Lean single-symbol strategy contracts."""

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecast import ForecastIntentConfig, ForecastIntentController
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.lightgbm import (
    LightGBMForecastModel,
    LightGBMForecastStrategy,
    fit_lightgbm_forecast,
)
from trade_rl.strategies.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent
from trade_rl.strategies.ppo import PPOIntentStrategy, PPOTradingEnv, fit_ppo_strategy
from trade_rl.strategies.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.supervised import (
    CausalForecastTrainingSet,
    build_causal_forecast_training_set,
)
from trade_rl.strategies.trend import TrendIntentConfig, TrendIntentStrategy

__all__ = [
    "CausalForecastTrainingSet",
    "ConstantIntentStrategy",
    "ForecastIntentConfig",
    "ForecastIntentController",
    "LightGBMForecastModel",
    "LightGBMForecastStrategy",
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "PPOIntentStrategy",
    "PPOTradingEnv",
    "PositionIntent",
    "RidgeForecastModel",
    "RidgeForecastStrategy",
    "SingleSymbolStrategy",
    "StrategyObservation",
    "TrendIntentConfig",
    "TrendIntentStrategy",
    "build_causal_forecast_training_set",
    "fit_lightgbm_forecast",
    "fit_ppo_strategy",
    "fit_ridge_forecast",
    "target_weight_for_intent",
]
