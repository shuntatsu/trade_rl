"""Frozen factories for the directional development study."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from trade_rl.data.features.price_channels import CHANNEL_NAMES
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments import ResolvedRunConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecasts.lightgbm import (
    LightGBMForecastStrategy,
    fit_lightgbm_forecast,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy
from trade_rl.strategies.rules.channel_breakout import ChannelBreakoutStrategy
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

ARMS = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "channel_breakout",
    "ppo0",
    "ppo1",
    "ppo2",
    "ppo3",
    "ppo4",
)
PPO_TIMESTEPS = 262_144


def validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown directional arm: {arm}")


def fit_directional_candidate(
    arm: str, dataset: MarketDataset, config: ResolvedRunConfig, output: Path
) -> Callable[[], SingleSymbolStrategy]:
    """Fit once on the fixed training side, then share the frozen model."""
    from dataclasses import replace

    validate_arm(arm)
    controls = {
        "cash": PositionIntent.FLAT,
        "constant_long": PositionIntent.LONG,
        "constant_short": PositionIntent.SHORT,
    }
    if arm in controls:
        return lambda: ConstantIntentStrategy(controls[arm])
    if arm == "channel_breakout":
        indices = tuple(dataset.feature_names.index(name) for name in CHANNEL_NAMES)
        return lambda: ChannelBreakoutStrategy(
            (indices[0], indices[1], indices[2], indices[3])
        )
    if arm == "trend":
        trend_config = TrendIntentConfig(
            config.signal_index, config.rule_entry_threshold, config.rule_exit_threshold
        )
        return lambda: TrendIntentStrategy(trend_config)
    if arm == "mean_reversion":
        reversal_config = MeanReversionIntentConfig(
            config.signal_index, config.rule_entry_threshold, config.rule_exit_threshold
        )
        return lambda: MeanReversionIntentStrategy(reversal_config)
    if arm == "ridge24":
        ridge = fit_ridge_forecast(
            dataset,
            feature_indices=config.feature_indices,
            fit_symbol_indices=config.fit_symbol_indices,
            fit_cutoff=np.datetime64(config.fit_cutoff),
            horizon_hours=24,
            alpha=1.0,
        )
        np.savez(
            output / "model.npz",
            mean=ridge.feature_mean,
            scale=ridge.feature_scale,
            coefficients=ridge.coefficients,
            intercept=np.array(ridge.intercept),
        )
        return lambda: RidgeForecastStrategy(
            ridge,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
        )
    if arm == "lightgbm24":
        gbm = fit_lightgbm_forecast(
            dataset,
            feature_indices=config.feature_indices,
            fit_symbol_indices=config.fit_symbol_indices,
            fit_cutoff=np.datetime64(config.fit_cutoff),
            horizon_hours=24,
            random_state=0,
        )
        getattr(gbm.predictor, "booster_").save_model(str(output / "model.txt"))
        return lambda: LightGBMForecastStrategy(
            gbm,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
        )
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    cutoff = (
        int(np.searchsorted(dataset.timestamps, np.datetime64(config.fit_cutoff))) - 1
    )
    ppo = fit_ppo_strategy(
        dataset,
        feature_indices=config.feature_indices,
        fit_symbol_indices=config.fit_symbol_indices,
        start_index=0,
        stop_index=cutoff,
        gross_budget=0.1,
        total_timesteps=PPO_TIMESTEPS,
        seed=int(arm[-1]),
        initial_capital=10_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(), processing_bar_volume_capacity=False
        ),
        training_layout="sequential",
    )
    getattr(ppo.policy, "save")(str(output / "model.zip"))
    return lambda: PPOIntentStrategy(ppo.policy, feature_indices=config.feature_indices)
