"""Fixed M2 candidate suite on the shared lean replay contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.strategy_comparison import (
    StrategyComparison,
    compare_strategies,
)
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.lightgbm import (
    LightGBMForecastStrategy,
    fit_lightgbm_forecast,
)
from trade_rl.strategies.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.ppo import fit_ppo_strategy
from trade_rl.strategies.ridge import RidgeForecastStrategy, fit_ridge_forecast
from trade_rl.strategies.trend import TrendIntentConfig, TrendIntentStrategy


@dataclass(frozen=True, slots=True)
class LeanCandidateConfig:
    """Only the pre-registered degrees of freedom for the initial M2 suite."""

    signal_index: int
    feature_indices: tuple[int, ...]
    fit_cutoff: np.datetime64
    rule_entry_threshold: float
    rule_exit_threshold: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    ppo_total_timesteps: int
    ppo_seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.signal_index, bool)
            or not isinstance(self.signal_index, int)
            or self.signal_index < 0
        ):
            raise ValueError("signal_index must be a non-negative integer")
        indices = tuple(self.feature_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be non-empty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("feature_indices must contain non-negative integers")
        for field_name, value in (
            ("rule_entry_threshold", self.rule_entry_threshold),
            ("forecast_entry_threshold", self.forecast_entry_threshold),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name, value in (
            ("rule_exit_threshold", self.rule_exit_threshold),
            ("forecast_exit_threshold", self.forecast_exit_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.rule_exit_threshold >= self.rule_entry_threshold:
            raise ValueError("rule exit threshold must be below entry threshold")
        if self.forecast_exit_threshold >= self.forecast_entry_threshold:
            raise ValueError("forecast exit threshold must be below entry threshold")
        if (
            isinstance(self.ppo_total_timesteps, bool)
            or not isinstance(self.ppo_total_timesteps, int)
            or self.ppo_total_timesteps <= 0
        ):
            raise ValueError("ppo_total_timesteps must be a positive integer")
        if isinstance(self.ppo_seed, bool) or not isinstance(self.ppo_seed, int):
            raise ValueError("ppo_seed must be a non-negative integer")
        if self.ppo_seed < 0:
            raise ValueError("ppo_seed must be a non-negative integer")
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "fit_cutoff", np.datetime64(self.fit_cutoff, "ns"))


def _ppo_training_stop_index(
    dataset: MarketDataset,
    fit_cutoff: np.datetime64,
) -> int:
    eligible = np.flatnonzero(dataset.timestamps < fit_cutoff)
    if eligible.size < 2:
        raise ValueError("PPO fitting requires at least two bars before fit_cutoff")
    return int(eligible[-1])


def run_lean_candidate_suite(
    dataset: MarketDataset,
    config: LeanCandidateConfig,
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
) -> StrategyComparison:
    """Fit and compare the five initial candidates plus three controls."""

    if dataset.n_symbols != 1:
        raise ValueError("lean candidate suite requires exactly one symbol")
    if config.signal_index >= dataset.n_features:
        raise ValueError("signal_index is outside dataset features")
    if max(config.feature_indices) >= dataset.n_features:
        raise ValueError("feature index is outside dataset features")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("evaluation range must satisfy 0 <= start < stop < n_bars")
    if dataset.timestamps[start_index] < config.fit_cutoff:
        raise ValueError("evaluation must not start before fit_cutoff")

    ridge_model = fit_ridge_forecast(
        dataset,
        feature_indices=config.feature_indices,
        fit_cutoff=config.fit_cutoff,
        horizon_hours=24,
        alpha=1.0,
    )
    lightgbm_model = fit_lightgbm_forecast(
        dataset,
        feature_indices=config.feature_indices,
        fit_cutoff=config.fit_cutoff,
        horizon_hours=24,
        random_state=0,
    )
    ppo_strategy = fit_ppo_strategy(
        dataset,
        feature_indices=config.feature_indices,
        start_index=0,
        stop_index=_ppo_training_stop_index(dataset, config.fit_cutoff),
        gross_budget=gross_budget,
        total_timesteps=config.ppo_total_timesteps,
        seed=config.ppo_seed,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
    )

    strategies: dict[str, SingleSymbolStrategy] = {
        "cash": ConstantIntentStrategy(PositionIntent.FLAT),
        "constant_long": ConstantIntentStrategy(PositionIntent.LONG),
        "constant_short": ConstantIntentStrategy(PositionIntent.SHORT),
        "trend": TrendIntentStrategy(
            TrendIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.rule_entry_threshold,
                exit_threshold=config.rule_exit_threshold,
            )
        ),
        "mean_reversion": MeanReversionIntentStrategy(
            MeanReversionIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.rule_entry_threshold,
                exit_threshold=config.rule_exit_threshold,
            )
        ),
        "ridge24": RidgeForecastStrategy(
            ridge_model,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
        ),
        "lightgbm24": LightGBMForecastStrategy(
            lightgbm_model,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
        ),
        "ppo": ppo_strategy,
    }
    return compare_strategies(
        dataset,
        strategies,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=gross_budget,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk=risk,
    )


__all__ = ["LeanCandidateConfig", "run_lean_candidate_suite"]
