"""Fixed M2 candidate suite on the shared lean replay contract."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import (
    SharedCashStrategyComparisonEntry,
    UniversalStrategyComparison,
    compare_strategy_factories_by_symbol,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
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
from trade_rl.strategies.rl.intent import (
    PPO_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA_V3,
    PPO_OBSERVATION_SCHEMAS,
)
from trade_rl.strategies.rl.ppo import (
    PPO_TRAINING_LAYOUT_INTERLEAVED,
    PPO_TRAINING_LAYOUT_SEQUENTIAL,
    PPOIntentStrategy,
    fit_ppo_strategy,
)
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy


@dataclass(frozen=True, slots=True)
class LeanCandidateConfig:
    """Only the pre-registered degrees of freedom for the initial M2 suite."""

    signal_index: int
    feature_indices: tuple[int, ...]
    fit_symbol_indices: tuple[int, ...]
    fit_cutoff: np.datetime64
    rule_entry_threshold: float
    rule_exit_threshold: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    ppo_total_timesteps: int
    ppo_seed: int = 0
    ppo_training_layout: str = PPO_TRAINING_LAYOUT_SEQUENTIAL
    ppo_rollout_steps_per_env: int | None = None
    ppo_minimum_hold_bars: int = 0
    ppo_observation_schema: str = PPO_OBSERVATION_SCHEMA
    ppo_settle_terminal_position: bool = False
    forecast_switch_cost: float | None = None

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
        fit_symbols = tuple(self.fit_symbol_indices)
        if not fit_symbols or len(set(fit_symbols)) != len(fit_symbols):
            raise ValueError("fit_symbol_indices must be non-empty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in fit_symbols
        ):
            raise ValueError("fit_symbol_indices must contain non-negative integers")
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
        if not isinstance(
            self.ppo_training_layout, str
        ) or self.ppo_training_layout not in {
            PPO_TRAINING_LAYOUT_SEQUENTIAL,
            PPO_TRAINING_LAYOUT_INTERLEAVED,
        }:
            raise ValueError("unsupported ppo_training_layout")
        if self.ppo_training_layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
            if self.ppo_rollout_steps_per_env is not None:
                raise ValueError(
                    "sequential training does not accept ppo_rollout_steps_per_env"
                )
        elif (
            isinstance(self.ppo_rollout_steps_per_env, bool)
            or not isinstance(self.ppo_rollout_steps_per_env, int)
            or self.ppo_rollout_steps_per_env <= 0
        ):
            raise ValueError(
                "interleaved training requires positive ppo_rollout_steps_per_env"
            )
        if (
            isinstance(self.ppo_minimum_hold_bars, bool)
            or not isinstance(self.ppo_minimum_hold_bars, int)
            or self.ppo_minimum_hold_bars < 0
        ):
            raise ValueError("ppo_minimum_hold_bars must be a non-negative integer")
        if self.ppo_observation_schema not in PPO_OBSERVATION_SCHEMAS:
            raise ValueError("unsupported PPO observation schema")
        if (
            self.ppo_minimum_hold_bars > 0
            and self.ppo_observation_schema != PPO_OBSERVATION_SCHEMA_V3
        ):
            raise ValueError("PPO minimum hold requires the age-aware observation")
        if not isinstance(self.ppo_settle_terminal_position, bool):
            raise ValueError("ppo_settle_terminal_position must be boolean")
        forecast_switch_cost = self.forecast_switch_cost
        if forecast_switch_cost is not None:
            if isinstance(forecast_switch_cost, bool) or not isinstance(
                forecast_switch_cost, (int, float)
            ):
                raise ValueError("forecast_switch_cost must be finite and non-negative")
            try:
                forecast_switch_cost = float(forecast_switch_cost)
            except OverflowError as error:
                raise ValueError(
                    "forecast_switch_cost must be finite and non-negative"
                ) from error
            if not math.isfinite(forecast_switch_cost) or forecast_switch_cost < 0.0:
                raise ValueError("forecast_switch_cost must be finite and non-negative")
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "fit_symbol_indices", fit_symbols)
        object.__setattr__(self, "fit_cutoff", np.datetime64(self.fit_cutoff, "ns"))
        object.__setattr__(self, "forecast_switch_cost", forecast_switch_cost)


def _ppo_training_stop_index(
    dataset: MarketDataset,
    fit_cutoff: np.datetime64,
) -> int:
    eligible = np.flatnonzero(dataset.timestamps < fit_cutoff)
    if eligible.size < 2:
        raise ValueError("PPO fitting requires at least two bars before fit_cutoff")
    return int(eligible[-1])


def require_age_aware_hourly_clock(
    dataset: MarketDataset,
    *,
    observation_schema: str,
) -> None:
    """Require an exact hourly clock for bar-count PPO duration semantics."""

    if observation_schema != PPO_OBSERVATION_SCHEMA_V3:
        return
    if not dataset.regular_cadence or not math.isclose(
        dataset.bar_hours,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "age-aware PPO comparison requires exactly regular one-hour bars"
        )


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
    include_ppo_shared_cash_replay: bool = False,
) -> UniversalStrategyComparison:
    """Fit one universal candidate set and compare it independently by symbol."""

    if config.ppo_observation_schema == PPO_OBSERVATION_SCHEMA_V3:
        if not config.ppo_settle_terminal_position:
            raise ValueError("age-aware PPO comparison requires terminal settlement")
        if risk is None:
            raise ValueError(
                "age-aware PPO comparison requires explicit pre-trade risk config"
            )
        if risk.config.drawdown_stop > 0.20:
            raise ValueError("PPO drawdown stop must not exceed 20%")
    if not isinstance(include_ppo_shared_cash_replay, bool):
        raise ValueError("include_ppo_shared_cash_replay must be boolean")
    if include_ppo_shared_cash_replay and (
        config.ppo_observation_schema != PPO_OBSERVATION_SCHEMA_V3
        or not config.ppo_settle_terminal_position
        or risk is None
    ):
        raise ValueError(
            "shared-cash PPO replay requires age-aware observations, "
            "terminal settlement, and explicit risk"
        )
    require_age_aware_hourly_clock(
        dataset,
        observation_schema=config.ppo_observation_schema,
    )
    if config.signal_index >= dataset.n_features:
        raise ValueError("signal_index is outside dataset features")
    if max(config.feature_indices) >= dataset.n_features:
        raise ValueError("feature index is outside dataset features")
    if max(config.fit_symbol_indices) >= dataset.n_symbols:
        raise ValueError("fit symbol index is outside dataset symbols")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("evaluation range must satisfy 0 <= start < stop < n_bars")
    if dataset.timestamps[start_index] < config.fit_cutoff:
        raise ValueError("evaluation must not start before fit_cutoff")

    ridge_model = fit_ridge_forecast(
        dataset,
        feature_indices=config.feature_indices,
        fit_symbol_indices=config.fit_symbol_indices,
        fit_cutoff=config.fit_cutoff,
        horizon_hours=24,
        alpha=1.0,
    )
    lightgbm_model = fit_lightgbm_forecast(
        dataset,
        feature_indices=config.feature_indices,
        fit_symbol_indices=config.fit_symbol_indices,
        fit_cutoff=config.fit_cutoff,
        horizon_hours=24,
        random_state=0,
    )
    ppo_strategy = fit_ppo_strategy(
        dataset,
        feature_indices=config.feature_indices,
        fit_symbol_indices=config.fit_symbol_indices,
        start_index=0,
        stop_index=_ppo_training_stop_index(dataset, config.fit_cutoff),
        gross_budget=gross_budget,
        total_timesteps=config.ppo_total_timesteps,
        seed=config.ppo_seed,
        training_layout=config.ppo_training_layout,
        rollout_steps_per_env=config.ppo_rollout_steps_per_env,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk_config=None if risk is None else risk.config,
        settle_terminal_position=config.ppo_settle_terminal_position,
        minimum_hold_bars=config.ppo_minimum_hold_bars,
        observation_schema=config.ppo_observation_schema,
    )
    ppo_training_timesteps = getattr(ppo_strategy.policy, "num_timesteps", None)
    if (
        isinstance(ppo_training_timesteps, bool)
        or not isinstance(ppo_training_timesteps, int)
        or ppo_training_timesteps <= 0
    ):
        raise ValueError("fitted PPO policy has invalid realized timesteps")

    trend_config = TrendIntentConfig(
        signal_index=config.signal_index,
        entry_threshold=config.rule_entry_threshold,
        exit_threshold=config.rule_exit_threshold,
    )
    mean_reversion_config = MeanReversionIntentConfig(
        signal_index=config.signal_index,
        entry_threshold=config.rule_entry_threshold,
        exit_threshold=config.rule_exit_threshold,
    )
    strategy_factories: dict[str, Callable[[], SingleSymbolStrategy]] = {
        "cash": lambda: ConstantIntentStrategy(PositionIntent.FLAT),
        "constant_long": lambda: ConstantIntentStrategy(PositionIntent.LONG),
        "constant_short": lambda: ConstantIntentStrategy(PositionIntent.SHORT),
        "trend": lambda: TrendIntentStrategy(trend_config),
        "mean_reversion": lambda: MeanReversionIntentStrategy(mean_reversion_config),
        "ridge24": lambda: RidgeForecastStrategy(
            ridge_model,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
            one_way_switch_cost=config.forecast_switch_cost,
        ),
        "lightgbm24": lambda: LightGBMForecastStrategy(
            lightgbm_model,
            entry_threshold=config.forecast_entry_threshold,
            exit_threshold=config.forecast_exit_threshold,
            one_way_switch_cost=config.forecast_switch_cost,
        ),
        "ppo": lambda: PPOIntentStrategy(
            ppo_strategy.policy,
            feature_indices=ppo_strategy.feature_indices,
            feature_names=ppo_strategy.feature_names,
            feature_normalizer=ppo_strategy.feature_normalizer,
            observation_schema=ppo_strategy.observation_schema,
            minimum_hold_bars=config.ppo_minimum_hold_bars,
            training_minimum_hold_suppressed_count=(
                ppo_strategy.training_minimum_hold_suppressed_count
            ),
        ),
    }
    comparison = compare_strategy_factories_by_symbol(
        dataset,
        strategy_factories,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=gross_budget,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk=risk,
        settle_terminal_position=config.ppo_settle_terminal_position,
    )
    shared_cash_ppo: SharedCashStrategyComparisonEntry | None = None
    if include_ppo_shared_cash_replay:
        shared_strategies = tuple(strategy_factories["ppo"]() for _ in dataset.symbols)
        shared_replay = run_shared_cash_replay(
            dataset,
            shared_strategies,
            start_index=start_index,
            stop_index=stop_index,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk=risk,
            minimum_hold_bars=config.ppo_minimum_hold_bars,
            settle_terminal_position=config.ppo_settle_terminal_position,
            capture_ledger_evidence=True,
        )
        diagnostics = shared_replay.diagnostics
        shared_cash_ppo = SharedCashStrategyComparisonEntry(
            name="ppo",
            replay=shared_replay,
            metrics=evaluate_performance(
                shared_replay.returns,
                turnover_total=diagnostics.turnover_total,
                total_cost=diagnostics.total_cost,
                funding_pnl=diagnostics.funding_pnl,
                borrow_cost=diagnostics.borrow_cost,
                n_trades=diagnostics.n_trades,
                rebalance_events=diagnostics.rebalance_events,
                termination_count=diagnostics.termination_count,
            ),
        )
    return replace(
        comparison,
        shared_cash_ppo=shared_cash_ppo,
        ppo_training_timesteps=ppo_training_timesteps,
        ppo_training_minimum_hold_suppressed_count=(
            ppo_strategy.training_minimum_hold_suppressed_count
        ),
    )


__all__ = [
    "LeanCandidateConfig",
    "require_age_aware_hourly_clock",
    "run_lean_candidate_suite",
]
