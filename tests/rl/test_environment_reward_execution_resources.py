from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_reward_execution_resources import (
    EnvironmentRewardExecutionResourcesBuilder,
)
from trade_rl.rl.episode import minimum_reward_start_index
from trade_rl.rl.rewards import RewardConfig
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionRuleStress,
)
from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 200
    close = np.column_stack(
        [
            np.linspace(100.0, 140.0, n_bars),
            np.linspace(80.0, 70.0, n_bars),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def reward_config(*, baseline_weight: float) -> RewardConfig:
    return RewardConfig(
        baseline_underperformance_weight=baseline_weight,
        baseline_window_hours=24.0,
        baseline_minimum_history_hours=12.0,
    )


def config(
    *,
    reward: RewardConfig,
    require_full_reward_preroll: bool = False,
) -> ResidualMarketEnvConfig:
    return ResidualMarketEnvConfig(
        initial_capital=100_000.0,
        episode_bars=16,
        decision_every=2,
        reward_config=reward,
        require_full_reward_preroll=require_full_reward_preroll,
        execution_cost=ExecutionCostConfig.zero(),
    )


def test_builder_returns_fresh_equivalent_resources_without_preroll() -> None:
    dataset = market()
    resolved_reward = reward_config(baseline_weight=0.0)
    resolved_config = config(reward=resolved_reward)
    stress = ExecutionRuleStress(name="identity-test")
    builder = EnvironmentRewardExecutionResourcesBuilder(
        dataset,
        config=resolved_config,
        reward_config=resolved_reward,
        resolved_decision_hours=2.0,
        minimum_start_index=8,
        execution_rule_stress=stress,
    )

    first = builder.build()
    second = builder.build()

    assert first.minimum_start_index == 8
    assert first.reward_tracker is not second.reward_tracker
    assert first.reward_tracker.config is resolved_reward
    assert first.reward_tracker.decision_hours == 2.0
    assert first.reward_tracker.baseline_window_steps == 12
    assert first.reward_tracker.baseline_minimum_history_steps == 6

    assert first.hybrid_executor is not first.shadow_executor
    assert first.hybrid_executor is not second.hybrid_executor
    assert first.shadow_executor is not second.shadow_executor
    assert first.executor is first.hybrid_executor
    assert first.hybrid_executor.dataset is dataset
    assert first.shadow_executor.dataset is dataset
    assert first.hybrid_executor.cost is resolved_config.execution_cost
    assert first.shadow_executor.cost is resolved_config.execution_cost
    assert first.hybrid_executor.rule_stress is stress
    assert first.shadow_executor.rule_stress is stress
    assert (
        first.hybrid_executor.execution_policy_digest
        == first.shadow_executor.execution_policy_digest
    )

    assert first.reward_history_cache == {}
    assert second.reward_history_cache == {}
    assert first.reward_history_cache is not second.reward_history_cache


def test_builder_derives_full_reward_preroll_minimum() -> None:
    dataset = market()
    resolved_reward = reward_config(baseline_weight=0.10)
    resolved_config = config(
        reward=resolved_reward,
        require_full_reward_preroll=True,
    )
    expected = minimum_reward_start_index(
        dataset,
        signal_minimum=8,
        window_hours=resolved_reward.baseline_window_hours,
    )

    resources = EnvironmentRewardExecutionResourcesBuilder(
        dataset,
        config=resolved_config,
        reward_config=resolved_reward,
        resolved_decision_hours=2.0,
        minimum_start_index=8,
        execution_rule_stress=None,
    ).build()

    assert resources.minimum_start_index == expected
    assert resources.minimum_start_index > 8


def test_reward_tracker_validation_precedes_downstream_resource_construction() -> None:
    dataset = market()
    resolved_reward = reward_config(baseline_weight=0.10)
    resolved_config = config(
        reward=resolved_reward,
        require_full_reward_preroll=True,
    )

    with pytest.raises(ValueError, match="decision_hours must be finite and positive"):
        EnvironmentRewardExecutionResourcesBuilder(
            dataset,
            config=resolved_config,
            reward_config=resolved_reward,
            resolved_decision_hours=0.0,
            minimum_start_index=-1,
            execution_rule_stress=None,
        ).build()


def test_environment_preserves_reward_and_executor_attributes() -> None:
    dataset = market()
    resolved_reward = reward_config(baseline_weight=0.0)
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        config=config(reward=resolved_reward),
    )

    assert env.reward_tracker.config is resolved_reward
    assert env.reward_tracker.decision_hours == env._resolved_decision_hours
    assert env.executor is env.hybrid_executor
    assert env.hybrid_executor is not env.shadow_executor
    assert env.hybrid_executor.execution_policy_digest == (
        env.shadow_executor.execution_policy_digest
    )
    assert env._reward_history_cache == {}


def test_builder_applies_identical_environment_stress_to_both_books() -> None:
    dataset = market()
    resolved_reward = reward_config(baseline_weight=0.0)
    base_cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.002,
        impact_rate=0.003,
        max_participation_rate=0.2,
        slippage_std=0.004,
        tail_slippage_probability=0.0,
        tail_slippage_multiplier=5.0,
        borrow_rate_multiplier=1.5,
        order_latency_bars=0,
    )
    resolved_config = replace(
        config(reward=resolved_reward),
        execution_cost=base_cost,
    )
    stress = ExecutionEnvironmentStress(
        name="joint-adverse",
        fee_multiplier=2.0,
        spread_multiplier=2.0,
        impact_multiplier=2.0,
        slippage_std_multiplier=2.0,
        participation_fraction=0.5,
        minimum_order_latency_bars=2,
        tail_slippage_probability_floor=0.01,
        tail_slippage_multiplier_floor=10.0,
        borrow_rate_multiplier=2.0,
    )

    resources = EnvironmentRewardExecutionResourcesBuilder(
        dataset,
        config=resolved_config,
        reward_config=resolved_reward,
        resolved_decision_hours=2.0,
        minimum_start_index=8,
        execution_rule_stress=stress,
    ).build()

    assert resources.hybrid_executor.cost is resources.shadow_executor.cost
    assert resources.hybrid_executor.cost is not base_cost
    assert resources.hybrid_executor.cost.fee_rate == pytest.approx(0.002)
    assert resources.hybrid_executor.cost.spread_rate == pytest.approx(0.004)
    assert resources.hybrid_executor.cost.impact_rate == pytest.approx(0.006)
    assert resources.hybrid_executor.cost.max_participation_rate == pytest.approx(0.1)
    assert resources.hybrid_executor.cost.order_latency_bars == 2
    assert resources.hybrid_executor.cost.tail_slippage_probability == (
        pytest.approx(0.01)
    )
    assert resources.hybrid_executor.cost.tail_slippage_multiplier == (
        pytest.approx(10.0)
    )
    assert resources.hybrid_executor.rule_stress is stress
    assert resources.shadow_executor.rule_stress is stress
    assert resources.hybrid_executor.execution_policy_digest == (
        resources.shadow_executor.execution_policy_digest
    )
