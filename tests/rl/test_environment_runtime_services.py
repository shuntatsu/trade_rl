from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec, BaselineResidualComposer
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_decision import EnvironmentDecisionPlanner
from trade_rl.rl.environment_episode import EpisodeContractSampler
from trade_rl.rl.environment_execution import EnvironmentExecutionCoordinator
from trade_rl.rl.environment_info import EnvironmentInfoBuilder
from trade_rl.rl.environment_observation import EnvironmentObservationAssembler
from trade_rl.rl.environment_observation_contract import EnvironmentObservationContract
from trade_rl.rl.environment_reward import EnvironmentRewardCoordinator
from trade_rl.rl.environment_risk import EnvironmentRiskProjector
from trade_rl.rl.environment_runtime_services import (
    EnvironmentRuntimeServices,
    EnvironmentRuntimeServicesBuilder,
)
from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
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


def environment() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        action_spec=ActionSpec(),
        composer=BaselineResidualComposer(),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=2,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def observation_contract(env: ResidualMarketEnv) -> EnvironmentObservationContract:
    return EnvironmentObservationContract(
        observation_builder=env.observation_builder,
        layout=env.layout,
        asset_active_column=env.asset_active_column,
        sequence_observation_builder=env.sequence_observation_builder,
        sequence_policy_plane=env.sequence_policy_plane,
        sequence_layout_metadata=env.sequence_layout_metadata,
        observation_schema=env.observation_schema,
        observation_contract_digest=env.observation_contract_digest,
        observation_space=env.observation_space,
        action_space=env.action_space,
        minimum_start_index=env._minimum_start_index,
    )


def test_environment_exposes_the_same_runtime_service_graph() -> None:
    env = environment()

    assert isinstance(env._episode_sampler, EpisodeContractSampler)
    assert isinstance(env._execution_coordinator, EnvironmentExecutionCoordinator)
    assert isinstance(env._observation_assembler, EnvironmentObservationAssembler)
    assert isinstance(env._decision_planner, EnvironmentDecisionPlanner)
    assert isinstance(env._risk_projector, EnvironmentRiskProjector)
    assert isinstance(env._reward_coordinator, EnvironmentRewardCoordinator)
    assert isinstance(env._info_builder, EnvironmentInfoBuilder)
    assert isinstance(env._termination_coordinator, EnvironmentTerminationCoordinator)

    assert env._episode_sampler.dataset is env.dataset
    assert env._episode_sampler.config is env.config
    assert env._episode_sampler.minimum_start_index == env._minimum_start_index
    assert env._execution_coordinator.dataset is env.dataset
    assert env._execution_coordinator.execution_cost is env.config.execution_cost
    assert env._observation_assembler.observation_builder is env.observation_builder
    assert env._observation_assembler.layout is env.layout
    assert env._observation_assembler.normalizer is env.normalizer
    assert env._observation_assembler.sequence_normalizer is env.sequence_normalizer
    assert env._decision_planner.action_spec is env.action_spec
    assert env._decision_planner.composer is env.composer
    assert env._risk_projector.emergency_risk_monitor is env.emergency_risk_monitor
    assert env._risk_projector.pre_trade_risk is env.pre_trade_risk
    assert env._risk_projector.portfolio_risk is env.portfolio_risk
    assert (
        env._risk_projector.portfolio_risk_inputs_provider
        is env.portfolio_risk_inputs_provider
    )
    assert env._reward_coordinator.reward_tracker is env.reward_tracker
    assert env._info_builder.reward_tracker is env.reward_tracker
    assert env._termination_coordinator.reward_tracker is env.reward_tracker
    assert env._termination_coordinator.hybrid_executor is env.hybrid_executor
    assert env._termination_coordinator.shadow_executor is env.shadow_executor
    assert (
        env._termination_coordinator.execution_coordinator is env._execution_coordinator
    )


def test_builder_returns_one_typed_bundle_with_shared_collaborators() -> None:
    env = environment()

    services = EnvironmentRuntimeServicesBuilder(
        env.dataset,
        env.config,
        minimum_start_index=env._minimum_start_index,
        observation_contract=observation_contract(env),
        normalizer=env.normalizer,
        sequence_normalizer=env.sequence_normalizer,
        action_spec=env.action_spec,
        composer=env.composer,
        pre_trade_risk=env.pre_trade_risk,
        alpha_enabled=env.alpha_enabled,
        emergency_risk_monitor=env.emergency_risk_monitor,
        portfolio_risk=env.portfolio_risk,
        portfolio_risk_inputs_provider=env.portfolio_risk_inputs_provider,
        reward_tracker=env.reward_tracker,
        hybrid_executor=env.hybrid_executor,
        shadow_executor=env.shadow_executor,
    ).build()

    assert isinstance(services, EnvironmentRuntimeServices)
    assert services.episode_sampler.minimum_start_index == env._minimum_start_index
    assert services.observation_assembler.observation_builder is env.observation_builder
    assert services.observation_assembler.layout is env.layout
    assert services.decision_planner.action_spec is env.action_spec
    assert services.decision_planner.composer is env.composer
    assert services.risk_projector.pre_trade_risk is env.pre_trade_risk
    assert services.reward_coordinator.reward_tracker is env.reward_tracker
    assert services.info_builder.reward_tracker is env.reward_tracker
    assert services.termination_coordinator.reward_tracker is env.reward_tracker
    assert services.termination_coordinator.hybrid_executor is env.hybrid_executor
    assert services.termination_coordinator.shadow_executor is env.shadow_executor
    assert (
        services.termination_coordinator.execution_coordinator
        is services.execution_coordinator
    )
