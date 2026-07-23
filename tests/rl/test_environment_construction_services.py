from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment_assembly import (
    EnvironmentServiceAssembler,
    EnvironmentServiceAssemblyRequest,
)
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_dependencies import (
    EnvironmentDependencyRequest,
    EnvironmentDependencyResolver,
)
from trade_rl.rl.environment_observation_contract import (
    EnvironmentObservationContractFactory,
    EnvironmentObservationContractRequest,
)
from trade_rl.rl.environment_state import (
    EnvironmentInitialStateFactory,
    EnvironmentInitialStateRequest,
)
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


def config() -> ResidualMarketEnvConfig:
    return ResidualMarketEnvConfig(
        initial_capital=100_000.0,
        episode_bars=8,
        decision_every=2,
        reward=AbsoluteGrowthRewardConfig(),
        execution_cost=ExecutionCostConfig.zero(),
    )


def resolve_dependencies(*, alpha_enabled: bool = False):
    dataset = market()
    return dataset, EnvironmentDependencyResolver.resolve(
        EnvironmentDependencyRequest(
            dataset=dataset,
            trend_strategy=TrendStrategy(
                TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
            ),
            market_input_resolver=None,
            alpha_provider=None,
            alpha_enabled=alpha_enabled,
            alpha_artifact_digest=None,
            alpha_contract=None,
            factor_basis=None,
            factor_basis_provider=None,
            factor_artifact_digest=None,
            factor_count=None,
            action_spec=None,
            composer=None,
            pre_trade_risk=None,
            portfolio_risk=None,
            portfolio_risk_inputs_provider=None,
            config=config(),
        )
    )


def test_dependency_resolver_preserves_default_constructor_contract() -> None:
    dataset, resolved = resolve_dependencies()

    assert resolved.action_names == resolved.action_spec.names_for_symbols(
        dataset.symbols
    )
    assert len(resolved.action_spec_digest) == 64
    assert resolved.nominal_episode_bars == 8
    assert resolved.nominal_decision_bars == 2
    assert resolved.resolved_decision_hours == pytest.approx(2.0)
    assert resolved.minimum_start_index == 8


def test_dependency_resolver_rejects_enabled_alpha_without_provider() -> None:
    with pytest.raises(ValueError, match="alpha_enabled requires an alpha_provider"):
        resolve_dependencies(alpha_enabled=True)


def test_observation_factory_builds_flat_contract_without_mutating_inputs() -> None:
    dataset, dependencies = resolve_dependencies()
    contract = EnvironmentObservationContractFactory.build(
        EnvironmentObservationContractRequest(
            dataset=dataset,
            config=dependencies.config,
            action_spec=dependencies.action_spec,
            action_spec_digest=dependencies.action_spec_digest,
            alpha_artifact_digest=dependencies.alpha_artifact_digest,
            factor_artifact_digest=dependencies.factor_artifact_digest,
            normalizer=None,
            sequence_normalizer=None,
            minimum_start_index=dependencies.minimum_start_index,
        )
    )

    assert isinstance(contract.observation_space, spaces.Box)
    assert contract.observation_space.shape == (contract.layout.size,)
    assert contract.action_space.shape == (dependencies.action_spec.size,)
    assert contract.minimum_start_index == dependencies.minimum_start_index
    assert len(contract.observation_contract_digest) == 64


def test_service_assembly_creates_independent_equivalent_executors() -> None:
    dataset, dependencies = resolve_dependencies()
    observation = EnvironmentObservationContractFactory.build(
        EnvironmentObservationContractRequest(
            dataset=dataset,
            config=dependencies.config,
            action_spec=dependencies.action_spec,
            action_spec_digest=dependencies.action_spec_digest,
            alpha_artifact_digest=dependencies.alpha_artifact_digest,
            factor_artifact_digest=dependencies.factor_artifact_digest,
            normalizer=None,
            sequence_normalizer=None,
            minimum_start_index=dependencies.minimum_start_index,
        )
    )
    assembly = EnvironmentServiceAssembler.assemble(
        EnvironmentServiceAssemblyRequest(
            dataset=dataset,
            config=dependencies.config,
            execution_rule_stress=None,
            minimum_start_index=observation.minimum_start_index,
            action_spec=dependencies.action_spec,
            composer=dependencies.composer,
            pre_trade_risk=dependencies.pre_trade_risk,
            portfolio_risk=dependencies.portfolio_risk,
            portfolio_risk_inputs_provider=dependencies.portfolio_risk_inputs_provider,
            alpha_enabled=dependencies.alpha_enabled,
            reward_tracker=dependencies.reward_tracker,
            observation_builder=observation.observation_builder,
            layout=observation.layout,
            normalizer=None,
            sequence_observation_builder=None,
            sequence_policy_plane=None,
            sequence_normalizer=None,
        )
    )

    assert assembly.hybrid_executor is not assembly.shadow_executor
    assert (
        assembly.hybrid_executor.execution_policy_digest
        == assembly.shadow_executor.execution_policy_digest
    )
    assert (
        assembly.episode_sampler.minimum_start_index == observation.minimum_start_index
    )


def test_initial_state_factory_returns_fresh_independent_mutable_state() -> None:
    dataset, dependencies = resolve_dependencies()
    first = EnvironmentInitialStateFactory.create(
        EnvironmentInitialStateRequest(
            dataset=dataset,
            config=dependencies.config,
            action_spec=dependencies.action_spec,
            minimum_start_index=dependencies.minimum_start_index,
        )
    )
    second = EnvironmentInitialStateFactory.create(
        EnvironmentInitialStateRequest(
            dataset=dataset,
            config=dependencies.config,
            action_spec=dependencies.action_spec,
            minimum_start_index=dependencies.minimum_start_index,
        )
    )

    assert first.hybrid is not first.shadow
    assert first.hybrid is not second.hybrid
    assert first.reward_history_cache is not second.reward_history_cache
    assert first.previous_action.dtype == np.float32
    assert first.previous_action.shape == (dependencies.action_spec.size,)
    assert first.position_age.dtype == np.float64
    assert first.position_age.tolist() == [0.0, 0.0]
    assert first.hybrid_order_book.active_orders == ()
    assert first.shadow_order_book.active_orders == ()
