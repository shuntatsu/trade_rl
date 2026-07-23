"""Construction-stable service assembly for ``ResidualMarketEnv``."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.risk.emergency import CausalEmergencyRiskMonitor
from trade_rl.risk.inputs import PortfolioRiskInputsProvider
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.actions import ActionSpec, BaselineResidualComposer
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_decision import EnvironmentDecisionPlanner
from trade_rl.rl.environment_episode import EpisodeContractSampler
from trade_rl.rl.environment_execution import EnvironmentExecutionCoordinator
from trade_rl.rl.environment_info import EnvironmentInfoBuilder
from trade_rl.rl.environment_observation import EnvironmentObservationAssembler
from trade_rl.rl.environment_reward import EnvironmentRewardCoordinator
from trade_rl.rl.environment_risk import EnvironmentRiskProjector
from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import ObservationBuilder, ObservationLayout
from trade_rl.rl.rewards import RewardTracker
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequencePolicyPlane,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionRuleStress


@dataclass(frozen=True, slots=True)
class EnvironmentServiceAssemblyRequest:
    dataset: MarketDataset
    config: ResidualMarketEnvConfig
    execution_rule_stress: ExecutionRuleStress | None
    minimum_start_index: int
    action_spec: ActionSpec
    composer: BaselineResidualComposer
    pre_trade_risk: PreTradeRisk
    portfolio_risk: PortfolioRiskModel
    portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None
    alpha_enabled: bool
    reward_tracker: RewardTracker
    observation_builder: ObservationBuilder
    layout: ObservationLayout
    normalizer: ObservationNormalizer | None
    sequence_observation_builder: SequenceObservationBuilder | None
    sequence_policy_plane: SequencePolicyPlane | None
    sequence_normalizer: SequenceFeatureNormalizer | None


@dataclass(frozen=True, slots=True)
class EnvironmentServiceAssembly:
    emergency_risk_monitor: CausalEmergencyRiskMonitor
    hybrid_executor: MarketExecutor
    shadow_executor: MarketExecutor
    episode_sampler: EpisodeContractSampler
    execution_coordinator: EnvironmentExecutionCoordinator
    observation_assembler: EnvironmentObservationAssembler
    decision_planner: EnvironmentDecisionPlanner
    risk_projector: EnvironmentRiskProjector
    reward_coordinator: EnvironmentRewardCoordinator
    info_builder: EnvironmentInfoBuilder
    termination_coordinator: EnvironmentTerminationCoordinator


class EnvironmentServiceAssembler:
    """Compose maintained environment collaborators without retaining state."""

    @staticmethod
    def assemble(
        request: EnvironmentServiceAssemblyRequest,
    ) -> EnvironmentServiceAssembly:
        dataset = request.dataset
        config = request.config
        emergency_risk_monitor = CausalEmergencyRiskMonitor(config.emergency_risk)
        hybrid_executor = MarketExecutor(
            dataset,
            config.execution_cost,
            rule_stress=request.execution_rule_stress,
        )
        shadow_executor = MarketExecutor(
            dataset,
            config.execution_cost,
            rule_stress=request.execution_rule_stress,
        )
        episode_sampler = EpisodeContractSampler(
            dataset,
            config,
            minimum_start_index=request.minimum_start_index,
        )
        execution_coordinator = EnvironmentExecutionCoordinator(
            dataset,
            config.execution_cost,
            initial_capital=config.initial_capital,
        )
        observation_assembler = EnvironmentObservationAssembler(
            dataset,
            observation_builder=request.observation_builder,
            layout=request.layout,
            normalizer=request.normalizer,
            sequence_observation_builder=request.sequence_observation_builder,
            sequence_policy_plane=request.sequence_policy_plane,
            sequence_normalizer=request.sequence_normalizer,
            action_size=request.action_spec.size,
            n_factors=request.action_spec.n_factors,
            finite_horizon=config.finite_horizon_observation,
        )
        decision_planner = EnvironmentDecisionPlanner(
            dataset,
            action_spec=request.action_spec,
            composer=request.composer,
            max_gross=request.pre_trade_risk.config.max_gross,
            alpha_enabled=request.alpha_enabled,
            accept_legacy_actions=config.accept_legacy_actions,
            signal_delay_decisions=config.signal_delay_decisions,
            decision_every=config.decision_every,
            decision_hours=config.decision_hours,
        )
        risk_projector = EnvironmentRiskProjector(
            dataset,
            emergency_risk_monitor=emergency_risk_monitor,
            pre_trade_risk=request.pre_trade_risk,
            portfolio_risk=request.portfolio_risk,
            portfolio_risk_inputs_provider=request.portfolio_risk_inputs_provider,
        )
        reward_coordinator = EnvironmentRewardCoordinator(
            request.reward_tracker,
            initial_capital=config.initial_capital,
        )
        info_builder = EnvironmentInfoBuilder(dataset, request.reward_tracker)
        termination_coordinator = EnvironmentTerminationCoordinator(
            config=config,
            reward_tracker=request.reward_tracker,
            pre_trade_risk=request.pre_trade_risk,
            hybrid_executor=hybrid_executor,
            shadow_executor=shadow_executor,
            execution_coordinator=execution_coordinator,
        )
        return EnvironmentServiceAssembly(
            emergency_risk_monitor=emergency_risk_monitor,
            hybrid_executor=hybrid_executor,
            shadow_executor=shadow_executor,
            episode_sampler=episode_sampler,
            execution_coordinator=execution_coordinator,
            observation_assembler=observation_assembler,
            decision_planner=decision_planner,
            risk_projector=risk_projector,
            reward_coordinator=reward_coordinator,
            info_builder=info_builder,
            termination_coordinator=termination_coordinator,
        )


__all__ = [
    "EnvironmentServiceAssembler",
    "EnvironmentServiceAssembly",
    "EnvironmentServiceAssemblyRequest",
]
