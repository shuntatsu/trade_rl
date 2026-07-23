"""Typed construction of the residual environment runtime service graph."""

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
from trade_rl.rl.environment_observation_contract import EnvironmentObservationContract
from trade_rl.rl.environment_reward import EnvironmentRewardCoordinator
from trade_rl.rl.environment_risk import EnvironmentRiskProjector
from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.rewards import RewardTracker
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.simulation import MarketExecutor


@dataclass(frozen=True, slots=True)
class EnvironmentRuntimeServices:
    """Resolved runtime services consumed by the environment facade."""

    episode_sampler: EpisodeContractSampler
    execution_coordinator: EnvironmentExecutionCoordinator
    observation_assembler: EnvironmentObservationAssembler
    decision_planner: EnvironmentDecisionPlanner
    risk_projector: EnvironmentRiskProjector
    reward_coordinator: EnvironmentRewardCoordinator
    info_builder: EnvironmentInfoBuilder
    termination_coordinator: EnvironmentTerminationCoordinator


class EnvironmentRuntimeServicesBuilder:
    """Build the existing runtime service graph from validated collaborators."""

    def __init__(
        self,
        dataset: MarketDataset,
        config: ResidualMarketEnvConfig,
        *,
        minimum_start_index: int,
        observation_contract: EnvironmentObservationContract,
        normalizer: ObservationNormalizer | None,
        sequence_normalizer: SequenceFeatureNormalizer | None,
        action_spec: ActionSpec,
        composer: BaselineResidualComposer,
        pre_trade_risk: PreTradeRisk,
        alpha_enabled: bool,
        emergency_risk_monitor: CausalEmergencyRiskMonitor,
        portfolio_risk: PortfolioRiskModel,
        portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None,
        reward_tracker: RewardTracker,
        hybrid_executor: MarketExecutor,
        shadow_executor: MarketExecutor,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.minimum_start_index = minimum_start_index
        self.observation_contract = observation_contract
        self.normalizer = normalizer
        self.sequence_normalizer = sequence_normalizer
        self.action_spec = action_spec
        self.composer = composer
        self.pre_trade_risk = pre_trade_risk
        self.alpha_enabled = alpha_enabled
        self.emergency_risk_monitor = emergency_risk_monitor
        self.portfolio_risk = portfolio_risk
        self.portfolio_risk_inputs_provider = portfolio_risk_inputs_provider
        self.reward_tracker = reward_tracker
        self.hybrid_executor = hybrid_executor
        self.shadow_executor = shadow_executor

    def build(self) -> EnvironmentRuntimeServices:
        episode_sampler = EpisodeContractSampler(
            self.dataset,
            self.config,
            minimum_start_index=self.minimum_start_index,
        )
        execution_coordinator = EnvironmentExecutionCoordinator(
            self.dataset,
            self.config.execution_cost,
            initial_capital=self.config.initial_capital,
        )
        observation_assembler = EnvironmentObservationAssembler(
            self.dataset,
            observation_builder=self.observation_contract.observation_builder,
            layout=self.observation_contract.layout,
            normalizer=self.normalizer,
            sequence_observation_builder=(
                self.observation_contract.sequence_observation_builder
            ),
            sequence_policy_plane=self.observation_contract.sequence_policy_plane,
            sequence_normalizer=self.sequence_normalizer,
            action_size=self.action_spec.size,
            n_factors=self.action_spec.n_factors,
            finite_horizon=self.config.finite_horizon_observation,
        )
        decision_planner = EnvironmentDecisionPlanner(
            self.dataset,
            action_spec=self.action_spec,
            composer=self.composer,
            max_gross=self.pre_trade_risk.config.max_gross,
            alpha_enabled=self.alpha_enabled,
            accept_legacy_actions=self.config.accept_legacy_actions,
            signal_delay_decisions=self.config.signal_delay_decisions,
            decision_every=self.config.decision_every,
            decision_hours=self.config.decision_hours,
        )
        risk_projector = EnvironmentRiskProjector(
            self.dataset,
            emergency_risk_monitor=self.emergency_risk_monitor,
            pre_trade_risk=self.pre_trade_risk,
            portfolio_risk=self.portfolio_risk,
            portfolio_risk_inputs_provider=self.portfolio_risk_inputs_provider,
        )
        reward_coordinator = EnvironmentRewardCoordinator(
            self.reward_tracker,
            initial_capital=self.config.initial_capital,
        )
        info_builder = EnvironmentInfoBuilder(
            self.dataset,
            self.reward_tracker,
        )
        termination_coordinator = EnvironmentTerminationCoordinator(
            config=self.config,
            reward_tracker=self.reward_tracker,
            pre_trade_risk=self.pre_trade_risk,
            hybrid_executor=self.hybrid_executor,
            shadow_executor=self.shadow_executor,
            execution_coordinator=execution_coordinator,
        )
        return EnvironmentRuntimeServices(
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
    "EnvironmentRuntimeServices",
    "EnvironmentRuntimeServicesBuilder",
]
