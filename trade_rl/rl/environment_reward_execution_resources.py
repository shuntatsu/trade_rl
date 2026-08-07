"""Reward and execution resource construction for the residual environment."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.episode import minimum_reward_start_index
from trade_rl.rl.rewards import RewardConfig, RewardTracker
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionRuleStress


@dataclass(frozen=True, slots=True)
class EnvironmentRewardExecutionResources:
    """Fresh runtime resources installed on one environment instance."""

    reward_tracker: RewardTracker
    minimum_start_index: int
    hybrid_executor: MarketExecutor
    shadow_executor: MarketExecutor
    executor: MarketExecutor
    reward_history_cache: dict[int, tuple[float, ...]]


class EnvironmentRewardExecutionResourcesBuilder:
    """Build reward and execution resources in the maintained validation order."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        config: ResidualMarketEnvConfig,
        reward_config: RewardConfig,
        resolved_decision_hours: float,
        minimum_start_index: int,
        execution_rule_stress: ExecutionRuleStress | None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.reward_config = reward_config
        self.resolved_decision_hours = resolved_decision_hours
        self.minimum_start_index = minimum_start_index
        self.execution_rule_stress = execution_rule_stress

    def build(self) -> EnvironmentRewardExecutionResources:
        reward_tracker = RewardTracker(
            self.reward_config,
            decision_hours=self.resolved_decision_hours,
        )
        minimum_start_index = self.minimum_start_index
        if (
            self.config.require_full_reward_preroll
            and self.reward_config.baseline_underperformance_weight > 0.0
        ):
            minimum_start_index = minimum_reward_start_index(
                self.dataset,
                signal_minimum=minimum_start_index,
                window_hours=self.reward_config.baseline_window_hours,
            )
        hybrid_executor = MarketExecutor(
            self.dataset,
            self.config.execution_cost,
            rule_stress=self.execution_rule_stress,
        )
        shadow_executor = MarketExecutor(
            self.dataset,
            self.config.execution_cost,
            rule_stress=self.execution_rule_stress,
        )
        return EnvironmentRewardExecutionResources(
            reward_tracker=reward_tracker,
            minimum_start_index=minimum_start_index,
            hybrid_executor=hybrid_executor,
            shadow_executor=shadow_executor,
            executor=hybrid_executor,
            reward_history_cache={},
        )


__all__ = [
    "EnvironmentRewardExecutionResources",
    "EnvironmentRewardExecutionResourcesBuilder",
]
