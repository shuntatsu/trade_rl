"""Deterministic policy and schedule construction for the residual environment."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.risk.emergency import CausalEmergencyRiskMonitor
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.rewards import RewardConfig


@dataclass(frozen=True, slots=True)
class EnvironmentPolicyScheduleContract:
    """Resolved immutable policy and episode-schedule construction values."""

    config: ResidualMarketEnvConfig
    emergency_risk_monitor: CausalEmergencyRiskMonitor
    action_spec: ActionSpec
    action_names: tuple[str, ...]
    nominal_episode_bars: int
    nominal_decision_bars: int
    reward_config: RewardConfig
    resolved_decision_hours: float


class EnvironmentPolicyScheduleContractBuilder:
    """Build policy and schedule values while preserving validation order."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        pre_trade_risk: PreTradeRisk,
        alpha_enabled: bool,
        factor_count: int,
        action_spec: ActionSpec | None,
        config: ResidualMarketEnvConfig | None,
    ) -> None:
        self.dataset = dataset
        self.pre_trade_risk = pre_trade_risk
        self.alpha_enabled = alpha_enabled
        self.factor_count = factor_count
        self.action_spec = action_spec
        self.config = config

    def build(self) -> EnvironmentPolicyScheduleContract:
        config = self.config or ResidualMarketEnvConfig()
        emergency_risk_monitor = CausalEmergencyRiskMonitor(config.emergency_risk)
        if self.pre_trade_risk.config.max_gross > config.execution_cost.max_leverage:
            raise ValueError("pre-trade max_gross cannot exceed execution max_leverage")
        if config.random_initial_gross > self.pre_trade_risk.config.max_gross:
            raise ValueError("random_initial_gross cannot exceed pre-trade max_gross")

        action_spec = self.action_spec
        if action_spec is None:
            action_spec = ActionSpec(
                alpha_enabled=self.alpha_enabled,
                n_factors=self.factor_count,
                validation_mode=config.action_validation_mode,
            )
        if action_spec.alpha_enabled != self.alpha_enabled:
            raise ValueError("action_spec alpha mode does not match environment")
        if action_spec.n_factors != self.factor_count:
            raise ValueError("action_spec factor count does not match environment")
        if (
            action_spec.mode is ActionMode.TARGET_WEIGHT
            and action_spec.target_weight_count != self.dataset.n_symbols
        ):
            raise ValueError("target weight count does not match dataset symbols")

        action_names = action_spec.names_for_symbols(self.dataset.symbols)
        nominal_episode_bars = config.resolve_nominal_episode_bars(self.dataset)
        nominal_decision_bars = config.resolve_nominal_decision_bars(self.dataset)
        if nominal_decision_bars > nominal_episode_bars:
            raise ValueError("decision interval cannot exceed episode duration")

        reward_config = config.resolved_reward_config()
        resolved_decision_hours = (
            nominal_decision_bars * self.dataset.bar_hours
            if config.decision_every is not None
            else config.decision_hours
        )
        if config.episode_hour_choices and any(
            choice + 1e-12 < resolved_decision_hours
            for choice in config.episode_hour_choices
        ):
            raise ValueError(
                "episode_hour_choices cannot be shorter than the resolved "
                "decision interval"
            )

        return EnvironmentPolicyScheduleContract(
            config=config,
            emergency_risk_monitor=emergency_risk_monitor,
            action_spec=action_spec,
            action_names=action_names,
            nominal_episode_bars=nominal_episode_bars,
            nominal_decision_bars=nominal_decision_bars,
            reward_config=reward_config,
            resolved_decision_hours=resolved_decision_hours,
        )


__all__ = [
    "EnvironmentPolicyScheduleContract",
    "EnvironmentPolicyScheduleContractBuilder",
]
