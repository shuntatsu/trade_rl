"""Validated environment configuration kept separate from the Gym facade."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trade_rl.data.contracts import timeframe_hours
from trade_rl.data.market import MarketDataset
from trade_rl.risk.emergency import EmergencyRiskConfig
from trade_rl.rl.actions import ActionValidationMode
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig, RewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig

RESET_STATE_MODES = frozenset(
    {"cash", "baseline", "random", "stress", "partial_fill", "restore"}
)
SAMPLED_INITIAL_STATE_MODES = RESET_STATE_MODES - {"restore"}


@dataclass(frozen=True, slots=True)
class ResidualMarketEnvConfig:
    episode_hours: float = 720.0
    decision_hours: float = 4.0
    episode_hour_choices: tuple[float, ...] = ()
    episode_bars: int | None = None
    decision_every: int | None = None
    signal_delay_decisions: int = 0
    reward_scale: float = 100.0
    initial_capital: float = math.nan
    minimum_equity_fraction: float = 1e-6
    downside_penalty: float = 0.0
    excess_drawdown_penalty: float = 0.0
    reward_config: RewardConfig | None = None
    reward: AbsoluteGrowthRewardConfig | None = None
    liquidate_on_end: bool = False
    finite_horizon_observation: bool = False
    structured_sequence_observation: bool = False
    sequence_windows: tuple[tuple[str, int], ...] = ()
    require_full_reward_preroll: bool = False
    initial_state_modes: tuple[str, ...] = ("cash",)
    random_initial_gross: float = 0.50
    stress_drawdown_fraction: float = 0.15
    partial_fill_fraction: float = 0.50
    episode_sampling_mode: str = "uniform"
    regime_feature_index: int = 0
    regime_bins: int = 4
    stress_quantile: float = 0.90
    accept_legacy_actions: bool = True
    action_validation_mode: ActionValidationMode | str = ActionValidationMode.CLIP
    emergency_risk: EmergencyRiskConfig = field(default_factory=EmergencyRiskConfig)
    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)

    def __post_init__(self) -> None:
        if math.isnan(self.initial_capital):
            raise ValueError(
                "initial_capital must be explicitly configured in quote-currency units"
            )
        for positive_field_name, positive_value in (
            ("episode_hours", self.episode_hours),
            ("decision_hours", self.decision_hours),
            ("reward_scale", self.reward_scale),
            ("initial_capital", self.initial_capital),
            ("minimum_equity_fraction", self.minimum_equity_fraction),
        ):
            if (
                isinstance(positive_value, bool)
                or not math.isfinite(positive_value)
                or positive_value <= 0.0
            ):
                raise ValueError(f"{positive_field_name} must be finite and positive")
        for optional_field_name, optional_value in (
            ("episode_bars", self.episode_bars),
            ("decision_every", self.decision_every),
        ):
            if optional_value is not None and (
                isinstance(optional_value, bool)
                or not isinstance(optional_value, int)
                or optional_value <= 0
            ):
                raise ValueError(f"{optional_field_name} must be a positive integer")
        if (
            isinstance(self.signal_delay_decisions, bool)
            or not isinstance(self.signal_delay_decisions, int)
            or self.signal_delay_decisions not in {0, 1}
        ):
            raise ValueError(
                "signal_delay_decisions must be exactly zero or one so the pending "
                "target remains fully observable"
            )
        if self.episode_bars is not None and self.episode_hour_choices:
            raise ValueError(
                "episode_bars cannot be combined with episode_hour_choices"
            )
        for field_name, value in (
            ("downside_penalty", self.downside_penalty),
            ("excess_drawdown_penalty", self.excess_drawdown_penalty),
            ("random_initial_gross", self.random_initial_gross),
            ("stress_drawdown_fraction", self.stress_drawdown_fraction),
            ("partial_fill_fraction", self.partial_fill_fraction),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.random_initial_gross > 10.0:
            raise ValueError("random_initial_gross must be within [0, 10]")
        if not 0.0 <= self.stress_drawdown_fraction < 1.0:
            raise ValueError("stress_drawdown_fraction must be within [0, 1)")
        if not 0.0 <= self.partial_fill_fraction <= 1.0:
            raise ValueError("partial_fill_fraction must be within [0, 1]")
        if self.episode_sampling_mode not in {
            "uniform",
            "regime_balanced",
            "stress_tail",
        }:
            raise ValueError("episode_sampling_mode is not supported")
        if (
            not math.isfinite(self.stress_quantile)
            or not 0.5 <= self.stress_quantile < 1.0
        ):
            raise ValueError("stress_quantile must be within [0.5, 1)")
        if (
            isinstance(self.regime_feature_index, bool)
            or not isinstance(self.regime_feature_index, int)
            or self.regime_feature_index < 0
        ):
            raise ValueError("regime_feature_index must be non-negative")
        if (
            isinstance(self.regime_bins, bool)
            or not isinstance(self.regime_bins, int)
            or self.regime_bins < 2
        ):
            raise ValueError("regime_bins must be at least two")
        for value in self.episode_hour_choices:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("episode_hour_choices must be finite and positive")
            if self.decision_every is None and value < self.decision_hours:
                raise ValueError(
                    "episode_hour_choices cannot be shorter than decision_hours"
                )
        if not self.initial_state_modes:
            raise ValueError("initial_state_modes must not be empty")
        if any(
            mode not in SAMPLED_INITIAL_STATE_MODES for mode in self.initial_state_modes
        ):
            raise ValueError(
                "initial_state_modes contains an unsupported sampled mode; "
                "restore is available only through reset options"
            )
        if len(set(self.initial_state_modes)) != len(self.initial_state_modes):
            raise ValueError("initial_state_modes must be unique")
        for field_name, value in (
            ("liquidate_on_end", self.liquidate_on_end),
            ("finite_horizon_observation", self.finite_horizon_observation),
            ("structured_sequence_observation", self.structured_sequence_observation),
            ("require_full_reward_preroll", self.require_full_reward_preroll),
            ("accept_legacy_actions", self.accept_legacy_actions),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a boolean")
        normalized_windows: list[tuple[str, int]] = []
        for item in self.sequence_windows:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "sequence_windows entries must be timeframe/length pairs"
                )
            timeframe, length = item
            if not isinstance(timeframe, str):
                raise ValueError("sequence timeframe must be a string")
            timeframe_hours(timeframe)
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError("sequence length must be a positive integer")
            normalized_windows.append((timeframe, length))
        if len({timeframe for timeframe, _ in normalized_windows}) != len(
            normalized_windows
        ):
            raise ValueError("sequence timeframes must be unique")
        if normalized_windows and not self.structured_sequence_observation:
            raise ValueError("sequence_windows require structured_sequence_observation")
        object.__setattr__(self, "sequence_windows", tuple(normalized_windows))
        if self.reward_config is not None and self.reward is not None:
            raise ValueError("reward and reward_config cannot both be configured")
        try:
            mode = ActionValidationMode(self.action_validation_mode)
        except ValueError as error:
            raise ValueError("action_validation_mode is not supported") from error
        object.__setattr__(self, "action_validation_mode", mode)
        if not isinstance(self.emergency_risk, EmergencyRiskConfig):
            raise ValueError("emergency_risk must be an EmergencyRiskConfig")

    @property
    def terminal_accounting_mode(self) -> str:
        return "liquidate_at_close" if self.liquidate_on_end else "mark_to_market"

    @property
    def resolved_sequence_windows(self) -> tuple[tuple[str, int], ...]:
        if not self.structured_sequence_observation:
            return ()
        if self.sequence_windows:
            return self.sequence_windows
        return (("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60))

    def resolved_reward_config(self) -> RewardConfig:
        if self.reward_config is not None:
            return self.reward_config
        if self.reward is not None:
            return RewardConfig.from_absolute_growth(self.reward)
        # Legacy fields remain migration knobs around the approved defaults.
        defaults = RewardConfig(scale=self.reward_scale)
        return RewardConfig(
            scale=self.reward_scale,
            incremental_drawdown_weight=(
                defaults.incremental_drawdown_weight
                + 0.10 * self.excess_drawdown_penalty
            ),
            terminal_equity_weight=(
                defaults.terminal_equity_weight + self.downside_penalty
            ),
        )

    def resolve_nominal_episode_bars(self, dataset: MarketDataset) -> int:
        if self.episode_bars is not None:
            return self.episode_bars
        if dataset.regular_cadence:
            return dataset.bars_for_hours(self.episode_hours)
        return max(1, int(round(self.episode_hours / dataset.bar_hours)))

    def resolve_nominal_decision_bars(self, dataset: MarketDataset) -> int:
        if self.decision_every is not None:
            return self.decision_every
        if dataset.regular_cadence:
            return dataset.bars_for_hours(self.decision_hours)
        return max(1, int(round(self.decision_hours / dataset.bar_hours)))


__all__ = [
    "RESET_STATE_MODES",
    "ResidualMarketEnvConfig",
    "SAMPLED_INITIAL_STATE_MODES",
]
