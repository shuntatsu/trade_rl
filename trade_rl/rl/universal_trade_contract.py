"""Frozen Observation / Action / Reward semantics for Universal Trade RL U1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import FeatureKind, FeatureSpec

UNIVERSAL_TRADE_OBSERVATION_SCHEMA: Final = "universal_trade_observation_v1"
UNIVERSAL_TRADE_ACTION_SCHEMA: Final = "normalized_target_exposure_v1"
UNIVERSAL_TRADE_REWARD_SCHEMA: Final = "universal_net_log_growth_reward_v1"
UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA: Final = "universal_trade_policy_state_v1"
UNIVERSAL_TRADE_SEQUENCE_WINDOWS: Final = (
    ("15m", 96),
    ("1h", 168),
    ("4h", 120),
    ("1d", 60),
)
UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS: Final = frozenset(
    {
        FeatureKind.LOG_RETURN,
        FeatureKind.BODY_RETURN,
        FeatureKind.HIGH_LOW_RANGE,
        FeatureKind.GAP_RETURN,
        FeatureKind.REALIZED_VOLATILITY,
        FeatureKind.DOWNSIDE_VOLATILITY,
        FeatureKind.UPSIDE_VOLATILITY,
        FeatureKind.VOLATILITY_OF_VOLATILITY,
        FeatureKind.ATR_PCT,
        FeatureKind.ATR_CHANGE,
        FeatureKind.EMA_DISTANCE,
        FeatureKind.EMA_SLOPE,
        FeatureKind.LINEAR_REGRESSION_SLOPE,
        FeatureKind.TREND_R2,
        FeatureKind.VOLUME_ZSCORE,
        FeatureKind.VOLUME_LOG_CHANGE,
        FeatureKind.RELATIVE_VOLUME,
        FeatureKind.FUNDING_BPS,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.FUNDING_ZSCORE,
    }
)

_BASE_TIMEFRAME: Final = "15m"
_FIXED_REWARD_SCALE: Final = 100.0
_SEQUENCE_TIMEFRAMES: Final = frozenset(
    timeframe for timeframe, _length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS
)


@dataclass(frozen=True, slots=True)
class UniversalTradePolicyContract:
    """Immutable symbol-independent U1 policy surface and runtime semantics."""

    feature_specs: tuple[FeatureSpec, ...]
    policy_weight_scale: float = 1.0
    reward_scale: float = _FIXED_REWARD_SCALE

    def __post_init__(self) -> None:
        if not isinstance(self.feature_specs, tuple) or not self.feature_specs:
            raise ValueError("U1 feature_specs must be a non-empty tuple")
        if any(not isinstance(spec, FeatureSpec) for spec in self.feature_specs):
            raise TypeError("U1 feature_specs must contain only FeatureSpec values")
        names = tuple(spec.name for spec in self.feature_specs)
        if len(set(names)) != len(names):
            raise ValueError("U1 feature names must be unique")

        for spec in self.feature_specs:
            if spec.kind not in UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS:
                raise ValueError(f"U1 feature kind is not allowed: {spec.kind.value}")
            timeframe = spec.resolved_timeframe(_BASE_TIMEFRAME)
            if timeframe not in _SEQUENCE_TIMEFRAMES:
                raise ValueError(f"U1 feature timeframe is not allowed: {timeframe}")
            expected_prefix = f"{timeframe}__"
            if not spec.name.startswith(expected_prefix):
                raise ValueError(
                    "U1 feature prefix must match its resolved timeframe: "
                    f"name={spec.name!r}, timeframe={timeframe!r}"
                )

        if (
            isinstance(self.policy_weight_scale, bool)
            or not math.isfinite(self.policy_weight_scale)
            or not 0.0 < self.policy_weight_scale <= 1.0
        ):
            raise ValueError("policy_weight_scale must be finite and within (0, 1]")
        if (
            isinstance(self.reward_scale, bool)
            or not math.isfinite(self.reward_scale)
            or self.reward_scale != _FIXED_REWARD_SCALE
        ):
            raise ValueError("reward_scale must be exactly 100.0 for U1 V1")

    def digest_payload(self) -> dict[str, object]:
        """Return the complete fixed U1 semantic contract used for identity."""

        return {
            "accept_legacy_actions": False,
            "action_mode": "target_weight",
            "action_schema": UNIVERSAL_TRADE_ACTION_SCHEMA,
            "action_validation_mode": "strict",
            "decision_every": None,
            "decision_hours": 0.25,
            "episode_bars": None,
            "episode_boundary_mode": "external_truncation",
            "episode_hour_choices": (),
            "episode_hours": 720.0,
            "feature_specs": tuple(
                spec.canonical_payload() for spec in self.feature_specs
            ),
            "finite_horizon_observation": False,
            "initial_state_modes": ("cash",),
            "liquidate_on_end": False,
            "observation_schema": UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
            "policy_weight_scale": self.policy_weight_scale,
            "reward_scale": self.reward_scale,
            "reward_schema": UNIVERSAL_TRADE_REWARD_SCHEMA,
            "sequence_windows": UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
            "signal_delay_decisions": 1,
            "state_layout_schema": UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA,
            "target_weight_count": 1,
        }

    @property
    def digest(self) -> str:
        """Canonical SHA-256 identity of the ordered U1 policy contract."""

        return content_digest(self.digest_payload())


__all__ = [
    "UNIVERSAL_TRADE_ACTION_SCHEMA",
    "UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS",
    "UNIVERSAL_TRADE_OBSERVATION_SCHEMA",
    "UNIVERSAL_TRADE_REWARD_SCHEMA",
    "UNIVERSAL_TRADE_SEQUENCE_WINDOWS",
    "UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA",
    "UniversalTradePolicyContract",
]
