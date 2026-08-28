"""Immutable configuration for Causal Alpha V10 hierarchical waves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest

CAUSAL_ALPHA_V10_CONFIG_SCHEMA: Final = "causal_alpha_v10_config_v1"


@dataclass(frozen=True, slots=True)
class CausalAlphaV10Config:
    fast_wave_weeks: int = 4
    slow_wave_weeks: int = 12
    decisions_per_hour: int = 4
    fast_horizon_hours: int = 4
    slow_horizon_hours: int = 72
    hidden_feature_count: int = 128
    head_seeds: tuple[int, ...] = (0, 1, 2)
    bias_seeds: tuple[int, ...] = (10, 11, 12)
    ridge_strength: float = 1.0
    edge_margin: float = 0.001
    entry_confirmation_count: int = 2
    exit_confirmation_count: int = 2
    slow_neutral_expiry_count: int = 6
    target_magnitude: float = 0.1
    schema_version: str = CAUSAL_ALPHA_V10_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "fast_wave_weeks": 4,
            "slow_wave_weeks": 12,
            "decisions_per_hour": 4,
            "fast_horizon_hours": 4,
            "slow_horizon_hours": 72,
            "hidden_feature_count": 128,
            "head_seeds": (0, 1, 2),
            "bias_seeds": (10, 11, 12),
            "ridge_strength": 1.0,
            "edge_margin": 0.001,
            "entry_confirmation_count": 2,
            "exit_confirmation_count": 2,
            "slow_neutral_expiry_count": 6,
            "target_magnitude": 0.1,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("V10 hierarchical wave constants must remain fixed")
        if self.schema_version != CAUSAL_ALPHA_V10_CONFIG_SCHEMA:
            raise ValueError("unsupported V10 config schema")

    @property
    def fast_horizon_decisions(self) -> int:
        return self.decisions_per_hour * self.fast_horizon_hours

    @property
    def slow_horizon_decisions(self) -> int:
        return self.decisions_per_hour * self.slow_horizon_hours

    @property
    def fast_lookback_decisions(self) -> int:
        return self.fast_wave_weeks * 7 * 24 * self.decisions_per_hour

    @property
    def slow_lookback_decisions(self) -> int:
        return self.slow_wave_weeks * 7 * 24 * self.decisions_per_hour

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["CAUSAL_ALPHA_V10_CONFIG_SCHEMA", "CausalAlphaV10Config"]

