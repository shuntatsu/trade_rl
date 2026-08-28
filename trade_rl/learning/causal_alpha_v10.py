"""Immutable configuration for Causal Alpha V10 hierarchical waves."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6TargetPath

CAUSAL_ALPHA_V10_CONFIG_SCHEMA: Final = "causal_alpha_v10_config_v1"
CAUSAL_ALPHA_V10_TARGET_SCHEMA: Final = "causal_alpha_v10_target_v1"


class CausalAlphaV10Candidate(str, Enum):
    V8_ROBUST_CONTROL = "v8_robust_control"
    V9_NONLINEAR_CONTROL = "v9_nonlinear_control"
    HIERARCHICAL_WAVE = "hierarchical_wave"


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


@dataclass(frozen=True, slots=True)
class CausalAlphaV10TargetPath:
    candidate: CausalAlphaV10Candidate
    v6_target_path: CausalAlphaV6TargetPath
    source_forecast_digest: str
    fast_fit_digest: str
    slow_fit_digest: str
    v10_config_digest: str
    schema_version: str = CAUSAL_ALPHA_V10_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV10Candidate(self.candidate)
        if not isinstance(self.v6_target_path, CausalAlphaV6TargetPath):
            raise TypeError("V10 target must contain a V6 target path")
        for name in (
            "source_forecast_digest",
            "fast_fit_digest",
            "slow_fit_digest",
            "v10_config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V10 target {name}")
        if self.schema_version != CAUSAL_ALPHA_V10_TARGET_SCHEMA:
            raise ValueError("unsupported V10 target schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 target digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate": self.candidate.value,
            "fast_fit_digest": self.fast_fit_digest,
            "schema_version": self.schema_version,
            "slow_fit_digest": self.slow_fit_digest,
            "source_forecast_digest": self.source_forecast_digest,
            "v10_config_digest": self.v10_config_digest,
            "v6_target_path_digest": self.v6_target_path.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CAUSAL_ALPHA_V10_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V10_TARGET_SCHEMA",
    "CausalAlphaV10Candidate",
    "CausalAlphaV10Config",
    "CausalAlphaV10TargetPath",
]
