"""Immutable configuration for Causal Alpha V9 nonlinear waves."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetPath,
)

CAUSAL_ALPHA_V9_CONFIG_SCHEMA: Final = "causal_alpha_v9_config_v1"
CAUSAL_ALPHA_V9_TARGET_SCHEMA: Final = "causal_alpha_v9_target_v1"


class CausalAlphaV9Candidate(str, Enum):
    V7_CONTROL = "v7_control"
    V8_ROBUST_CONTROL = "v8_robust_control"
    NONLINEAR_WAVE = "nonlinear_wave"


@dataclass(frozen=True, slots=True)
class CausalAlphaV9Config:
    wave_weeks: int = 4
    decisions_per_hour: int = 4
    prediction_horizon_hours: int = 4
    hidden_feature_count: int = 128
    head_seeds: tuple[int, ...] = (0, 1, 2)
    bias_seeds: tuple[int, ...] = (10, 11, 12)
    ridge_strength: float = 1.0
    edge_margin: float = 0.001
    confirmation_count: int = 2
    target_magnitude: float = 0.025
    schema_version: str = CAUSAL_ALPHA_V9_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "wave_weeks": 4,
            "decisions_per_hour": 4,
            "prediction_horizon_hours": 4,
            "hidden_feature_count": 128,
            "head_seeds": (0, 1, 2),
            "bias_seeds": (10, 11, 12),
            "ridge_strength": 1.0,
            "edge_margin": 0.001,
            "confirmation_count": 2,
            "target_magnitude": 0.025,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("V9 nonlinear wave constants must remain fixed")
        if self.schema_version != CAUSAL_ALPHA_V9_CONFIG_SCHEMA:
            raise ValueError("unsupported V9 config schema")

    @property
    def horizon_decisions(self) -> int:
        return self.decisions_per_hour * self.prediction_horizon_hours

    @property
    def lookback_decisions(self) -> int:
        return self.wave_weeks * 7 * 24 * self.decisions_per_hour

    def to_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV9TargetPath:
    candidate: CausalAlphaV9Candidate
    v6_target_path: CausalAlphaV6TargetPath
    source_forecast_digest: str
    wave_fit_digest: str
    v9_config_digest: str
    schema_version: str = CAUSAL_ALPHA_V9_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV9Candidate(self.candidate)
        if self.v6_target_path.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V9 target must use V6 fast-only replay semantics")
        for name in (
            "source_forecast_digest",
            "wave_fit_digest",
            "v9_config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V9 target {name}")
        if self.schema_version != CAUSAL_ALPHA_V9_TARGET_SCHEMA:
            raise ValueError("unsupported V9 target schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V9 target digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate": self.candidate.value,
            "schema_version": self.schema_version,
            "source_forecast_digest": self.source_forecast_digest,
            "v6_target_path_digest": self.v6_target_path.digest,
            "v9_config_digest": self.v9_config_digest,
            "wave_fit_digest": self.wave_fit_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CAUSAL_ALPHA_V9_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V9_TARGET_SCHEMA",
    "CausalAlphaV9Candidate",
    "CausalAlphaV9Config",
    "CausalAlphaV9TargetPath",
]
