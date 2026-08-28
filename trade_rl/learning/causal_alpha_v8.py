"""Contracts for the Causal Alpha V8 exposure state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)

CAUSAL_ALPHA_V8_TARGET_CONFIG_SCHEMA: Final = "causal_alpha_v8_target_config_v1"
CAUSAL_ALPHA_V8_TARGET_SCHEMA: Final = "causal_alpha_v8_target_v1"


class CausalAlphaV8Candidate(str, Enum):
    V7_CONTROL = "v7_control"
    ROBUST_CONTRARIAN = "robust_contrarian"
    ROBUST_CALIBRATED = "robust_calibrated"


@dataclass(frozen=True, slots=True)
class CausalAlphaV8TargetConfig:
    """Bind the robust exposure objective to the unchanged V6 constants."""

    base: CausalAlphaV6TargetConfig = field(default_factory=CausalAlphaV6TargetConfig)
    direct_flip_allowed: bool = False
    schema_version: str = CAUSAL_ALPHA_V8_TARGET_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.base, CausalAlphaV6TargetConfig):
            raise TypeError("V8 target base config is invalid")
        if self.direct_flip_allowed:
            raise ValueError("V8 direct flips must remain disabled")
        if self.schema_version != CAUSAL_ALPHA_V8_TARGET_CONFIG_SCHEMA:
            raise ValueError("unsupported V8 target config schema")

    def to_payload(self) -> dict[str, object]:
        return {
            "base_config": self.base.to_payload(),
            "base_config_digest": self.base.digest,
            "direct_flip_allowed": self.direct_flip_allowed,
            "objective": "robust_position_utility",
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV8TargetPath:
    candidate: CausalAlphaV8Candidate
    v6_target_path: CausalAlphaV6TargetPath
    source_forecast_digest: str
    calibration_fit_digest: str
    v8_config_digest: str
    schema_version: str = CAUSAL_ALPHA_V8_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        from trade_rl.domain.common import require_sha256

        candidate = CausalAlphaV8Candidate(self.candidate)
        if not isinstance(self.v6_target_path, CausalAlphaV6TargetPath):
            raise TypeError("V8 target requires a V6-compatible target path")
        if self.v6_target_path.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V8 target path must use V6 fast-only replay semantics")
        for name in (
            "source_forecast_digest",
            "calibration_fit_digest",
            "v8_config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V8 target {name}")
        if self.schema_version != CAUSAL_ALPHA_V8_TARGET_SCHEMA:
            raise ValueError("unsupported V8 target schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V8 target digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_fit_digest": self.calibration_fit_digest,
            "candidate": self.candidate.value,
            "schema_version": self.schema_version,
            "source_forecast_digest": self.source_forecast_digest,
            "v6_target_path_digest": self.v6_target_path.digest,
            "v8_config_digest": self.v8_config_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CAUSAL_ALPHA_V8_TARGET_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V8_TARGET_SCHEMA",
    "CausalAlphaV8Candidate",
    "CausalAlphaV8TargetConfig",
    "CausalAlphaV8TargetPath",
]
