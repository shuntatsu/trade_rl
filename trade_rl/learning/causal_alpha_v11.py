"""Immutable contracts for Causal Alpha V11 policy research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6TargetPath

CAUSAL_ALPHA_V11_CONFIG_SCHEMA: Final = "causal_alpha_v11_config_v1"
CAUSAL_ALPHA_V11_TARGET_SCHEMA: Final = "causal_alpha_v11_target_v1"
CAUSAL_ALPHA_V11_SIZING_FEASIBILITY_SCHEMA: Final = (
    "causal_alpha_v11_sizing_feasibility_v1"
)
_EPSILON: Final = 1e-12


class CausalAlphaV11Candidate(str, Enum):
    """The three identities allowed inside one independent study-arm gate."""

    V8_CASH_SANITY = "v8_cash_sanity"
    V9_CONTROL = "v9_control"
    TREATMENT = "treatment"


class CausalAlphaV11StudyArm(str, Enum):
    """Pre-registered V11 treatment hypotheses."""

    NEUTRAL_EXPIRY_2 = "neutral_expiry_2"
    AFTER_COST_ENTRY = "after_cost_entry"
    SIGN_CALIBRATED_ENTRY = "sign_calibrated_entry"
    CALIBRATED_EDGE_SIZING = "calibrated_edge_sizing"


@dataclass(frozen=True, slots=True)
class CausalAlphaV11Config:
    """Fixed V11 constants shared by every independently executed study arm."""

    neutral_expiry_count: int = 2
    calibration_hours: int = 168
    calibration_ridge_strength: float = 1.0
    sizing_epsilon: float = _EPSILON
    schema_version: str = CAUSAL_ALPHA_V11_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "neutral_expiry_count": 2,
            "calibration_hours": 168,
            "calibration_ridge_strength": 1.0,
            "sizing_epsilon": _EPSILON,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("V11 constants must remain fixed")
        if self.schema_version != CAUSAL_ALPHA_V11_CONFIG_SCHEMA:
            raise ValueError("unsupported V11 config schema")

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV11TargetPath:
    """V11 identity wrapper around one complete V6-compatible target path."""

    candidate: CausalAlphaV11Candidate
    study_arm: CausalAlphaV11StudyArm | None
    v6_target_path: CausalAlphaV6TargetPath
    source_forecast_digest: str
    wave_fit_digest: str
    v9_config_digest: str
    v11_config_digest: str
    calibration_digest: str | None = None
    schema_version: str = CAUSAL_ALPHA_V11_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV11Candidate(self.candidate)
        study_arm = (
            None if self.study_arm is None else CausalAlphaV11StudyArm(self.study_arm)
        )
        if candidate is CausalAlphaV11Candidate.V8_CASH_SANITY:
            raise ValueError("cash sanity does not own a V11 target path")
        if candidate is CausalAlphaV11Candidate.V9_CONTROL and study_arm is not None:
            raise ValueError("V9 control must not carry a study arm")
        if candidate is CausalAlphaV11Candidate.TREATMENT and study_arm is None:
            raise ValueError("V11 treatment requires a study arm")
        for field_name in (
            "source_forecast_digest",
            "wave_fit_digest",
            "v9_config_digest",
            "v11_config_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V11 {field_name}")
        if self.calibration_digest is not None:
            require_sha256(self.calibration_digest, field="V11 calibration_digest")
        if self.schema_version != CAUSAL_ALPHA_V11_TARGET_SCHEMA:
            raise ValueError("unsupported V11 target schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "study_arm", study_arm)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V11 target digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_digest": self.calibration_digest,
            "candidate": self.candidate.value,
            "schema_version": self.schema_version,
            "source_forecast_digest": self.source_forecast_digest,
            "study_arm": None if self.study_arm is None else self.study_arm.value,
            "v11_config_digest": self.v11_config_digest,
            "v6_target_path_digest": self.v6_target_path.digest,
            "v9_config_digest": self.v9_config_digest,
            "wave_fit_digest": self.wave_fit_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV11SizingFeasibility:
    """Artifact-bound preflight for whether generated sizes can enter a trade."""

    entry_threshold: float
    no_trade_band: float
    generated_nonzero_count: int
    executable_nonzero_count: int
    maximum_absolute_target: float
    target_digest: str
    rejection_reasons: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V11_SIZING_FEASIBILITY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.target_digest, field="V11 sizing target_digest")
        if self.schema_version != CAUSAL_ALPHA_V11_SIZING_FEASIBILITY_SCHEMA:
            raise ValueError("unsupported V11 sizing feasibility schema")
        if (
            self.generated_nonzero_count < 0
            or self.executable_nonzero_count < 0
            or self.executable_nonzero_count > self.generated_nonzero_count
        ):
            raise ValueError("invalid V11 sizing feasibility counts")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V11 sizing feasibility digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def executable(self) -> bool:
        return self.executable_nonzero_count > 0 and not self.rejection_reasons

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "entry_threshold": self.entry_threshold,
            "executable": self.executable,
            "executable_nonzero_count": self.executable_nonzero_count,
            "generated_nonzero_count": self.generated_nonzero_count,
            "maximum_absolute_target": self.maximum_absolute_target,
            "no_trade_band": self.no_trade_band,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "target_digest": self.target_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_v11_sizing_feasibility(
    *,
    targets: object,
    entry_threshold: float,
    no_trade_band: float,
) -> CausalAlphaV11SizingFeasibility:
    """Evaluate generated targets against the frozen execution thresholds."""

    if (
        not math.isfinite(entry_threshold)
        or not math.isfinite(no_trade_band)
        or entry_threshold <= 0.0
        or no_trade_band <= 0.0
        or no_trade_band > entry_threshold
    ):
        raise ValueError("V11 execution thresholds are invalid")
    array = np.asarray(targets, dtype=np.float64).reshape(-1).copy(order="C")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("V11 sizing targets must be non-empty and finite")
    absolute = np.abs(array)
    generated = absolute > _EPSILON
    executable = generated & (absolute >= entry_threshold) & (absolute > no_trade_band)
    maximum = float(np.max(absolute))
    reasons = () if np.any(executable) else ("entry_threshold",)
    target_digest = content_and_arrays_digest(
        {"schema_version": CAUSAL_ALPHA_V11_SIZING_FEASIBILITY_SCHEMA},
        (("targets", array),),
    )
    return CausalAlphaV11SizingFeasibility(
        entry_threshold=float(entry_threshold),
        no_trade_band=float(no_trade_band),
        generated_nonzero_count=int(np.count_nonzero(generated)),
        executable_nonzero_count=int(np.count_nonzero(executable)),
        maximum_absolute_target=maximum,
        target_digest=target_digest,
        rejection_reasons=reasons,
    )


__all__ = [
    "CAUSAL_ALPHA_V11_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V11_SIZING_FEASIBILITY_SCHEMA",
    "CAUSAL_ALPHA_V11_TARGET_SCHEMA",
    "CausalAlphaV11Candidate",
    "CausalAlphaV11Config",
    "CausalAlphaV11SizingFeasibility",
    "CausalAlphaV11StudyArm",
    "CausalAlphaV11TargetPath",
    "evaluate_v11_sizing_feasibility",
]
