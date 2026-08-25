"""Fixed contracts for Causal Alpha V7 causal calibration research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V7_CALIBRATION_CONFIG_SCHEMA: Final = (
    "causal_alpha_v7_calibration_config_v1"
)
CAUSAL_ALPHA_V7_CALIBRATION_RANGE_SCHEMA: Final = (
    "causal_alpha_v7_calibration_range_v1"
)
CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES: Final = (
    "fast_return_raw",
    "fast_direction_raw",
    "log_fast_uncertainty",
    "realized_volatility",
    "liquidity",
    "basis_positioning_stress",
    "slow_return_24h",
    "slow_return_72h",
    "slow_direction_24h",
    "slow_direction_72h",
)


def _require_exact_float(value: object, *, expected: float, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"V7 {field} must remain {expected}")


def _require_exact_int(value: object, *, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"V7 {field} must remain {expected}")


class CausalAlphaV7Candidate(str, Enum):
    V6_CONTROL = "v6_control"
    SYMMETRIC_CONTRARIAN = "symmetric_contrarian"
    CAUSAL_CALIBRATED = "causal_calibrated"


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CalibrationConfig:
    """The single predeclared V7 calibration hypothesis."""

    calibration_fraction: float = 0.50
    forward_block_count: int = 4
    ridge_strength: float = 1.0
    minimum_pooled_support: int = 256
    minimum_symbol_support: int = 16
    working_memory_rows: int = 4_096
    schema_version: str = CAUSAL_ALPHA_V7_CALIBRATION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _require_exact_float(
            self.calibration_fraction,
            expected=0.50,
            field="calibration fraction",
        )
        _require_exact_int(
            self.forward_block_count,
            expected=4,
            field="forward block count",
        )
        _require_exact_float(
            self.ridge_strength,
            expected=1.0,
            field="ridge strength",
        )
        _require_exact_int(
            self.minimum_pooled_support,
            expected=256,
            field="minimum pooled support",
        )
        _require_exact_int(
            self.minimum_symbol_support,
            expected=16,
            field="minimum symbol support",
        )
        _require_exact_int(
            self.working_memory_rows,
            expected=4_096,
            field="working memory rows",
        )
        if self.schema_version != CAUSAL_ALPHA_V7_CALIBRATION_CONFIG_SCHEMA:
            raise ValueError("V7 calibration config schema must remain fixed")

    def to_payload(self) -> dict[str, object]:
        return {
            "calibration_fraction": self.calibration_fraction,
            "forward_block_count": self.forward_block_count,
            "minimum_pooled_support": self.minimum_pooled_support,
            "minimum_symbol_support": self.minimum_symbol_support,
            "ridge_strength": self.ridge_strength,
            "schema_version": self.schema_version,
            "working_memory_rows": self.working_memory_rows,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CalibrationRange:
    """Bind one V5 chronological split to the V7 base fit and feature schema."""

    base_fit_cutoff: int
    calibration_start: int
    train_stop: int
    block_boundaries: tuple[int, ...]
    split_digest: str
    feature_names: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V7_CALIBRATION_RANGE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in ("base_fit_cutoff", "calibration_start", "train_stop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"V7 {name} is invalid")
        if self.base_fit_cutoff != self.calibration_start:
            raise ValueError("V7 base fit cutoff must equal calibration start")
        boundaries = tuple(self.block_boundaries)
        if (
            len(boundaries) != 5
            or boundaries[0] != self.calibration_start
            or boundaries[-1] != self.train_stop
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("V7 block boundaries are invalid")
        require_sha256(self.split_digest, field="V7 calibration split digest")
        feature_names = tuple(self.feature_names)
        if any("symbol" in name.lower() for name in feature_names):
            raise ValueError("V7 calibration cannot contain symbol identity features")
        if feature_names != CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES:
            raise ValueError("V7 calibration feature schema drifted")
        if self.schema_version != CAUSAL_ALPHA_V7_CALIBRATION_RANGE_SCHEMA:
            raise ValueError("unsupported V7 calibration range schema")
        object.__setattr__(self, "block_boundaries", boundaries)
        object.__setattr__(self, "feature_names", feature_names)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 calibration range digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "base_fit_cutoff": self.base_fit_cutoff,
            "block_boundaries": self.block_boundaries,
            "calibration_start": self.calibration_start,
            "feature_names": self.feature_names,
            "schema_version": self.schema_version,
            "split_digest": self.split_digest,
            "train_stop": self.train_stop,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES",
    "CausalAlphaV7CalibrationConfig",
    "CausalAlphaV7CalibrationRange",
    "CausalAlphaV7Candidate",
]
