"""Fixed fast-first target contracts for Causal Alpha V6 research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V6_TARGET_CONFIG_SCHEMA: Final = (
    "causal_alpha_v6_target_config_v1"
)
CAUSAL_ALPHA_V6_TARGET_SCHEMA: Final = "causal_alpha_v6_target_v1"
CAUSAL_ALPHA_V6_TARGET_REASONS: Final = frozenset(
    {
        "liquidity_deleverage",
        "risk_projection",
        "execution_band_hold",
        "unactionable_hold",
        "cadence_hold",
        "direction_disagreement_hold",
        "cost_or_uncertainty_hold",
        "confirmation_hold",
        "slow_support_hold",
        "slow_add_suppressed",
        "hold_flat",
        "hold_position",
        "entry",
        "add",
        "reduce",
        "exit",
        "flip",
    }
)
_V6_MAXIMUM_ABSOLUTE_TARGET: Final = 0.25
_V6_EPSILON: Final = 1e-12


def _require_exact_float(value: object, *, expected: float, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"{field} must remain {expected}")


def _require_exact_int(value: object, *, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{field} must remain {expected}")


def _readonly_vector(value: object, *, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
    if array.size == 0:
        raise ValueError(f"{field} must be non-empty")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


class CausalAlphaV6Candidate(str, Enum):
    FAST_ONLY = "fast_only"
    FAST_SLOW_RETENTION = "fast_slow_retention"


class CausalAlphaV6SlowState(str, Enum):
    FLAT = "flat"
    SUPPORTIVE = "supportive"
    MIXED = "mixed"
    OPPOSED = "opposed"


@dataclass(frozen=True, slots=True)
class CausalAlphaV6TargetConfig:
    """The single predeclared V6 fast-first target hypothesis."""

    target_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.25)
    maximum_absolute_target: float = _V6_MAXIMUM_ABSOLUTE_TARGET
    maximum_target_delta: float = 0.125
    fast_rebalance_decisions: int = 4
    slow_context_decisions: int = 16
    uncertainty_multiplier: float = 1.0
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    confirmation_count: int = 2
    strong_reversal_threshold: float = 0.02
    liquidity_lookback_decisions: int = 96
    liquidity_lower_quantile: float = 0.10
    liquidity_safety_multiplier: float = 0.80
    schema_version: str = CAUSAL_ALPHA_V6_TARGET_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected_floats = {
            "maximum_absolute_target": 0.25,
            "maximum_target_delta": 0.125,
            "uncertainty_multiplier": 1.0,
            "execution_cost_multiplier": 1.5,
            "edge_margin": 0.001,
            "strong_reversal_threshold": 0.02,
            "liquidity_lower_quantile": 0.10,
            "liquidity_safety_multiplier": 0.80,
        }
        expected_ints = {
            "fast_rebalance_decisions": 4,
            "slow_context_decisions": 16,
            "confirmation_count": 2,
            "liquidity_lookback_decisions": 96,
        }
        if self.target_magnitudes != (0.0, 0.025, 0.05, 0.10, 0.25):
            raise ValueError("V6 target magnitudes must remain fixed")
        for name, expected in expected_floats.items():
            _require_exact_float(
                getattr(self, name), expected=expected, field=f"V6 {name}"
            )
        for name, expected in expected_ints.items():
            _require_exact_int(
                getattr(self, name), expected=expected, field=f"V6 {name}"
            )
        if self.schema_version != CAUSAL_ALPHA_V6_TARGET_CONFIG_SCHEMA:
            raise ValueError("V6 schema_version must remain fixed")

    def to_payload(self) -> dict[str, object]:
        return {
            "confirmation_count": self.confirmation_count,
            "edge_margin": self.edge_margin,
            "execution_cost_multiplier": self.execution_cost_multiplier,
            "fast_rebalance_decisions": self.fast_rebalance_decisions,
            "liquidity_lookback_decisions": self.liquidity_lookback_decisions,
            "liquidity_lower_quantile": self.liquidity_lower_quantile,
            "liquidity_safety_multiplier": self.liquidity_safety_multiplier,
            "maximum_absolute_target": self.maximum_absolute_target,
            "maximum_target_delta": self.maximum_target_delta,
            "schema_version": self.schema_version,
            "slow_context_decisions": self.slow_context_decisions,
            "strong_reversal_threshold": self.strong_reversal_threshold,
            "target_magnitudes": self.target_magnitudes,
            "uncertainty_multiplier": self.uncertainty_multiplier,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV6TargetPath:
    """Immutable per-symbol target decisions and complete decision evidence."""

    candidate: CausalAlphaV6Candidate
    initial_weight: float
    decision_indices: np.ndarray
    targets: np.ndarray
    fast_proposals: np.ndarray
    expected_returns_4h: np.ndarray
    expected_returns_24h: np.ndarray
    expected_returns_72h: np.ndarray
    direction_scores_4h: np.ndarray
    uncertainties_4h: np.ndarray
    one_way_cost_rates: np.ndarray
    liquidity_weight_caps: np.ndarray
    risk_weight_caps: np.ndarray
    objectives: np.ndarray
    confirmation_counts: np.ndarray
    actionable_mask: np.ndarray
    slow_states: tuple[CausalAlphaV6SlowState, ...]
    reasons: tuple[str, ...]
    reason_counts: tuple[tuple[str, int], ...]
    submitted_change_count: int
    sign_flip_count: int
    liquidity_deleveraging_count: int
    risk_projection_count: int
    forecast_digest: str
    config_digest: str
    schema_version: str = CAUSAL_ALPHA_V6_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.candidate)
        if not math.isfinite(self.initial_weight):
            raise ValueError("V6 target initial weight must be finite")
        require_sha256(self.forecast_digest, field="V6 target forecast digest")
        require_sha256(self.config_digest, field="V6 target config digest")
        if self.schema_version != CAUSAL_ALPHA_V6_TARGET_SCHEMA:
            raise ValueError("unsupported V6 target schema")

        arrays = self._canonical_arrays()
        rows = int(arrays["decision_indices"].size)
        if np.any(arrays["decision_indices"] < 0) or np.any(
            np.diff(arrays["decision_indices"]) <= 0
        ):
            raise ValueError("V6 decision indices must be strictly increasing")
        for name in ("uncertainties_4h", "one_way_cost_rates"):
            if np.any(arrays[name] < 0.0):
                raise ValueError(f"V6 {name} must be non-negative")
        for name in ("liquidity_weight_caps", "risk_weight_caps"):
            if np.any(arrays[name] < 0.0):
                raise ValueError(f"V6 {name} must be non-negative")
        if np.any(np.abs(arrays["targets"]) > _V6_MAXIMUM_ABSOLUTE_TARGET):
            raise ValueError("V6 targets exceeded the absolute bound")
        if np.any(arrays["confirmation_counts"] < 0):
            raise ValueError("V6 confirmation counts must be non-negative")

        states = tuple(CausalAlphaV6SlowState(state) for state in self.slow_states)
        reasons = tuple(self.reasons)
        if len(states) != rows:
            raise ValueError("V6 slow states must align with decisions")
        if len(reasons) != rows or any(
            reason not in CAUSAL_ALPHA_V6_TARGET_REASONS for reason in reasons
        ):
            raise ValueError("V6 target reasons must cover every decision")
        counts = tuple(sorted((reason, reasons.count(reason)) for reason in set(reasons)))
        if tuple(self.reason_counts) != counts:
            raise ValueError("V6 target reason counts do not match reasons")
        self._validate_counts(reasons, arrays["targets"])

        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "slow_states", states)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "reason_counts", counts)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        expected = content_and_arrays_digest(self._digest_metadata(), tuple(arrays.items()))
        if self.digest and self.digest != expected:
            raise ValueError("V6 target path digest mismatch")
        object.__setattr__(self, "digest", expected)

    def _canonical_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "decision_indices": _readonly_vector(
                self.decision_indices, dtype=np.int64, field="V6 decision indices"
            )
        }
        rows = int(arrays["decision_indices"].size)
        float_names = (
            "targets",
            "fast_proposals",
            "expected_returns_4h",
            "expected_returns_24h",
            "expected_returns_72h",
            "direction_scores_4h",
            "uncertainties_4h",
            "one_way_cost_rates",
            "liquidity_weight_caps",
            "risk_weight_caps",
            "objectives",
        )
        for name in float_names:
            arrays[name] = _readonly_vector(
                getattr(self, name), dtype=np.float64, field=f"V6 {name}"
            )
        arrays["confirmation_counts"] = _readonly_vector(
            self.confirmation_counts, dtype=np.int64, field="V6 confirmation counts"
        )
        arrays["actionable_mask"] = _readonly_vector(
            self.actionable_mask, dtype=np.bool_, field="V6 actionable mask"
        )
        if any(array.shape != (rows,) for array in arrays.values()):
            raise ValueError("V6 target arrays must align")
        return arrays

    def _validate_counts(self, reasons: tuple[str, ...], targets: np.ndarray) -> None:
        count_names = (
            "submitted_change_count",
            "sign_flip_count",
            "liquidity_deleveraging_count",
            "risk_projection_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V6 target {name} is invalid")
        previous = np.concatenate(([self.initial_weight], targets[:-1]))
        submitted = int(np.count_nonzero(np.abs(targets - previous) > _V6_EPSILON))
        flips = int(np.count_nonzero(previous * targets < 0.0))
        expected = {
            "submitted_change_count": submitted,
            "sign_flip_count": flips,
            "liquidity_deleveraging_count": reasons.count("liquidity_deleverage"),
            "risk_projection_count": reasons.count("risk_projection"),
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("V6 target transition counts do not match evidence")

    def _digest_metadata(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "config_digest": self.config_digest,
            "forecast_digest": self.forecast_digest,
            "initial_weight": self.initial_weight,
            "liquidity_deleveraging_count": self.liquidity_deleveraging_count,
            "reason_counts": self.reason_counts,
            "reasons": self.reasons,
            "risk_projection_count": self.risk_projection_count,
            "schema_version": self.schema_version,
            "sign_flip_count": self.sign_flip_count,
            "slow_states": tuple(state.value for state in self.slow_states),
            "submitted_change_count": self.submitted_change_count,
        }


__all__ = [
    "CAUSAL_ALPHA_V6_TARGET_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V6_TARGET_REASONS",
    "CAUSAL_ALPHA_V6_TARGET_SCHEMA",
    "CausalAlphaV6Candidate",
    "CausalAlphaV6SlowState",
    "CausalAlphaV6TargetConfig",
    "CausalAlphaV6TargetPath",
]
