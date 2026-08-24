"""Train-only calibration and selective slow-lane contracts for Causal Alpha V5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_teacher import CausalAlphaRidgeModel
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4Forecast,
    CausalAlphaV4TargetConfig,
    _v4_choose_best,
    _v4_consensus_allows,
    _v4_fast_candidates,
    _v4_is_risk_reduction,
    _v4_slow_candidates,
    _v4_staged_objective,
)

CAUSAL_ALPHA_V5_CALIBRATION_CONFIG_SCHEMA: Final = (
    "causal_alpha_v5_calibration_config_v1"
)
CAUSAL_ALPHA_V5_CALIBRATION_FIT_SCHEMA: Final = "causal_alpha_v5_calibration_fit_v1"
CAUSAL_ALPHA_V5_SELECTIVE_FORECAST_SCHEMA: Final = (
    "causal_alpha_v5_selective_forecast_v1"
)
CAUSAL_ALPHA_V5_TARGET_SCHEMA: Final = "causal_alpha_v5_target_v1"
CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES: Final = (
    "slow_return_raw",
    "slow_direction_raw",
    "log_slow_uncertainty",
    *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
)
_V5_EPSILON: Final = 1e-12


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


def _readonly_matrix(value: object, *, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{field} must be a non-empty matrix")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


def _digest_tuple(value: object, *, count: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} digests")
    resolved = tuple(str(item) for item in value)
    for index, digest in enumerate(resolved):
        require_sha256(digest, field=f"{field}[{index}]")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV5CalibrationConfig:
    """The single predeclared train-only calibration and abstention hypothesis."""

    calibration_fraction: float = 0.20
    forward_block_count: int = 4
    ridge_strength: float = 1.0
    minimum_pooled_support: int = 256
    minimum_symbol_support: int = 16
    minimum_selective_confidence: float = 1.0
    minimum_active_coverage: float = 0.25
    minimum_scope_active_fraction: float = 0.20
    minimum_scope_active_count: int = 3
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    schema_version: str = CAUSAL_ALPHA_V5_CALIBRATION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _require_exact_float(
            self.calibration_fraction,
            expected=0.20,
            field="V5 calibration fraction",
        )
        _require_exact_int(
            self.forward_block_count,
            expected=4,
            field="V5 forward block count",
        )
        _require_exact_float(
            self.ridge_strength,
            expected=1.0,
            field="V5 calibration ridge strength",
        )
        _require_exact_int(
            self.minimum_pooled_support,
            expected=256,
            field="V5 minimum pooled support",
        )
        _require_exact_int(
            self.minimum_symbol_support,
            expected=16,
            field="V5 minimum symbol support",
        )
        _require_exact_float(
            self.minimum_selective_confidence,
            expected=1.0,
            field="V5 minimum selective confidence",
        )
        _require_exact_float(
            self.minimum_active_coverage,
            expected=0.25,
            field="V5 minimum active coverage",
        )
        _require_exact_float(
            self.minimum_scope_active_fraction,
            expected=0.20,
            field="V5 minimum scope active fraction",
        )
        _require_exact_int(
            self.minimum_scope_active_count,
            expected=3,
            field="V5 minimum scope active count",
        )
        _require_exact_float(
            self.execution_cost_multiplier,
            expected=1.5,
            field="V5 execution cost multiplier",
        )
        _require_exact_float(
            self.edge_margin,
            expected=0.001,
            field="V5 edge margin",
        )
        if self.schema_version != CAUSAL_ALPHA_V5_CALIBRATION_CONFIG_SCHEMA:
            raise ValueError("unsupported V5 calibration config schema")

    def to_payload(self) -> dict[str, object]:
        return {
            "calibration_fraction": self.calibration_fraction,
            "edge_margin": self.edge_margin,
            "execution_cost_multiplier": self.execution_cost_multiplier,
            "forward_block_count": self.forward_block_count,
            "minimum_active_coverage": self.minimum_active_coverage,
            "minimum_pooled_support": self.minimum_pooled_support,
            "minimum_scope_active_count": self.minimum_scope_active_count,
            "minimum_scope_active_fraction": self.minimum_scope_active_fraction,
            "minimum_selective_confidence": self.minimum_selective_confidence,
            "minimum_symbol_support": self.minimum_symbol_support,
            "ridge_strength": self.ridge_strength,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV5CalibrationFit:
    """Immutable final and forward calibration evidence."""

    v4_fit_digest: str
    v4_fit_config_digest: str
    v4_sample_scope_digest: str
    calibration_start: int
    train_stop: int
    model: CausalAlphaRidgeModel
    forward_model_digests: tuple[str, ...]
    forward_residual_digests: tuple[str, ...]
    final_weight_digest: str
    forward_weight_digests: tuple[str, ...]
    per_symbol_support: tuple[tuple[str, int], ...]
    calibration_block_support: tuple[int, ...]
    forward_block_symbol_counts: tuple[int, ...]
    calibration_residual_rmse: float
    direction_score_rmse: float
    config: CausalAlphaV5CalibrationConfig
    schema_version: str = CAUSAL_ALPHA_V5_CALIBRATION_FIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "v4_fit_digest",
            "v4_fit_config_digest",
            "v4_sample_scope_digest",
            "final_weight_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V5 {field_name}")
        if (
            isinstance(self.calibration_start, bool)
            or not isinstance(self.calibration_start, int)
            or isinstance(self.train_stop, bool)
            or not isinstance(self.train_stop, int)
            or self.calibration_start <= 0
            or self.train_stop <= self.calibration_start
        ):
            raise ValueError("V5 calibration boundaries are invalid")
        if not isinstance(self.config, CausalAlphaV5CalibrationConfig):
            raise TypeError("V5 calibration fit config is invalid")
        if not isinstance(self.model, CausalAlphaRidgeModel):
            raise TypeError("V5 calibration fit model is invalid")
        if self.model.feature_names != CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES:
            raise ValueError("V5 calibration feature schema drifted")
        if self.model.knowledge_cutoff != self.train_stop:
            raise ValueError("V5 calibration model cutoff drifted")
        if self.model.config.ridge_strength != self.config.ridge_strength:
            raise ValueError("V5 calibration model ridge strength drifted")

        forward_models = _digest_tuple(
            self.forward_model_digests,
            count=self.config.forward_block_count - 1,
            field="V5 forward model digests",
        )
        forward_residuals = _digest_tuple(
            self.forward_residual_digests,
            count=self.config.forward_block_count - 1,
            field="V5 forward residual digests",
        )
        forward_weights = _digest_tuple(
            self.forward_weight_digests,
            count=self.config.forward_block_count - 1,
            field="V5 forward weight digests",
        )

        support = tuple(self.per_symbol_support)
        if (
            not support
            or tuple(sorted(support)) != support
            or len({symbol for symbol, _ in support}) != len(support)
        ):
            raise ValueError("V5 calibration symbol support identity is invalid")
        for symbol, count in support:
            if (
                not isinstance(symbol, str)
                or not symbol
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < self.config.minimum_symbol_support
            ):
                raise ValueError("V5 calibration symbol support is insufficient")
        pooled_support = sum(count for _, count in support)
        if pooled_support < self.config.minimum_pooled_support:
            raise ValueError("V5 calibration pooled support is insufficient")
        if self.model.sample_count != pooled_support:
            raise ValueError(
                "V5 calibration model sample_count must match pooled support"
            )

        block_support = tuple(self.calibration_block_support)
        if (
            len(block_support) != self.config.forward_block_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in block_support
            )
            or sum(block_support) != pooled_support
        ):
            raise ValueError("V5 calibration block support is invalid")
        block_symbols = tuple(self.forward_block_symbol_counts)
        if len(block_symbols) != self.config.forward_block_count - 1 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 2 <= value <= len(support)
            for value in block_symbols
        ):
            raise ValueError("V5 forward block symbol support is invalid")
        for field_name in ("calibration_residual_rmse", "direction_score_rmse"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V5 {field_name} must be non-negative")
        if self.schema_version != CAUSAL_ALPHA_V5_CALIBRATION_FIT_SCHEMA:
            raise ValueError("unsupported V5 calibration fit schema")

        object.__setattr__(self, "forward_model_digests", forward_models)
        object.__setattr__(self, "forward_residual_digests", forward_residuals)
        object.__setattr__(self, "forward_weight_digests", forward_weights)
        object.__setattr__(self, "per_symbol_support", support)
        object.__setattr__(self, "calibration_block_support", block_support)
        object.__setattr__(self, "forward_block_symbol_counts", block_symbols)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 calibration fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def pooled_support(self) -> int:
        return sum(count for _, count in self.per_symbol_support)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_block_support": self.calibration_block_support,
            "calibration_residual_rmse": self.calibration_residual_rmse,
            "calibration_start": self.calibration_start,
            "config_digest": self.config.digest,
            "direction_score_rmse": self.direction_score_rmse,
            "final_model_digest": self.model.digest,
            "final_weight_digest": self.final_weight_digest,
            "forward_block_symbol_counts": self.forward_block_symbol_counts,
            "forward_model_digests": self.forward_model_digests,
            "forward_residual_digests": self.forward_residual_digests,
            "forward_weight_digests": self.forward_weight_digests,
            "per_symbol_support": self.per_symbol_support,
            "pooled_support": self.pooled_support,
            "schema_version": self.schema_version,
            "train_stop": self.train_stop,
            "v4_fit_config_digest": self.v4_fit_config_digest,
            "v4_fit_digest": self.v4_fit_digest,
            "v4_sample_scope_digest": self.v4_sample_scope_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


class V5SelectiveState(str, Enum):
    ACTIVE = "active"
    DIRECTION_DISAGREEMENT = "direction_disagreement"
    CONFIDENCE_ABSTAIN = "confidence_abstain"
    EDGE_BELOW_HURDLE = "edge_below_hurdle"
    UNACTIONABLE = "unactionable"


@dataclass(frozen=True, slots=True)
class CausalAlphaV5SelectiveForecast:
    """One calibrated slow forecast with explicit active/abstention evidence."""

    symbol: str
    decision_indices: np.ndarray
    slow_return_raw: np.ndarray
    slow_direction_raw: np.ndarray
    slow_uncertainty_raw: np.ndarray
    slow_return_calibrated: np.ndarray
    slow_uncertainty_calibrated: np.ndarray
    return_confidence: np.ndarray
    direction_confidence: np.ndarray
    selective_confidence: np.ndarray
    execution_hurdle: np.ndarray
    actionable_mask: np.ndarray
    active_mask: np.ndarray
    states: tuple[V5SelectiveState, ...]
    v4_forecast_digest: str
    calibration_fit_digest: str
    schema_version: str = CAUSAL_ALPHA_V5_SELECTIVE_FORECAST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V5 selective forecast symbol must be non-empty")
        for field_name in ("v4_forecast_digest", "calibration_fit_digest"):
            require_sha256(
                getattr(self, field_name),
                field=f"V5 selective forecast {field_name}",
            )
        decisions = _readonly_vector(
            self.decision_indices,
            dtype=np.int64,
            field="V5 selective decision indices",
        )
        rows = int(decisions.size)
        if np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V5 selective decisions must be strictly increasing")

        arrays: dict[str, np.ndarray] = {}
        for field_name in (
            "slow_return_raw",
            "slow_direction_raw",
            "slow_uncertainty_raw",
            "slow_return_calibrated",
            "slow_uncertainty_calibrated",
            "return_confidence",
            "direction_confidence",
            "selective_confidence",
            "execution_hurdle",
        ):
            array = _readonly_vector(
                getattr(self, field_name),
                dtype=np.float64,
                field=f"V5 selective {field_name}",
            )
            if array.shape != (rows,):
                raise ValueError("V5 selective forecast arrays are not aligned")
            arrays[field_name] = array
        for field_name in ("actionable_mask", "active_mask"):
            mask = _readonly_vector(
                getattr(self, field_name),
                dtype=np.bool_,
                field=f"V5 selective {field_name}",
            )
            if mask.shape != (rows,):
                raise ValueError("V5 selective masks are not aligned")
            arrays[field_name] = mask
        for field_name in (
            "slow_uncertainty_raw",
            "slow_uncertainty_calibrated",
            "return_confidence",
            "direction_confidence",
            "selective_confidence",
            "execution_hurdle",
        ):
            if np.any(arrays[field_name] < 0.0):
                raise ValueError(f"V5 selective {field_name} became negative")
        if np.any(
            arrays["slow_uncertainty_calibrated"] + _V5_EPSILON
            < arrays["slow_uncertainty_raw"]
        ):
            raise ValueError("V5 calibrated uncertainty became smaller than V4")

        states = tuple(V5SelectiveState(state) for state in self.states)
        if len(states) != rows:
            raise ValueError("V5 selective states are not decision aligned")
        state_active = np.asarray(
            [state is V5SelectiveState.ACTIVE for state in states],
            dtype=np.bool_,
        )
        if not np.array_equal(state_active, arrays["active_mask"]):
            raise ValueError("V5 selective states and active mask disagree")
        if np.any(arrays["active_mask"] & ~arrays["actionable_mask"]):
            raise ValueError("V5 inactive rows cannot be marked active")
        if self.schema_version != CAUSAL_ALPHA_V5_SELECTIVE_FORECAST_SCHEMA:
            raise ValueError("unsupported V5 selective forecast schema")

        object.__setattr__(self, "decision_indices", decisions)
        for field_name, array in arrays.items():
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "states", states)
        expected = content_and_arrays_digest(
            {
                "calibration_fit_digest": self.calibration_fit_digest,
                "schema_version": self.schema_version,
                "states": tuple(state.value for state in states),
                "symbol": self.symbol,
                "v4_forecast_digest": self.v4_forecast_digest,
            },
            (
                ("decision_indices", decisions),
                *tuple((name, array) for name, array in arrays.items()),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V5 selective forecast digest mismatch")
        object.__setattr__(self, "digest", expected)


def _aligned_vector(
    value: object,
    *,
    rows: int,
    dtype: Any,
    field: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"{field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return array


def build_causal_alpha_v5_selective_forecast(
    *,
    v4_forecast: CausalAlphaV4Forecast,
    slow_uncertainty: object,
    instrument_descriptors: object,
    instrument_descriptor_available: object,
    one_way_cost_rates: object,
    actionable_mask: object,
    calibration_fit: CausalAlphaV5CalibrationFit,
) -> CausalAlphaV5SelectiveForecast:
    """Calibrate the V4 slow return and predeclare active versus abstained rows."""

    if not isinstance(v4_forecast, CausalAlphaV4Forecast):
        raise TypeError("V5 selective forecast requires a V4 forecast")
    if not isinstance(calibration_fit, CausalAlphaV5CalibrationFit):
        raise TypeError("V5 selective forecast requires calibration evidence")
    if calibration_fit.v4_fit_digest != v4_forecast.fit_digest:
        raise ValueError("V5 calibration and V4 forecast fit identities drifted")

    decisions = np.asarray(v4_forecast.decision_indices, dtype=np.int64).reshape(-1)
    rows = int(decisions.size)
    uncertainty = _aligned_vector(
        slow_uncertainty,
        rows=rows,
        dtype=np.float64,
        field="V5 slow uncertainty",
    )
    costs = _aligned_vector(
        one_way_cost_rates,
        rows=rows,
        dtype=np.float64,
        field="V5 one-way costs",
    )
    actionable = _aligned_vector(
        actionable_mask,
        rows=rows,
        dtype=np.bool_,
        field="V5 actionable mask",
    ).astype(np.bool_, copy=False)
    if np.any(uncertainty < 0.0) or np.any(costs < 0.0):
        raise ValueError("V5 uncertainty and costs must be non-negative")

    descriptors = np.asarray(instrument_descriptors, dtype=np.float64)
    descriptor_available = np.asarray(
        instrument_descriptor_available,
        dtype=np.bool_,
    )
    descriptor_shape = (rows, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
    if (
        descriptors.shape != descriptor_shape
        or descriptor_available.shape != descriptor_shape
    ):
        raise ValueError("V5 instrument descriptors do not match the maintained schema")
    if not np.isfinite(descriptors).all():
        raise ValueError("V5 instrument descriptors must use finite inert storage")

    final = v4_forecast.final_predictions
    directions = v4_forecast.direction_scores
    raw_return = 0.5 * (
        np.asarray(final["24h"], dtype=np.float64)
        + np.asarray(final["72h"], dtype=np.float64) / 3.0
    )
    raw_direction = 0.5 * (
        np.asarray(directions["24h"], dtype=np.float64)
        + np.asarray(directions["72h"], dtype=np.float64)
    )
    if raw_return.shape != (rows,) or raw_direction.shape != (rows,):
        raise ValueError("V5 V4 forecast arrays are not decision aligned")
    if not np.isfinite(raw_return).all() or not np.isfinite(raw_direction).all():
        raise ValueError("V5 V4 forecast arrays must be finite")

    calibration_features = np.column_stack(
        (
            raw_return,
            raw_direction,
            np.log(np.maximum(uncertainty, _V5_EPSILON)),
            descriptors,
        )
    )
    calibration_available = np.column_stack(
        (
            np.ones((rows, 3), dtype=np.bool_),
            descriptor_available,
        )
    )
    residual = calibration_fit.model.predict(
        calibration_features,
        feature_available=calibration_available,
    )
    calibrated_return = raw_return + residual
    calibrated_uncertainty = np.sqrt(
        np.square(uncertainty) + calibration_fit.calibration_residual_rmse**2
    )
    return_confidence = np.abs(calibrated_return) / np.maximum(
        calibrated_uncertainty,
        _V5_EPSILON,
    )
    direction_confidence = np.abs(raw_direction) / max(
        calibration_fit.direction_score_rmse,
        _V5_EPSILON,
    )
    selective_confidence = np.minimum(return_confidence, direction_confidence)
    threshold = calibration_fit.config.minimum_selective_confidence
    selective_confidence = np.where(
        np.isclose(selective_confidence, threshold, rtol=0.0, atol=_V5_EPSILON),
        threshold,
        selective_confidence,
    )
    execution_hurdle = (
        calibration_fit.config.execution_cost_multiplier * costs
        + calibration_fit.config.edge_margin
    )

    effective_actionable = (
        actionable
        & np.asarray(v4_forecast.beta_available, dtype=np.bool_)
        & np.all(descriptor_available, axis=1)
    )
    active = np.zeros(rows, dtype=np.bool_)
    states: list[V5SelectiveState] = []
    for index in range(rows):
        if not bool(effective_actionable[index]):
            states.append(V5SelectiveState.UNACTIONABLE)
            continue
        if (
            abs(float(calibrated_return[index])) <= _V5_EPSILON
            or abs(float(raw_direction[index])) <= _V5_EPSILON
            or calibrated_return[index] * raw_direction[index] <= 0.0
        ):
            states.append(V5SelectiveState.DIRECTION_DISAGREEMENT)
            continue
        if selective_confidence[index] < threshold:
            states.append(V5SelectiveState.CONFIDENCE_ABSTAIN)
            continue
        edge_after_hurdle = (
            abs(float(calibrated_return[index]))
            - float(calibrated_uncertainty[index])
            - float(execution_hurdle[index])
        )
        if edge_after_hurdle <= _V5_EPSILON:
            states.append(V5SelectiveState.EDGE_BELOW_HURDLE)
            continue
        active[index] = True
        states.append(V5SelectiveState.ACTIVE)

    return CausalAlphaV5SelectiveForecast(
        symbol=v4_forecast.symbol,
        decision_indices=decisions,
        slow_return_raw=raw_return,
        slow_direction_raw=raw_direction,
        slow_uncertainty_raw=uncertainty,
        slow_return_calibrated=calibrated_return,
        slow_uncertainty_calibrated=calibrated_uncertainty,
        return_confidence=return_confidence,
        direction_confidence=direction_confidence,
        selective_confidence=selective_confidence,
        execution_hurdle=execution_hurdle,
        actionable_mask=effective_actionable,
        active_mask=active,
        states=tuple(states),
        v4_forecast_digest=v4_forecast.digest,
        calibration_fit_digest=calibration_fit.digest,
    )


_V5_TARGET_REASONS: Final = frozenset(
    {
        "hold_flat",
        "hold_position",
        "entry",
        "add",
        "reduce",
        "exit",
        "flip",
        "unactionable_hold",
        "confidence_abstain",
        "direction_disagreement_hold",
        "edge_below_hurdle_hold",
        "cadence_hold",
        "liquidity_deleverage",
        "risk_projection",
    }
)


def _v5_transition_reason(previous: float, selected: float) -> str:
    if abs(selected - previous) <= _V5_EPSILON:
        return "hold_flat" if abs(previous) <= _V5_EPSILON else "hold_position"
    if abs(previous) <= _V5_EPSILON:
        return "entry"
    if abs(selected) <= _V5_EPSILON:
        return "exit"
    if previous * selected < 0.0:
        return "flip"
    return "add" if abs(selected) > abs(previous) else "reduce"


def _v5_inactive_hold_reason(state: V5SelectiveState) -> str:
    reasons = {
        V5SelectiveState.UNACTIONABLE: "unactionable_hold",
        V5SelectiveState.CONFIDENCE_ABSTAIN: "confidence_abstain",
        V5SelectiveState.DIRECTION_DISAGREEMENT: "direction_disagreement_hold",
        V5SelectiveState.EDGE_BELOW_HURDLE: "edge_below_hurdle_hold",
    }
    try:
        return reasons[state]
    except KeyError as error:
        raise ValueError("active V5 row has no abstention reason") from error


@dataclass(frozen=True, slots=True)
class CausalAlphaV5TargetPath:
    """Selective slow anchor plus unchanged bounded V4 fast impulse."""

    initial_weight: float
    slow_anchors: np.ndarray
    fast_deviations: np.ndarray
    targets: np.ndarray
    slow_expected_returns: np.ndarray
    fast_expected_returns: np.ndarray
    slow_uncertainties: np.ndarray
    fast_uncertainties: np.ndarray
    liquidity_weight_caps: np.ndarray
    risk_weight_caps: np.ndarray
    slow_objectives: np.ndarray
    fast_objective_improvements: np.ndarray
    final_objectives: np.ndarray
    active_mask: np.ndarray
    reasons: tuple[str, ...]
    reason_counts: tuple[tuple[str, int], ...]
    slow_anchor_change_count: int
    fast_impulse_change_count: int
    submitted_change_count: int
    liquidity_deleveraging_count: int
    risk_projection_count: int
    sign_flip_count: int
    selective_forecast_digest: str
    config_digest: str
    schema_version: str = CAUSAL_ALPHA_V5_TARGET_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("V5 target initial weight must be finite")
        require_sha256(
            self.selective_forecast_digest, field="V5 target forecast digest"
        )
        require_sha256(self.config_digest, field="V5 target config digest")
        if self.schema_version != CAUSAL_ALPHA_V5_TARGET_SCHEMA:
            raise ValueError("unsupported V5 target schema")
        names = (
            "slow_anchors",
            "fast_deviations",
            "targets",
            "slow_expected_returns",
            "fast_expected_returns",
            "slow_uncertainties",
            "fast_uncertainties",
            "liquidity_weight_caps",
            "risk_weight_caps",
            "slow_objectives",
            "fast_objective_improvements",
            "final_objectives",
        )
        arrays: dict[str, np.ndarray] = {}
        rows: int | None = None
        for name in names:
            array = _readonly_vector(getattr(self, name), dtype=np.float64, field=name)
            rows = int(array.size) if rows is None else rows
            if array.shape != (rows,):
                raise ValueError("V5 target arrays must align")
            arrays[name] = array
        active = _readonly_vector(self.active_mask, dtype=np.bool_, field="active_mask")
        if active.shape != (rows,):
            raise ValueError("V5 target active mask must align")
        arrays["active_mask"] = active
        if np.any(arrays["slow_uncertainties"] < 0.0) or np.any(
            arrays["fast_uncertainties"] < 0.0
        ):
            raise ValueError("V5 target uncertainty must be non-negative")
        if np.any(arrays["liquidity_weight_caps"] < 0.0) or np.any(
            arrays["risk_weight_caps"] < 0.0
        ):
            raise ValueError("V5 target caps must be non-negative")
        reasons = tuple(self.reasons)
        if len(reasons) != rows or any(
            reason not in _V5_TARGET_REASONS for reason in reasons
        ):
            raise ValueError("V5 target reasons must cover every decision")
        counts = tuple(
            sorted((reason, reasons.count(reason)) for reason in set(reasons))
        )
        if tuple(self.reason_counts) != counts:
            raise ValueError("V5 target reason counts do not match reasons")
        count_names = (
            "slow_anchor_change_count",
            "fast_impulse_change_count",
            "submitted_change_count",
            "liquidity_deleveraging_count",
            "risk_projection_count",
            "sign_flip_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V5 target {name} is invalid")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "reason_counts", counts)
        expected = content_and_arrays_digest(
            {
                "config_digest": self.config_digest,
                "initial_weight": self.initial_weight,
                "reason_counts": counts,
                "reasons": reasons,
                "schema_version": self.schema_version,
                "selective_forecast_digest": self.selective_forecast_digest,
                **{name: getattr(self, name) for name in count_names},
            },
            tuple(arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V5 target path digest mismatch")
        object.__setattr__(self, "digest", expected)


def causal_alpha_v5_target_path(
    selective_forecast: CausalAlphaV5SelectiveForecast,
    prediction_4h: object,
    *,
    direction_score_4h: object,
    uncertainty_4h: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    config: CausalAlphaV4TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> CausalAlphaV5TargetPath:
    """Compile a target without letting abstained rows increase exposure."""

    if not isinstance(selective_forecast, CausalAlphaV5SelectiveForecast):
        raise TypeError("V5 target compiler requires a selective forecast")
    if not isinstance(config, CausalAlphaV4TargetConfig):
        raise TypeError("V5 target compiler requires frozen V4 target config")
    if not math.isfinite(initial_weight):
        raise ValueError("V5 target initial weight must be finite")
    rows = int(selective_forecast.decision_indices.size)
    fast_mu = _aligned_vector(
        prediction_4h, rows=rows, dtype=np.float64, field="fast return"
    )
    direction = _aligned_vector(
        direction_score_4h, rows=rows, dtype=np.float64, field="fast direction"
    )
    fast_sigma = _aligned_vector(
        uncertainty_4h, rows=rows, dtype=np.float64, field="fast uncertainty"
    )
    costs = _aligned_vector(
        one_way_cost_rates, rows=rows, dtype=np.float64, field="one-way costs"
    )
    liquidity_caps = _aligned_vector(
        liquidity_weight_caps, rows=rows, dtype=np.float64, field="liquidity caps"
    )
    risk_caps = (
        np.ones(rows)
        if risk_weight_caps is None
        else _aligned_vector(
            risk_weight_caps, rows=rows, dtype=np.float64, field="risk caps"
        )
    )
    if any(
        np.any(value < 0.0) for value in (fast_sigma, costs, liquidity_caps, risk_caps)
    ):
        raise ValueError("V5 target uncertainty, costs, and caps must be non-negative")
    slow_mu = np.asarray(selective_forecast.slow_return_calibrated)
    slow_sigma = np.asarray(selective_forecast.slow_uncertainty_calibrated)
    active = np.asarray(selective_forecast.active_mask)
    actionable = np.asarray(selective_forecast.actionable_mask)
    output = {
        name: np.empty(rows)
        for name in (
            "slow_anchors",
            "fast_deviations",
            "targets",
            "slow_objectives",
            "fast_objective_improvements",
            "final_objectives",
        )
    }
    reasons: list[str] = []
    previous = current_anchor = float(initial_weight)
    slow_changes = fast_changes = submitted = liquidity_count = risk_count = flips = 0

    def staged(index: int, anchor: float, final: float) -> tuple[float, float, float]:
        return _v4_staged_objective(
            previous=previous,
            anchor=anchor,
            final=final,
            slow_expected_return=float(slow_mu[index]),
            slow_uncertainty=float(slow_sigma[index]),
            fast_expected_return=float(fast_mu[index]),
            fast_uncertainty=float(fast_sigma[index]),
            one_way_cost_rate=float(costs[index]),
            config=config,
        )

    for index in range(rows):
        liquidity_cap = min(float(liquidity_caps[index]), 1.0)
        risk_cap = min(float(risk_caps[index]), 1.0)
        cap = min(liquidity_cap, risk_cap)
        old_anchor = current_anchor
        selected_anchor = float(np.clip(current_anchor, -cap, cap))
        selected = previous
        override: str | None = None
        if abs(previous) > liquidity_cap + _V5_EPSILON:
            selected = selected_anchor = float(
                np.clip(previous, -liquidity_cap, liquidity_cap)
            )
            override = "liquidity_deleverage"
            liquidity_count += 1
        elif abs(previous) > risk_cap + _V5_EPSILON:
            selected = selected_anchor = float(np.clip(previous, -risk_cap, risk_cap))
            override = "risk_projection"
            risk_count += 1
        elif not bool(actionable[index]):
            selected_anchor = current_anchor
            override = "unactionable_hold"
        else:
            if index % config.slow_rebalance_decisions == 0:
                candidates = _v4_slow_candidates(
                    previous=previous,
                    current_anchor=current_anchor,
                    cap=cap,
                    config=config,
                )
                if not bool(active[index]):
                    candidates = tuple(
                        value
                        for value in candidates
                        if _v4_is_risk_reduction(previous, value)
                    )
                selected_anchor, _ = _v4_choose_best(
                    candidates,
                    tuple(staged(index, value, value)[0] for value in candidates),
                    previous=previous,
                )
            else:
                selected_anchor = float(np.clip(current_anchor, -cap, cap))
            if index % config.fast_rebalance_decisions != 0:
                override = "cadence_hold"
            else:
                candidates = _v4_fast_candidates(
                    previous=previous, anchor=selected_anchor, cap=cap, config=config
                )
                if not bool(active[index]):
                    candidates = tuple(
                        value
                        for value in candidates
                        if _v4_is_risk_reduction(previous, value)
                    )
                allowed = tuple(
                    (value, staged(index, selected_anchor, value))
                    for value in candidates
                    if _v4_consensus_allows(
                        previous=previous,
                        target=value,
                        fast_expected_return=float(fast_mu[index]),
                        direction_score=float(direction[index]),
                    )
                )
                if not allowed:
                    selected_anchor = previous
                else:
                    selected, _ = _v4_choose_best(
                        tuple(value for value, _ in allowed),
                        tuple(values[2] for _, values in allowed),
                        previous=previous,
                    )
        slow_score, fast_improvement, final_score = staged(
            index, selected_anchor, selected
        )
        transition = _v5_transition_reason(previous, selected)
        reason = override or (
            _v5_inactive_hold_reason(selective_forecast.states[index])
            if transition in {"hold_flat", "hold_position"} and not bool(active[index])
            else transition
        )
        if abs(selected_anchor - old_anchor) > _V5_EPSILON:
            slow_changes += 1
        current_anchor = float(selected_anchor)
        deviation = selected - selected_anchor
        bounded_exceptions = {
            "cadence_hold",
            "unactionable_hold",
            "liquidity_deleverage",
            "risk_projection",
        }
        if (
            abs(deviation) > config.maximum_fast_absolute_deviation + _V5_EPSILON
            and reason not in bounded_exceptions
        ):
            raise RuntimeError("V5 fast deviation exceeded authored bound")
        if abs(deviation) > _V5_EPSILON:
            fast_changes += 1
        if abs(selected - previous) > _V5_EPSILON:
            submitted += 1
        if previous * selected < 0.0:
            flips += 1
        for name, value in (
            ("slow_anchors", selected_anchor),
            ("fast_deviations", deviation),
            ("targets", selected),
            ("slow_objectives", slow_score),
            ("fast_objective_improvements", fast_improvement),
            ("final_objectives", final_score),
        ):
            output[name][index] = value
        reasons.append(reason)
        previous = float(selected)

    counts = tuple(sorted((reason, reasons.count(reason)) for reason in set(reasons)))
    return CausalAlphaV5TargetPath(
        initial_weight=float(initial_weight),
        slow_anchors=output["slow_anchors"],
        fast_deviations=output["fast_deviations"],
        targets=output["targets"],
        slow_expected_returns=slow_mu,
        fast_expected_returns=fast_mu,
        slow_uncertainties=slow_sigma,
        fast_uncertainties=fast_sigma,
        liquidity_weight_caps=liquidity_caps,
        risk_weight_caps=risk_caps,
        slow_objectives=output["slow_objectives"],
        fast_objective_improvements=output["fast_objective_improvements"],
        final_objectives=output["final_objectives"],
        active_mask=active,
        reasons=tuple(reasons),
        reason_counts=counts,
        slow_anchor_change_count=slow_changes,
        fast_impulse_change_count=fast_changes,
        submitted_change_count=submitted,
        liquidity_deleveraging_count=liquidity_count,
        risk_projection_count=risk_count,
        sign_flip_count=flips,
        selective_forecast_digest=selective_forecast.digest,
        config_digest=config.digest,
    )


__all__ = [
    "CAUSAL_ALPHA_V5_CALIBRATION_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES",
    "CAUSAL_ALPHA_V5_CALIBRATION_FIT_SCHEMA",
    "CAUSAL_ALPHA_V5_SELECTIVE_FORECAST_SCHEMA",
    "CAUSAL_ALPHA_V5_TARGET_SCHEMA",
    "CausalAlphaV5CalibrationConfig",
    "CausalAlphaV5CalibrationFit",
    "CausalAlphaV5SelectiveForecast",
    "CausalAlphaV5TargetPath",
    "V5SelectiveState",
    "build_causal_alpha_v5_selective_forecast",
    "causal_alpha_v5_target_path",
]
