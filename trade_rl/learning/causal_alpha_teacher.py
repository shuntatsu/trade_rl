"""Deterministic train-only signal and controller primitives for Universal BC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest

CAUSAL_ALPHA_TEACHER_KIND: Final = "causal_alpha_ridge"
CAUSAL_ALPHA_RIDGE_SCHEMA: Final = "causal_alpha_ridge_v1"
CAUSAL_ALPHA_CONTROLLER_SCHEMA: Final = "causal_alpha_controller_v1"
_EPSILON: Final = 1e-12


class CausalAlphaHorizonMix(str, Enum):
    """Maintained ways to combine the independently fitted 24h/72h signals."""

    H24 = "24h"
    H72 = "72h"
    EQUAL = "equal"


@dataclass(frozen=True, slots=True)
class ForwardLogReturnLabel:
    """One gross forward return whose timing is explicit and auditable."""

    value: float
    decision_index: int
    execution_start_index: int
    label_end_index: int
    horizon_bars: int


def _single_symbol_matrix(value: object, *, field: str, n_bars: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (n_bars, 1) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a finite single-symbol bar matrix")
    return array


def forward_log_return_label(
    dataset: Any,
    *,
    decision_index: int,
    horizon_hours: float,
    signal_delay_decisions: int,
    decision_bars: int,
) -> ForwardLogReturnLabel:
    """Return gross open-to-close forward log return for one causal decision.

    MarketDataset rows represent bar closes.  A target submitted at close ``t``
    can first fill on processing bar ``t + 1``.  With the maintained one-decision
    delay, that submitted target becomes executable one complete decision later.
    """

    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise ValueError("decision_index must be an integer")
    if decision_index < 0:
        raise ValueError("decision_index must be non-negative")
    if signal_delay_decisions not in {0, 1}:
        raise ValueError("signal_delay_decisions must be zero or one")
    if isinstance(decision_bars, bool) or not isinstance(decision_bars, int):
        raise ValueError("decision_bars must be an integer")
    if decision_bars <= 0:
        raise ValueError("decision_bars must be positive")
    if not math.isfinite(horizon_hours) or horizon_hours <= 0.0:
        raise ValueError("horizon_hours must be finite and positive")
    if not bool(getattr(dataset, "regular_cadence", False)):
        raise ValueError("causal alpha labels require a regular market cadence")
    bars_for_hours = getattr(dataset, "bars_for_hours", None)
    if not callable(bars_for_hours):
        raise TypeError("dataset cannot resolve exact real-time horizons")
    horizon_bars = int(bars_for_hours(float(horizon_hours)))
    if horizon_bars <= 0:
        raise ValueError("label horizon must contain at least one bar")
    n_bars = int(getattr(dataset, "n_bars", 0))
    if n_bars <= 0:
        raise ValueError("dataset bar count is invalid")
    execution_start = decision_index + signal_delay_decisions * decision_bars + 1
    label_end = execution_start + horizon_bars - 1
    if execution_start <= decision_index or label_end >= n_bars:
        raise ValueError("label horizon is incomplete inside the dataset")
    open_values = _single_symbol_matrix(dataset.open, field="open", n_bars=n_bars)
    close_values = _single_symbol_matrix(dataset.close, field="close", n_bars=n_bars)
    start_price = float(open_values[execution_start, 0])
    end_price = float(close_values[label_end, 0])
    if start_price <= 0.0 or end_price <= 0.0:
        raise ValueError("label prices must be positive")
    value = math.log(end_price / start_price)
    if not math.isfinite(value):
        raise ValueError("forward log-return label must be finite")
    return ForwardLogReturnLabel(
        value=value,
        decision_index=decision_index,
        execution_start_index=execution_start,
        label_end_index=label_end,
        horizon_bars=horizon_bars,
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaRidgeConfig:
    """Numerically explicit pooled ridge configuration."""

    ridge_strength: float
    schema_version: str = CAUSAL_ALPHA_RIDGE_SCHEMA

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge_strength) or self.ridge_strength <= 0.0:
            raise ValueError("ridge_strength must be finite and positive")
        if self.schema_version != CAUSAL_ALPHA_RIDGE_SCHEMA:
            raise ValueError("unsupported causal alpha ridge schema")

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaRidgeModel:
    """Immutable prefix-fitted scaler and ridge coefficients."""

    feature_names: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray
    coefficients: np.ndarray
    intercept: float
    sample_count: int
    knowledge_cutoff: int
    eligible_indices: np.ndarray
    config: CausalAlphaRidgeConfig
    digest: str = ""

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        width = len(names)
        if width == 0 or any(not name for name in names) or len(set(names)) != width:
            raise ValueError("causal alpha feature names must be non-empty and unique")
        arrays: dict[str, np.ndarray] = {}
        for field, value, dtype in (
            ("location", self.location, np.float64),
            ("scale", self.scale, np.float64),
            ("coefficients", self.coefficients, np.float64),
            ("constant_mask", self.constant_mask, np.bool_),
        ):
            array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
            if array.shape != (width,):
                raise ValueError(f"{field} must match causal alpha feature width")
            if dtype is not np.bool_ and not np.isfinite(array).all():
                raise ValueError(f"{field} must be finite")
            array.setflags(write=False)
            arrays[field] = array
        if np.any(arrays["scale"] <= 0.0):
            raise ValueError("causal alpha feature scale must be positive")
        eligible = np.asarray(self.eligible_indices, dtype=np.int64).reshape(-1).copy()
        if eligible.size != self.sample_count or np.any(eligible < 0):
            raise ValueError("eligible_indices must match fitted sample_count")
        eligible.setflags(write=False)
        if self.sample_count < 2:
            raise ValueError("causal alpha ridge requires at least two samples")
        if self.knowledge_cutoff <= 0:
            raise ValueError("knowledge_cutoff must be positive")
        if not math.isfinite(self.intercept):
            raise ValueError("causal alpha ridge intercept must be finite")
        if not isinstance(self.config, CausalAlphaRidgeConfig):
            raise TypeError("config must be CausalAlphaRidgeConfig")
        object.__setattr__(self, "feature_names", names)
        for field, array in arrays.items():
            object.__setattr__(self, field, array)
        object.__setattr__(self, "eligible_indices", eligible)
        expected = content_digest(self.to_payload())
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha ridge model digest mismatch")
        object.__setattr__(self, "digest", expected)

    def transform(self, features: object) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match causal alpha schema")
        if not np.isfinite(values).all():
            raise ValueError("prediction features must be finite")
        scaled = (values - self.location) / self.scale
        scaled[:, self.constant_mask] = 0.0
        return scaled

    def predict(self, features: object) -> np.ndarray:
        scaled = self.transform(features)
        prediction = self.intercept + scaled @ self.coefficients
        if not np.isfinite(prediction).all():
            raise ValueError("causal alpha prediction became non-finite")
        return np.asarray(prediction, dtype=np.float64)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": CAUSAL_ALPHA_RIDGE_SCHEMA,
            "feature_names": list(self.feature_names),
            "location": self.location.tolist(),
            "scale": self.scale.tolist(),
            "constant_mask": self.constant_mask.astype(bool).tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "sample_count": int(self.sample_count),
            "knowledge_cutoff": int(self.knowledge_cutoff),
            "eligible_indices": self.eligible_indices.tolist(),
            "ridge_config_digest": self.config.digest,
        }


def fit_causal_alpha_ridge(
    *,
    features: object,
    labels: object,
    feature_available: object,
    label_end_indices: object,
    knowledge_cutoff: int,
    feature_names: tuple[str, ...],
    config: CausalAlphaRidgeConfig,
) -> CausalAlphaRidgeModel:
    """Fit scaler and ridge using only fully realized prefix labels."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    available = np.asarray(feature_available, dtype=np.bool_)
    label_end = np.asarray(label_end_indices, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("causal alpha training features have invalid shape")
    rows = values.shape[0]
    if target.shape != (rows,) or available.shape != values.shape:
        raise ValueError("causal alpha training arrays are not sample aligned")
    if label_end.shape != (rows,):
        raise ValueError("causal alpha label-end indices are not sample aligned")
    if isinstance(knowledge_cutoff, bool) or not isinstance(knowledge_cutoff, int):
        raise ValueError("knowledge_cutoff must be an integer")
    finite_rows = np.isfinite(values).all(axis=1) & np.isfinite(target)
    eligible_mask = (
        finite_rows
        & available.all(axis=1)
        & (label_end >= 0)
        & (label_end < knowledge_cutoff)
    )
    eligible_indices = np.flatnonzero(eligible_mask).astype(np.int64)
    if eligible_indices.size < 2:
        raise ValueError("causal alpha prefix contains insufficient fitted samples")
    x = values[eligible_indices]
    y = target[eligible_indices]
    location = x.mean(axis=0, dtype=np.float64)
    raw_scale = x.std(axis=0, dtype=np.float64)
    constant_mask = raw_scale <= _EPSILON
    scale = np.where(constant_mask, 1.0, raw_scale)
    scaled = (x - location) / scale
    scaled[:, constant_mask] = 0.0
    design = np.column_stack((np.ones(scaled.shape[0], dtype=np.float64), scaled))
    gram = design.T @ design
    penalty = np.eye(design.shape[1], dtype=np.float64) * config.ridge_strength
    penalty[0, 0] = 0.0
    rhs = design.T @ y
    try:
        solution = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("causal alpha ridge solve failed") from error
    if not np.isfinite(solution).all():
        raise ValueError("causal alpha ridge coefficients are non-finite")
    return CausalAlphaRidgeModel(
        feature_names=tuple(feature_names),
        location=location,
        scale=scale,
        constant_mask=constant_mask,
        coefficients=solution[1:],
        intercept=float(solution[0]),
        sample_count=int(eligible_indices.size),
        knowledge_cutoff=knowledge_cutoff,
        eligible_indices=eligible_indices,
        config=config,
    )


def combine_causal_alpha_predictions(
    prediction_24h: object,
    prediction_72h: object,
    horizon_mix: CausalAlphaHorizonMix | str,
) -> np.ndarray:
    first = np.asarray(prediction_24h, dtype=np.float64)
    second = np.asarray(prediction_72h, dtype=np.float64)
    if (
        first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("causal alpha horizon predictions must be aligned and finite")
    mix = CausalAlphaHorizonMix(horizon_mix)
    if mix is CausalAlphaHorizonMix.H24:
        return first.copy()
    if mix is CausalAlphaHorizonMix.H72:
        return second.copy()
    return (0.5 * (first + second)).astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class CausalAlphaControllerConfig:
    horizon_mix: CausalAlphaHorizonMix | str
    score_scale: float
    entry_threshold: float
    exit_threshold: float
    no_trade_band: float
    max_target_delta: float
    schema_version: str = CAUSAL_ALPHA_CONTROLLER_SCHEMA

    def __post_init__(self) -> None:
        mix = CausalAlphaHorizonMix(self.horizon_mix)
        if not math.isfinite(self.score_scale) or self.score_scale <= 0.0:
            raise ValueError("score_scale must be finite and positive")
        for field, value in (
            ("entry_threshold", self.entry_threshold),
            ("exit_threshold", self.exit_threshold),
            ("no_trade_band", self.no_trade_band),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be lower than entry_threshold")
        if not math.isfinite(self.max_target_delta) or self.max_target_delta <= 0.0:
            raise ValueError("max_target_delta must be finite and positive")
        if self.schema_version != CAUSAL_ALPHA_CONTROLLER_SCHEMA:
            raise ValueError("unsupported causal alpha controller schema")
        object.__setattr__(self, "horizon_mix", mix)

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaTargetPath:
    initial_weight: float
    targets: np.ndarray
    submitted_change_count: int
    suppressed_change_count: int
    sign_flip_count: int
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("initial_weight must be finite")
        targets = np.asarray(self.targets, dtype=np.float64).reshape(-1).copy()
        if not np.isfinite(targets).all():
            raise ValueError("causal alpha targets must be finite")
        targets.setflags(write=False)
        for field in (
            "submitted_change_count",
            "suppressed_change_count",
            "sign_flip_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        object.__setattr__(self, "targets", targets)
        expected = content_digest(
            {
                "initial_weight": float(self.initial_weight),
                "targets": targets.tolist(),
                "submitted_change_count": self.submitted_change_count,
                "suppressed_change_count": self.suppressed_change_count,
                "sign_flip_count": self.sign_flip_count,
                "schema_version": CAUSAL_ALPHA_CONTROLLER_SCHEMA,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha target path digest mismatch")
        object.__setattr__(self, "digest", expected)


def _desired_target(
    score: float, previous: float, config: CausalAlphaControllerConfig
) -> float:
    previous_sign = 0 if abs(previous) <= _EPSILON else (1 if previous > 0.0 else -1)
    score_sign = 0 if abs(score) <= _EPSILON else (1 if score > 0.0 else -1)
    magnitude = abs(score)
    if previous_sign == 0:
        if magnitude < config.entry_threshold:
            return 0.0
    elif score_sign != previous_sign:
        if magnitude < config.entry_threshold:
            return 0.0
    elif magnitude < config.exit_threshold:
        return 0.0
    return math.tanh(score * config.score_scale)


def causal_alpha_target_path(
    scores: object,
    *,
    config: CausalAlphaControllerConfig,
    initial_weight: float,
) -> CausalAlphaTargetPath:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("causal alpha scores must be finite")
    if not math.isfinite(initial_weight):
        raise ValueError("initial_weight must be finite")
    previous = float(initial_weight)
    targets = np.empty(values.size, dtype=np.float64)
    submitted = 0
    suppressed = 0
    sign_flips = 0
    for index, score in enumerate(values):
        desired = _desired_target(float(score), previous, config)
        bounded = float(
            np.clip(
                desired,
                previous - config.max_target_delta,
                previous + config.max_target_delta,
            )
        )
        if abs(bounded - previous) <= config.no_trade_band:
            target = previous
            if abs(desired - previous) > _EPSILON:
                suppressed += 1
        else:
            target = bounded
            submitted += 1
        if previous * target < 0.0:
            sign_flips += 1
        targets[index] = target
        previous = target
    return CausalAlphaTargetPath(
        initial_weight=float(initial_weight),
        targets=targets,
        submitted_change_count=submitted,
        suppressed_change_count=suppressed,
        sign_flip_count=sign_flips,
    )


__all__ = [
    "CAUSAL_ALPHA_TEACHER_KIND",
    "CausalAlphaControllerConfig",
    "CausalAlphaHorizonMix",
    "CausalAlphaRidgeConfig",
    "CausalAlphaRidgeModel",
    "CausalAlphaTargetPath",
    "ForwardLogReturnLabel",
    "causal_alpha_target_path",
    "combine_causal_alpha_predictions",
    "fit_causal_alpha_ridge",
    "forward_log_return_label",
]
