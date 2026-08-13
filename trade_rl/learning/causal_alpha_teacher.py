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
CAUSAL_ALPHA_COST_AWARE_SCHEMA: Final = "causal_alpha_cost_aware_v1"
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

    def transform(
        self,
        features: object,
        *,
        feature_available: object | None = None,
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match causal alpha schema")
        if not np.isfinite(values).all():
            raise ValueError("prediction features must be finite")
        availability: np.ndarray | None = None
        if feature_available is not None:
            availability = np.asarray(feature_available, dtype=np.bool_)
            if availability.shape != values.shape:
                raise ValueError(
                    "prediction feature availability must match prediction features"
                )
        scaled = (values - self.location) / self.scale
        scaled[:, self.constant_mask] = 0.0
        if availability is not None:
            scaled = np.where(availability, scaled, 0.0)
        return scaled

    def predict(
        self,
        features: object,
        *,
        feature_available: object | None = None,
    ) -> np.ndarray:
        scaled = self.transform(features, feature_available=feature_available)
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
    eligible_mask = finite_rows & (label_end >= 0) & (label_end < knowledge_cutoff)
    eligible_indices = np.flatnonzero(eligible_mask).astype(np.int64)
    if eligible_indices.size < 2:
        raise ValueError("causal alpha prefix contains insufficient fitted samples")
    x = values[eligible_indices]
    x_available = available[eligible_indices]
    y = target[eligible_indices]

    # Mirror the Universal observation contract: unavailable feature values are
    # represented by standardized zero rather than discarding the entire row.
    # Statistics are feature-local and use only values known to be available.
    available_count = x_available.sum(axis=0, dtype=np.int64)
    available_sum = np.where(x_available, x, 0.0).sum(axis=0, dtype=np.float64)
    location = np.zeros(x.shape[1], dtype=np.float64)
    np.divide(
        available_sum,
        available_count,
        out=location,
        where=available_count > 0,
    )
    centered = np.where(x_available, x - location, 0.0)
    squared_sum = np.square(centered).sum(axis=0, dtype=np.float64)
    variance = np.zeros(x.shape[1], dtype=np.float64)
    np.divide(
        squared_sum,
        available_count,
        out=variance,
        where=available_count > 0,
    )
    raw_scale = np.sqrt(variance)
    constant_mask = (available_count < 2) | (raw_scale <= _EPSILON)
    scale = np.where(constant_mask, 1.0, raw_scale)
    scaled = np.where(x_available, (x - location) / scale, 0.0)
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
class CausalAlphaCostAwareConfig:
    execution_cost_multiplier: float
    edge_margin: float
    confirmation_count: int
    strong_reversal_threshold: float
    max_abs_target: float
    schema_version: str = CAUSAL_ALPHA_COST_AWARE_SCHEMA

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.execution_cost_multiplier)
            or self.execution_cost_multiplier <= 0.0
        ):
            raise ValueError("execution_cost_multiplier must be finite and positive")
        if not math.isfinite(self.edge_margin) or self.edge_margin < 0.0:
            raise ValueError("edge_margin must be finite and non-negative")
        if (
            isinstance(self.confirmation_count, bool)
            or not isinstance(self.confirmation_count, int)
            or self.confirmation_count < 1
        ):
            raise ValueError("confirmation_count must be a positive integer")
        if (
            not math.isfinite(self.strong_reversal_threshold)
            or self.strong_reversal_threshold <= 0.0
        ):
            raise ValueError("strong_reversal_threshold must be finite and positive")
        if not math.isfinite(self.max_abs_target) or not (
            0.0 < self.max_abs_target <= 1.0
        ):
            raise ValueError("max_abs_target must be finite and in (0, 1]")
        if self.schema_version != CAUSAL_ALPHA_COST_AWARE_SCHEMA:
            raise ValueError("unsupported causal alpha cost-aware schema")

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
    actionable_mask: np.ndarray | None = None
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("initial_weight must be finite")
        targets = np.asarray(self.targets, dtype=np.float64).reshape(-1).copy()
        if not np.isfinite(targets).all():
            raise ValueError("causal alpha targets must be finite")
        targets.setflags(write=False)
        if self.actionable_mask is None:
            actionable = np.ones(targets.shape, dtype=np.bool_)
        else:
            actionable = (
                np.asarray(self.actionable_mask, dtype=np.bool_).reshape(-1).copy()
            )
            if actionable.shape != targets.shape:
                raise ValueError("actionable_mask must align with causal alpha targets")
        actionable.setflags(write=False)
        for field in (
            "submitted_change_count",
            "suppressed_change_count",
            "sign_flip_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "actionable_mask", actionable)
        expected = content_digest(
            {
                "actionable_mask": actionable.astype(bool).tolist(),
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


@dataclass(frozen=True, slots=True)
class CausalAlphaCostAwareTargetPath:
    initial_weight: float
    targets: np.ndarray
    proposed_turnover: np.ndarray
    predicted_incremental_edge: np.ndarray
    estimated_cost_hurdle: np.ndarray
    edge_to_cost_ratio: np.ndarray
    confirmation_state: np.ndarray
    cost_suppressed_change_count: int
    submitted_change_count: int
    strong_reversal_count: int
    sign_flip_count: int
    actionable_mask: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("initial_weight must be finite")
        arrays: dict[str, np.ndarray] = {}
        for field in (
            "targets",
            "proposed_turnover",
            "predicted_incremental_edge",
            "estimated_cost_hurdle",
            "edge_to_cost_ratio",
        ):
            value = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1).copy()
            if not np.isfinite(value).all():
                raise ValueError(f"{field} must be finite")
            value.setflags(write=False)
            arrays[field] = value
        shape = arrays["targets"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("cost-aware target path arrays must align")
        confirmation = (
            np.asarray(self.confirmation_state, dtype=np.int64).reshape(-1).copy()
        )
        actionable = np.asarray(self.actionable_mask, dtype=np.bool_).reshape(-1).copy()
        if confirmation.shape != shape or actionable.shape != shape:
            raise ValueError("cost-aware state arrays must align with targets")
        confirmation.setflags(write=False)
        actionable.setflags(write=False)
        for field in (
            "cost_suppressed_change_count",
            "submitted_change_count",
            "strong_reversal_count",
            "sign_flip_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field, value in arrays.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "confirmation_state", confirmation)
        object.__setattr__(self, "actionable_mask", actionable)
        payload = {
            "actionable_mask": actionable.astype(bool).tolist(),
            "confirmation_state": confirmation.tolist(),
            "cost_suppressed_change_count": self.cost_suppressed_change_count,
            "edge_to_cost_ratio": arrays["edge_to_cost_ratio"].tolist(),
            "estimated_cost_hurdle": arrays["estimated_cost_hurdle"].tolist(),
            "initial_weight": float(self.initial_weight),
            "predicted_incremental_edge": arrays[
                "predicted_incremental_edge"
            ].tolist(),
            "proposed_turnover": arrays["proposed_turnover"].tolist(),
            "schema_version": CAUSAL_ALPHA_COST_AWARE_SCHEMA,
            "sign_flip_count": self.sign_flip_count,
            "strong_reversal_count": self.strong_reversal_count,
            "submitted_change_count": self.submitted_change_count,
            "targets": arrays["targets"].tolist(),
        }
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("cost-aware target path digest mismatch")
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
    actionable_mask: object | None = None,
) -> CausalAlphaTargetPath:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("causal alpha scores must be finite")
    if not math.isfinite(initial_weight):
        raise ValueError("initial_weight must be finite")
    if actionable_mask is None:
        actionable = np.ones(values.shape, dtype=np.bool_)
    else:
        actionable = np.asarray(actionable_mask, dtype=np.bool_).reshape(-1)
        if actionable.shape != values.shape:
            raise ValueError("actionable_mask must align with causal alpha scores")
    previous = float(initial_weight)
    targets = np.empty(values.size, dtype=np.float64)
    submitted = 0
    suppressed = 0
    sign_flips = 0
    for index, score in enumerate(values):
        if not bool(actionable[index]):
            targets[index] = previous
            continue
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
        actionable_mask=actionable,
    )


def causal_alpha_cost_aware_target_path(
    scores: object,
    *,
    one_way_cost_rates: object,
    controller: CausalAlphaControllerConfig,
    economic: CausalAlphaCostAwareConfig,
    initial_weight: float,
    actionable_mask: object | None = None,
) -> CausalAlphaCostAwareTargetPath:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    cost_rates = np.asarray(one_way_cost_rates, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("causal alpha scores must be finite")
    if cost_rates.shape != values.shape or not np.isfinite(cost_rates).all():
        raise ValueError("one_way_cost_rates must align with scores and be finite")
    if np.any(cost_rates < 0.0):
        raise ValueError("one_way_cost_rates must be non-negative")
    if not math.isfinite(initial_weight):
        raise ValueError("initial_weight must be finite")
    if actionable_mask is None:
        actionable = np.ones(values.shape, dtype=np.bool_)
    else:
        actionable = np.asarray(actionable_mask, dtype=np.bool_).reshape(-1)
        if actionable.shape != values.shape:
            raise ValueError("actionable_mask must align with causal alpha scores")

    targets = np.empty(values.size, dtype=np.float64)
    proposed_turnover = np.zeros(values.size, dtype=np.float64)
    incremental_edge = np.zeros(values.size, dtype=np.float64)
    cost_hurdle = np.zeros(values.size, dtype=np.float64)
    edge_to_cost = np.zeros(values.size, dtype=np.float64)
    confirmation_state = np.zeros(values.size, dtype=np.int64)
    previous = float(initial_weight)
    pending_direction = 0
    pending_count = 0
    cost_suppressed = 0
    submitted = 0
    strong_reversals = 0
    sign_flips = 0

    for index, score_value in enumerate(values):
        if abs(previous) > economic.max_abs_target:
            target = float(
                np.clip(
                    previous,
                    -economic.max_abs_target,
                    economic.max_abs_target,
                )
            )
            turnover = abs(target - previous)
            proposed_turnover[index] = turnover
            cost_hurdle[index] = turnover * (
                float(cost_rates[index]) * economic.execution_cost_multiplier
                + economic.edge_margin
            )
            previous = target
            targets[index] = previous
            submitted += 1
            pending_direction = 0
            pending_count = 0
            continue
        if not bool(actionable[index]):
            targets[index] = previous
            confirmation_state[index] = pending_direction * pending_count
            continue

        score = float(score_value)
        desired = float(
            np.clip(
                _desired_target(score, previous, controller),
                -economic.max_abs_target,
                economic.max_abs_target,
            )
        )
        delta = desired - previous
        turnover = abs(delta)
        edge = score * delta
        hurdle = turnover * (
            float(cost_rates[index]) * economic.execution_cost_multiplier
            + economic.edge_margin
        )
        proposed_turnover[index] = turnover
        incremental_edge[index] = edge
        cost_hurdle[index] = hurdle
        edge_to_cost[index] = edge / hurdle if hurdle > _EPSILON else 0.0

        if turnover <= _EPSILON:
            pending_direction = 0
            pending_count = 0
            targets[index] = previous
            continue
        direction = 1 if delta > 0.0 else -1
        if edge <= hurdle:
            pending_direction = 0
            pending_count = 0
            cost_suppressed += 1
            targets[index] = previous
            continue
        if direction == pending_direction:
            pending_count += 1
        else:
            pending_direction = direction
            pending_count = 1

        is_reversal = previous * desired < 0.0
        is_strong_reversal = (
            is_reversal and abs(score) >= economic.strong_reversal_threshold
        )
        if is_strong_reversal:
            pending_count = 1
        confirmation_state[index] = pending_direction * pending_count
        if not is_strong_reversal and pending_count < economic.confirmation_count:
            targets[index] = previous
            continue

        bounded = float(
            np.clip(
                desired,
                previous - controller.max_target_delta,
                previous + controller.max_target_delta,
            )
        )
        if abs(bounded - previous) <= controller.no_trade_band:
            targets[index] = previous
            continue
        if is_strong_reversal:
            strong_reversals += 1
        if previous * bounded < 0.0:
            sign_flips += 1
        previous = bounded
        targets[index] = previous
        submitted += 1

    return CausalAlphaCostAwareTargetPath(
        initial_weight=float(initial_weight),
        targets=targets,
        proposed_turnover=proposed_turnover,
        predicted_incremental_edge=incremental_edge,
        estimated_cost_hurdle=cost_hurdle,
        edge_to_cost_ratio=edge_to_cost,
        confirmation_state=confirmation_state,
        cost_suppressed_change_count=cost_suppressed,
        submitted_change_count=submitted,
        strong_reversal_count=strong_reversals,
        sign_flip_count=sign_flips,
        actionable_mask=actionable,
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaTeacherHoldoutMetric:
    """One untouched train-symbol holdout replay for teacher pre-admission."""

    symbol: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    maximum_drawdown: float
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("causal alpha teacher holdout symbol must be non-empty")
        for field, value in (
            ("gross_return", self.gross_return),
            ("net_return", self.net_return),
            ("turnover_per_day", self.turnover_per_day),
            ("total_execution_cost", self.total_execution_cost),
            ("maximum_drawdown", self.maximum_drawdown),
        ):
            if not math.isfinite(value):
                raise ValueError(f"causal alpha teacher {field} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha teacher turnover/cost must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("causal alpha teacher trade_count must be non-negative")
        expected = content_digest(
            {
                "gross_return": self.gross_return,
                "maximum_drawdown": self.maximum_drawdown,
                "net_return": self.net_return,
                "schema_version": "causal_alpha_teacher_holdout_metric_v1",
                "symbol": self.symbol,
                "total_execution_cost": self.total_execution_cost,
                "trade_count": self.trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher holdout metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "gross_return": self.gross_return,
            "maximum_drawdown": self.maximum_drawdown,
            "net_return": self.net_return,
            "schema_version": "causal_alpha_teacher_holdout_metric_v1",
            "symbol": self.symbol,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaTeacherAdmissionEvidence:
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...]
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics or len({item.symbol for item in metrics}) != len(metrics):
            raise ValueError("causal alpha teacher holdout symbols must be unique")
        if not math.isfinite(self.aggregate_gross_return) or not math.isfinite(
            self.aggregate_net_return
        ):
            raise ValueError("causal alpha teacher aggregate returns must be finite")
        if (
            isinstance(self.negative_gross_symbol_count, bool)
            or not isinstance(self.negative_gross_symbol_count, int)
            or not 0 <= self.negative_gross_symbol_count <= len(metrics)
        ):
            raise ValueError("causal alpha teacher negative holdout count is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("causal alpha teacher admission reasons are inconsistent")
        expected = content_digest(
            {
                "aggregate_gross_return": self.aggregate_gross_return,
                "aggregate_net_return": self.aggregate_net_return,
                "metric_digests": tuple(item.digest for item in metrics),
                "negative_gross_symbol_count": self.negative_gross_symbol_count,
                "passed": self.passed,
                "rejection_reasons": reasons,
                "schema_version": "causal_alpha_teacher_admission_v1",
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher admission digest mismatch")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "artifact_digest": self.digest,
            "metrics": [item.to_payload() for item in self.metrics],
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
            "schema_version": "causal_alpha_teacher_admission_v1",
        }


def evaluate_causal_alpha_teacher_admission(
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...],
) -> CausalAlphaTeacherAdmissionEvidence:
    """Apply the maintained pre-BC teacher economics gate to untouched holdouts."""

    values = tuple(metrics)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("causal alpha teacher holdout symbols must be unique")
    aggregate_gross = float(sum(item.gross_return for item in values))
    aggregate_net = float(sum(item.net_return for item in values))
    negative_count = sum(item.gross_return < 0.0 for item in values)
    reasons: list[str] = []
    if aggregate_gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if negative_count > len(values) // 2:
        reasons.append("majority_negative_gross_holdouts")
    return CausalAlphaTeacherAdmissionEvidence(
        metrics=values,
        aggregate_gross_return=aggregate_gross,
        aggregate_net_return=aggregate_net,
        negative_gross_symbol_count=negative_count,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CAUSAL_ALPHA_TEACHER_KIND",
    "CausalAlphaCostAwareConfig",
    "CausalAlphaCostAwareTargetPath",
    "CausalAlphaControllerConfig",
    "CausalAlphaHorizonMix",
    "CausalAlphaRidgeConfig",
    "CausalAlphaRidgeModel",
    "CausalAlphaTargetPath",
    "CausalAlphaTeacherAdmissionEvidence",
    "CausalAlphaTeacherHoldoutMetric",
    "ForwardLogReturnLabel",
    "causal_alpha_cost_aware_target_path",
    "causal_alpha_target_path",
    "combine_causal_alpha_predictions",
    "fit_causal_alpha_ridge",
    "forward_log_return_label",
]
