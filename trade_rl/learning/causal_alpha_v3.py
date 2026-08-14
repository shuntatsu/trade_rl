"""Research-only overlap and target primitives for causal alpha V3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest

CAUSAL_ALPHA_V3_FIT_SCHEMA: Final = "causal_alpha_v3_fit_config_v1"
CAUSAL_ALPHA_V3_FORECAST_SCHEMA: Final = "causal_alpha_v3_forecast_v1"
CAUSAL_ALPHA_V3_TARGET_SCHEMA: Final = "causal_alpha_v3_target_v1"
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class CausalAlphaV3FitConfig:
    """Numerical contract for the research-only V3 pooled predictor."""

    ridge_strength: float = 0.1
    schema_version: str = CAUSAL_ALPHA_V3_FIT_SCHEMA

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge_strength) or self.ridge_strength <= 0.0:
            raise ValueError("ridge_strength must be finite and positive")
        if self.schema_version != CAUSAL_ALPHA_V3_FIT_SCHEMA:
            raise ValueError("unsupported causal alpha V3 fit schema")

    @property
    def digest(self) -> str:
        return content_digest(self)


def causal_alpha_overlap_uniqueness_weights(
    decision_indices: object,
    label_end_indices: object,
    *,
    knowledge_cutoff: int,
) -> np.ndarray:
    """Return inverse-concurrency uniqueness for fully realized label intervals.

    A label interval is represented by the first bar after the decision through
    its realized label end. Rows whose label is not fully known before the
    knowledge cutoff receive zero weight.
    """

    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    label_ends = np.asarray(label_end_indices, dtype=np.int64).reshape(-1)
    if decisions.shape != label_ends.shape or decisions.size == 0:
        raise ValueError("V3 overlap inputs must be non-empty and sample aligned")
    if np.any(decisions < 0):
        raise ValueError("V3 decision indices must be non-negative")
    if isinstance(knowledge_cutoff, bool) or not isinstance(knowledge_cutoff, int):
        raise ValueError("knowledge_cutoff must be an integer")
    starts = decisions + 1
    eligible = (
        (label_ends >= starts)
        & (label_ends >= 0)
        & (label_ends < knowledge_cutoff)
    )
    weights = np.zeros(decisions.shape, dtype=np.float64)
    if not np.any(eligible):
        return weights

    eligible_starts = starts[eligible]
    eligible_ends = label_ends[eligible]
    offset = int(np.min(eligible_starts))
    final = int(np.max(eligible_ends))
    difference = np.zeros(final - offset + 2, dtype=np.int64)
    for start, end in zip(eligible_starts, eligible_ends, strict=True):
        difference[int(start) - offset] += 1
        difference[int(end) - offset + 1] -= 1
    concurrency = np.cumsum(difference[:-1], dtype=np.int64)
    if np.any(concurrency < 0):
        raise RuntimeError("V3 label concurrency became negative")

    for row in np.flatnonzero(eligible):
        start = int(starts[row]) - offset
        end = int(label_ends[row]) - offset + 1
        interval = concurrency[start:end]
        if interval.size == 0 or np.any(interval <= 0):
            raise RuntimeError("V3 label concurrency does not cover an eligible row")
        weights[row] = float(np.mean(1.0 / interval, dtype=np.float64))
    return weights


@dataclass(frozen=True, slots=True)
class CausalAlphaV3Forecast:
    prediction_24h: np.ndarray
    prediction_72h: np.ndarray
    expected_return_24h_equivalent: np.ndarray
    uncertainty_24h_equivalent: np.ndarray
    signal_to_uncertainty: np.ndarray
    residual_rmse_24h: float
    residual_rmse_72h: float
    digest: str = ""

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        shape: tuple[int, ...] | None = None
        for field in (
            "prediction_24h",
            "prediction_72h",
            "expected_return_24h_equivalent",
            "uncertainty_24h_equivalent",
            "signal_to_uncertainty",
        ):
            value = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1).copy()
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"V3 forecast {field} must be finite and non-empty")
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError("V3 forecast arrays must be sample aligned")
            value.setflags(write=False)
            arrays[field] = value
        if np.any(arrays["uncertainty_24h_equivalent"] < 0.0):
            raise ValueError("V3 forecast uncertainty must be non-negative")
        for field in ("residual_rmse_24h", "residual_rmse_72h"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V3 forecast {field} must be finite and non-negative")
        for field, value in arrays.items():
            object.__setattr__(self, field, value)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 forecast digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "expected_return_24h_equivalent": (
                self.expected_return_24h_equivalent.tolist()
            ),
            "prediction_24h": self.prediction_24h.tolist(),
            "prediction_72h": self.prediction_72h.tolist(),
            "residual_rmse_24h": self.residual_rmse_24h,
            "residual_rmse_72h": self.residual_rmse_72h,
            "schema_version": CAUSAL_ALPHA_V3_FORECAST_SCHEMA,
            "signal_to_uncertainty": self.signal_to_uncertainty.tolist(),
            "uncertainty_24h_equivalent": self.uncertainty_24h_equivalent.tolist(),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def causal_alpha_v3_forecast(
    prediction_24h: object,
    prediction_72h: object,
    *,
    residual_rmse_24h: float,
    residual_rmse_72h: float,
) -> CausalAlphaV3Forecast:
    """Convert horizon predictions into one deterministic 24h-equivalent bundle."""

    first = np.asarray(prediction_24h, dtype=np.float64).reshape(-1)
    second = np.asarray(prediction_72h, dtype=np.float64).reshape(-1)
    if (
        first.shape != second.shape
        or first.size == 0
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("V3 horizon predictions must be finite and sample aligned")
    for field, value in (
        ("residual_rmse_24h", residual_rmse_24h),
        ("residual_rmse_72h", residual_rmse_72h),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")

    second_equivalent = second / 3.0
    expected = 0.5 * (first + second_equivalent)
    residual_variance = 0.25 * (
        residual_rmse_24h**2 + (residual_rmse_72h / 3.0) ** 2
    )
    disagreement = 0.5 * np.abs(first - second_equivalent)
    uncertainty = np.sqrt(residual_variance + np.square(disagreement))
    ratio = np.divide(
        expected,
        uncertainty,
        out=np.zeros_like(expected),
        where=uncertainty > _EPSILON,
    )
    return CausalAlphaV3Forecast(
        prediction_24h=first,
        prediction_72h=second,
        expected_return_24h_equivalent=expected,
        uncertainty_24h_equivalent=uncertainty,
        signal_to_uncertainty=ratio,
        residual_rmse_24h=float(residual_rmse_24h),
        residual_rmse_72h=float(residual_rmse_72h),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3TargetConfig:
    """Discrete conservative target compiler configuration."""

    target_magnitudes: tuple[float, ...]
    uncertainty_multiplier: float
    execution_cost_multiplier: float
    edge_margin: float
    alpha_rebalance_decisions: int
    strong_reversal_threshold: float
    max_target_delta: float
    schema_version: str = CAUSAL_ALPHA_V3_TARGET_SCHEMA

    def __post_init__(self) -> None:
        magnitudes = tuple(float(value) for value in self.target_magnitudes)
        if (
            not magnitudes
            or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in magnitudes)
            or tuple(sorted(set(magnitudes))) != magnitudes
            or magnitudes[0] != 0.0
        ):
            raise ValueError(
                "target_magnitudes must be unique sorted finite values in [0, 1] including zero"
            )
        if (
            not math.isfinite(self.uncertainty_multiplier)
            or self.uncertainty_multiplier < 0.0
        ):
            raise ValueError("uncertainty_multiplier must be finite and non-negative")
        if (
            not math.isfinite(self.execution_cost_multiplier)
            or self.execution_cost_multiplier <= 0.0
        ):
            raise ValueError("execution_cost_multiplier must be finite and positive")
        if not math.isfinite(self.edge_margin) or self.edge_margin < 0.0:
            raise ValueError("edge_margin must be finite and non-negative")
        if (
            isinstance(self.alpha_rebalance_decisions, bool)
            or not isinstance(self.alpha_rebalance_decisions, int)
            or self.alpha_rebalance_decisions <= 0
        ):
            raise ValueError("alpha_rebalance_decisions must be a positive integer")
        if (
            not math.isfinite(self.strong_reversal_threshold)
            or self.strong_reversal_threshold <= 0.0
        ):
            raise ValueError("strong_reversal_threshold must be finite and positive")
        if (
            not math.isfinite(self.max_target_delta)
            or not 0.0 < self.max_target_delta <= 2.0
        ):
            raise ValueError("max_target_delta must be finite and within (0, 2]")
        if self.schema_version != CAUSAL_ALPHA_V3_TARGET_SCHEMA:
            raise ValueError("unsupported causal alpha V3 target schema")
        object.__setattr__(self, "target_magnitudes", magnitudes)

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3TargetPath:
    initial_weight: float
    targets: np.ndarray
    expected_returns: np.ndarray
    uncertainties: np.ndarray
    liquidity_weight_caps: np.ndarray
    chosen_objectives: np.ndarray
    stay_objectives: np.ndarray
    reasons: tuple[str, ...]
    submitted_change_count: int
    liquidity_deleveraging_count: int
    sign_flip_count: int
    config_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("V3 initial weight must be finite")
        arrays: dict[str, np.ndarray] = {}
        shape: tuple[int, ...] | None = None
        for field in (
            "targets",
            "expected_returns",
            "uncertainties",
            "liquidity_weight_caps",
            "chosen_objectives",
            "stay_objectives",
        ):
            value = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1).copy()
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"V3 target path {field} must be finite and non-empty")
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError("V3 target path arrays must align")
            value.setflags(write=False)
            arrays[field] = value
        if np.any(arrays["uncertainties"] < 0.0) or np.any(
            arrays["liquidity_weight_caps"] < 0.0
        ):
            raise ValueError("V3 uncertainty/liquidity values must be non-negative")
        reasons = tuple(self.reasons)
        if len(reasons) != len(arrays["targets"]) or any(not item for item in reasons):
            raise ValueError("V3 target reasons must cover every decision")
        for field in (
            "submitted_change_count",
            "liquidity_deleveraging_count",
            "sign_flip_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V3 target path {field} is invalid")
        if not isinstance(self.config_digest, str) or len(self.config_digest) != 64:
            raise ValueError("V3 target path config digest is invalid")
        for field, value in arrays.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 target path digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "chosen_objectives": self.chosen_objectives.tolist(),
            "config_digest": self.config_digest,
            "expected_returns": self.expected_returns.tolist(),
            "initial_weight": self.initial_weight,
            "liquidity_deleveraging_count": self.liquidity_deleveraging_count,
            "liquidity_weight_caps": self.liquidity_weight_caps.tolist(),
            "reasons": self.reasons,
            "schema_version": CAUSAL_ALPHA_V3_TARGET_SCHEMA,
            "sign_flip_count": self.sign_flip_count,
            "stay_objectives": self.stay_objectives.tolist(),
            "submitted_change_count": self.submitted_change_count,
            "targets": self.targets.tolist(),
            "uncertainties": self.uncertainties.tolist(),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _incremental_objective(
    target: float,
    previous: float,
    expected_return: float,
    uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV3TargetConfig,
) -> float:
    delta = target - previous
    turnover = abs(delta)
    return (
        delta * expected_return
        - config.uncertainty_multiplier * turnover * uncertainty
        - turnover
        * (one_way_cost_rate * config.execution_cost_multiplier + config.edge_margin)
    )


def _candidate_targets(
    previous: float,
    cap: float,
    config: CausalAlphaV3TargetConfig,
) -> tuple[float, ...]:
    values = {0.0, float(np.clip(previous, -cap, cap)), -cap, cap}
    for magnitude in config.target_magnitudes:
        bounded = min(magnitude, cap)
        values.add(bounded)
        values.add(-bounded)
    values.add(float(np.clip(previous + config.max_target_delta, -cap, cap)))
    values.add(float(np.clip(previous - config.max_target_delta, -cap, cap)))
    return tuple(sorted(values))


def causal_alpha_v3_target_path(
    expected_returns: object,
    *,
    uncertainties: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    config: CausalAlphaV3TargetConfig,
    initial_weight: float,
) -> CausalAlphaV3TargetPath:
    """Compile forecasts into auditable targets using conservative incremental edge."""

    expected = np.asarray(expected_returns, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    costs = np.asarray(one_way_cost_rates, dtype=np.float64).reshape(-1)
    caps = np.asarray(liquidity_weight_caps, dtype=np.float64).reshape(-1)
    if (
        expected.size == 0
        or uncertainty.shape != expected.shape
        or costs.shape != expected.shape
        or caps.shape != expected.shape
        or not np.isfinite(expected).all()
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(costs).all()
        or not np.isfinite(caps).all()
    ):
        raise ValueError("V3 target compiler inputs must be finite and sample aligned")
    if np.any(uncertainty < 0.0) or np.any(costs < 0.0) or np.any(caps < 0.0):
        raise ValueError("V3 uncertainty, cost and liquidity inputs must be non-negative")
    if not isinstance(config, CausalAlphaV3TargetConfig):
        raise TypeError("V3 target compiler requires CausalAlphaV3TargetConfig")
    if not math.isfinite(initial_weight):
        raise ValueError("initial_weight must be finite")

    targets = np.empty(expected.shape, dtype=np.float64)
    chosen_objectives = np.zeros(expected.shape, dtype=np.float64)
    stay_objectives = np.zeros(expected.shape, dtype=np.float64)
    reasons: list[str] = []
    previous = float(initial_weight)
    submitted = 0
    liquidity_deleveraging = 0
    sign_flips = 0

    for index, (mu, sigma, cost, cap_value) in enumerate(
        zip(expected, uncertainty, costs, caps, strict=True)
    ):
        cap = min(float(cap_value), 1.0)
        stay = _incremental_objective(
            previous,
            previous,
            float(mu),
            float(sigma),
            float(cost),
            config,
        )
        stay_objectives[index] = stay

        if abs(previous) > cap + _EPSILON:
            selected = float(np.clip(previous, -cap, cap))
            chosen = _incremental_objective(
                selected,
                previous,
                float(mu),
                float(sigma),
                float(cost),
                config,
            )
            reason = "liquidity_deleverage"
            liquidity_deleveraging += 1
        else:
            strong_reversal = (
                abs(previous) > _EPSILON
                and previous * float(mu) < 0.0
                and abs(float(mu)) >= config.strong_reversal_threshold
            )
            on_cadence = index % config.alpha_rebalance_decisions == 0
            if not on_cadence and not strong_reversal:
                selected = previous
                chosen = stay
                reason = "cadence_hold"
            else:
                candidates = tuple(
                    value
                    for value in _candidate_targets(previous, cap, config)
                    if abs(value - previous) <= config.max_target_delta + _EPSILON
                )
                if not candidates:
                    candidates = (previous,)
                scores = tuple(
                    _incremental_objective(
                        value,
                        previous,
                        float(mu),
                        float(sigma),
                        float(cost),
                        config,
                    )
                    for value in candidates
                )
                maximum = max(scores)
                tied = tuple(
                    (value, score)
                    for value, score in zip(candidates, scores, strict=True)
                    if score >= maximum - 1e-15
                )
                selected, chosen = min(
                    tied,
                    key=lambda item: (
                        abs(item[0] - previous),
                        abs(item[0]),
                        item[0],
                    ),
                )
                if abs(selected - previous) <= _EPSILON:
                    reason = "hold"
                elif strong_reversal:
                    reason = "strong_reversal"
                else:
                    reason = "rebalance"

        if abs(selected - previous) > _EPSILON:
            submitted += 1
        if previous * selected < 0.0:
            sign_flips += 1
        targets[index] = selected
        chosen_objectives[index] = chosen
        reasons.append(reason)
        previous = float(selected)

    return CausalAlphaV3TargetPath(
        initial_weight=float(initial_weight),
        targets=targets,
        expected_returns=expected,
        uncertainties=uncertainty,
        liquidity_weight_caps=caps,
        chosen_objectives=chosen_objectives,
        stay_objectives=stay_objectives,
        reasons=tuple(reasons),
        submitted_change_count=submitted,
        liquidity_deleveraging_count=liquidity_deleveraging,
        sign_flip_count=sign_flips,
        config_digest=config.digest,
    )


__all__ = [
    "CAUSAL_ALPHA_V3_FIT_SCHEMA",
    "CAUSAL_ALPHA_V3_FORECAST_SCHEMA",
    "CAUSAL_ALPHA_V3_TARGET_SCHEMA",
    "CausalAlphaV3FitConfig",
    "CausalAlphaV3Forecast",
    "CausalAlphaV3TargetConfig",
    "CausalAlphaV3TargetPath",
    "causal_alpha_overlap_uniqueness_weights",
    "causal_alpha_v3_forecast",
    "causal_alpha_v3_target_path",
]
