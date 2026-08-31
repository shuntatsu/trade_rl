"""Deterministic pooled nonlinear wave fitting for Causal Alpha V9."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config

_ROWS_SCHEMA: Final = "causal_alpha_v9_training_rows_v1"
_FIT_SCHEMA: Final = "causal_alpha_v9_wave_fit_v1"


def _readonly(value: object, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalAlphaV9TrainingRows:
    symbol: str
    decision_indices: np.ndarray
    label_end_indices: np.ndarray
    features: np.ndarray
    feature_available: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str = _ROWS_SCHEMA

    def __post_init__(self) -> None:
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        ends = _readonly(self.label_end_indices, dtype=np.int64).reshape(-1)
        features = _readonly(self.features, dtype=np.float64)
        available = _readonly(self.feature_available, dtype=np.bool_)
        labels = _readonly(self.labels, dtype=np.float64).reshape(-1)
        names = tuple(self.feature_names)
        rows = len(decisions)
        if not self.symbol or rows == 0:
            raise ValueError("V9 training row identity is invalid")
        if any("symbol" in name.lower() for name in names):
            raise ValueError("V9 cannot use a symbol identity feature")
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("V9 feature names are invalid")
        if (
            features.shape != (rows, len(names))
            or available.shape != features.shape
            or ends.shape != (rows,)
            or labels.shape != (rows,)
        ):
            raise ValueError("V9 training arrays are not aligned")
        valid_labels = ends >= 0
        if (
            not np.isfinite(features).all()
            or np.any(valid_labels & ~np.isfinite(labels))
            or np.any(~valid_labels & np.isfinite(labels))
            or np.any(valid_labels & (ends <= decisions))
            or np.any(np.diff(decisions) <= 0)
        ):
            raise ValueError("V9 training arrays are invalid")
        if self.schema_version != _ROWS_SCHEMA:
            raise ValueError("unsupported V9 training row schema")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "label_end_indices", ends)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_available", available)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "feature_names", names)


@dataclass(frozen=True, slots=True)
class CausalAlphaV9WaveFit:
    knowledge_cutoff: int
    maximum_label_end_index: int
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    coefficients: np.ndarray
    training_row_count: int
    config_digest: str
    schema_version: str = _FIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        mean = _readonly(self.feature_mean, dtype=np.float64).reshape(-1)
        scale = _readonly(self.feature_scale, dtype=np.float64).reshape(-1)
        weights = _readonly(self.hidden_weights, dtype=np.float64)
        bias = _readonly(self.hidden_bias, dtype=np.float64)
        coefficients = _readonly(self.coefficients, dtype=np.float64)
        width = len(self.feature_names)
        if (
            self.maximum_label_end_index >= self.knowledge_cutoff
            or self.training_row_count <= 0
        ):
            raise ValueError("V9 fit causal range is invalid")
        if mean.shape != (width,) or scale.shape != (width,) or np.any(scale <= 0.0):
            raise ValueError("V9 fit normalization is invalid")
        if weights.ndim != 3 or weights.shape[0] != 3 or weights.shape[1] != width:
            raise ValueError("V9 fit hidden weights are invalid")
        if bias.shape != (3, weights.shape[2]):
            raise ValueError("V9 fit hidden bias is invalid")
        if coefficients.shape != (3, width + weights.shape[2]):
            raise ValueError("V9 fit coefficients are invalid")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "hidden_weights", weights)
        object.__setattr__(self, "hidden_bias", bias)
        object.__setattr__(self, "coefficients", coefficients)
        expected = content_and_arrays_digest(
            self.to_payload(include_digest=False),
            (
                ("feature_mean", mean),
                ("feature_scale", scale),
                ("hidden_weights", weights),
                ("hidden_bias", bias),
                ("coefficients", coefficients),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V9 fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    def predict_heads(self, features: object) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("V9 prediction features are invalid")
        if not np.isfinite(values).all():
            raise ValueError("V9 prediction features must be finite")
        normalized = (values - self.feature_mean) / self.feature_scale
        result = np.empty((3, len(values)), dtype=np.float64)
        for head in range(3):
            hidden = np.maximum(
                0.0,
                normalized @ self.hidden_weights[head] + self.hidden_bias[head],
            )
            design = np.column_stack((normalized, hidden))
            result[head] = design @ self.coefficients[head]
        result.setflags(write=False)
        return result

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "feature_names": self.feature_names,
            "knowledge_cutoff": self.knowledge_cutoff,
            "maximum_label_end_index": self.maximum_label_end_index,
            "schema_version": self.schema_version,
            "training_row_count": self.training_row_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def fit_causal_alpha_v9_wave(
    rows: Mapping[str, CausalAlphaV9TrainingRows],
    *,
    knowledge_cutoff: int,
    config: CausalAlphaV9Config,
) -> CausalAlphaV9WaveFit:
    """Fit three deterministic random-ReLU ridge heads on one causal window."""

    records = dict(rows)
    if not records or tuple(sorted(records)) != tuple(
        sorted(record.symbol for record in records.values())
    ):
        raise ValueError("V9 training row symbols are invalid")
    names = next(iter(records.values())).feature_names
    if any(record.feature_names != names for record in records.values()):
        raise ValueError("V9 feature schema drifted across symbols")
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    end_blocks: list[np.ndarray] = []
    start = knowledge_cutoff - config.lookback_decisions
    for record in records.values():
        selected = (
            (record.decision_indices >= start)
            & (record.label_end_indices >= 0)
            & (record.label_end_indices < knowledge_cutoff)
            & (
                (knowledge_cutoff - record.decision_indices) % config.horizon_decisions
                == 0
            )
            & np.all(record.feature_available, axis=1)
        )
        if not np.any(selected):
            raise ValueError(f"V9 {record.symbol} has no causal training rows")
        feature_blocks.append(record.features[selected])
        label_blocks.append(record.labels[selected])
        end_blocks.append(record.label_end_indices[selected])
    features = np.concatenate(feature_blocks)
    labels = np.concatenate(label_blocks)
    ends = np.concatenate(end_blocks)
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, dtype=np.float64)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (features - mean) / scale
    weights = np.stack(
        tuple(
            np.random.default_rng(seed).normal(
                size=(features.shape[1], config.hidden_feature_count)
            )
            / np.sqrt(features.shape[1])
            for seed in config.head_seeds
        )
    )
    bias = np.stack(
        tuple(
            np.random.default_rng(seed).normal(size=config.hidden_feature_count)
            for seed in config.bias_seeds
        )
    )
    coefficients: list[np.ndarray] = []
    for head in range(3):
        hidden = np.maximum(0.0, normalized @ weights[head] + bias[head])
        design = np.column_stack((normalized, hidden))
        gram = design.T @ design
        coefficients.append(
            np.linalg.solve(
                gram + config.ridge_strength * np.eye(gram.shape[0]),
                design.T @ labels,
            )
        )
    return CausalAlphaV9WaveFit(
        knowledge_cutoff=knowledge_cutoff,
        maximum_label_end_index=int(np.max(ends)),
        feature_names=names,
        feature_mean=mean,
        feature_scale=scale,
        hidden_weights=weights,
        hidden_bias=bias,
        coefficients=np.stack(coefficients),
        training_row_count=len(features),
        config_digest=config.digest,
    )


def _aligned(value: object, *, rows: int, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V9 {field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"V9 {field} must be finite")
    return array


def causal_alpha_v9_wave_target_path(
    *,
    decision_indices: object,
    head_predictions: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object,
    actionable_mask: object,
    source_forecast_digest: str,
    config: CausalAlphaV9Config,
    initial_weight: float,
) -> CausalAlphaV6TargetPath:
    """Compile one confirmed wave without neutral-signal churn or direct flips."""

    require_sha256(source_forecast_digest, field="V9 source forecast digest")
    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    rows = len(decisions)
    heads = np.asarray(head_predictions, dtype=np.float64)
    if heads.shape != (3, rows) or not np.isfinite(heads).all():
        raise ValueError("V9 head predictions are invalid")
    costs = _aligned(one_way_cost_rates, rows=rows, dtype=np.float64, field="costs")
    liquidity = _aligned(
        liquidity_weight_caps,
        rows=rows,
        dtype=np.float64,
        field="liquidity caps",
    )
    risk = _aligned(risk_weight_caps, rows=rows, dtype=np.float64, field="risk caps")
    actionable = _aligned(
        actionable_mask,
        rows=rows,
        dtype=np.bool_,
        field="actionable mask",
    )
    if np.any(costs < 0.0) or np.any(liquidity < 0.0) or np.any(risk < 0.0):
        raise ValueError("V9 costs and caps must be non-negative")
    mean = np.mean(heads, axis=0, dtype=np.float64)
    uncertainty = np.std(heads, axis=0, dtype=np.float64)
    signs = np.sign(heads)
    agreed = np.all(signs == signs[0], axis=0) & (signs[0] != 0.0)
    qualified = np.where(
        agreed & (np.abs(mean) > uncertainty + config.edge_margin),
        np.sign(mean),
        0.0,
    ).astype(np.int8)
    targets = np.empty(rows, dtype=np.float64)
    reasons: list[str] = []
    confirmation_counts = np.zeros(rows, dtype=np.int64)
    objectives = np.zeros(rows, dtype=np.float64)
    current = float(
        np.clip(initial_weight, -config.target_magnitude, config.target_magnitude)
    )
    inherited = abs(initial_weight) > 1e-12
    intent = 0
    count = 0
    for index in range(rows):
        cap = min(
            config.target_magnitude,
            float(liquidity[index]),
            float(risk[index]),
        )
        if abs(current) > cap:
            current = float(np.sign(current) * cap)
        reason = "hold_position" if abs(current) > 1e-12 else "hold_flat"
        if index % config.horizon_decisions == 0 and bool(actionable[index]):
            signal = int(qualified[index])
            current_sign = int(np.sign(current))
            if inherited and current_sign != 0:
                wanted = current_sign if signal == current_sign else 0
                if wanted == intent:
                    count += 1
                else:
                    intent, count = wanted, 1
                if count >= config.confirmation_count:
                    if wanted == 0:
                        current = 0.0
                        reason = "exit"
                    else:
                        reason = "hold_position"
                    inherited = False
                    intent, count = 0, 0
                else:
                    reason = "confirmation_hold"
            elif current_sign == 0:
                if signal == 0:
                    intent, count = 0, 0
                    reason = "hold_flat"
                else:
                    if signal == intent:
                        count += 1
                    else:
                        intent, count = signal, 1
                    if count >= config.confirmation_count:
                        current = float(signal * cap)
                        reason = "entry"
                        intent, count = 0, 0
                    else:
                        reason = "confirmation_hold"
            elif signal == -current_sign:
                if signal == intent:
                    count += 1
                else:
                    intent, count = signal, 1
                if count >= config.confirmation_count:
                    current = 0.0
                    reason = "exit"
                    intent, count = 0, 0
                else:
                    reason = "confirmation_hold"
            else:
                intent, count = 0, 0
                reason = "hold_position"
        elif index % config.horizon_decisions == 0:
            reason = "unactionable_hold"
        targets[index] = current
        reasons.append(reason)
        confirmation_counts[index] = count
        objectives[index] = current * mean[index] - abs(current) * (
            uncertainty[index] + config.edge_margin
        )
    previous = np.concatenate(([initial_weight], targets[:-1]))
    forecast_digest = content_and_arrays_digest(
        {
            "schema_version": "causal_alpha_v9_wave_forecast_v1",
            "source_forecast_digest": source_forecast_digest,
        },
        (("head_predictions", heads),),
    )
    reason_counts = tuple(
        sorted((reason, reasons.count(reason)) for reason in set(reasons))
    )
    return CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=float(initial_weight),
        decision_indices=decisions,
        targets=targets,
        fast_proposals=targets,
        expected_returns_4h=mean,
        expected_returns_24h=np.zeros(rows),
        expected_returns_72h=np.zeros(rows),
        direction_scores_4h=qualified.astype(np.float64),
        uncertainties_4h=uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=liquidity,
        risk_weight_caps=risk,
        objectives=objectives,
        confirmation_counts=confirmation_counts,
        actionable_mask=actionable,
        slow_states=tuple(CausalAlphaV6SlowState.MIXED for _ in range(rows)),
        reasons=tuple(reasons),
        reason_counts=reason_counts,
        submitted_change_count=int(
            np.count_nonzero(np.abs(targets - previous) > 1e-12)
        ),
        sign_flip_count=int(np.count_nonzero(targets * previous < 0.0)),
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        forecast_digest=forecast_digest,
        config_digest=config.digest,
    )


__all__ = [
    "CausalAlphaV9TrainingRows",
    "CausalAlphaV9WaveFit",
    "causal_alpha_v9_wave_target_path",
    "fit_causal_alpha_v9_wave",
]
