"""Deterministic pooled dual-horizon fitting for Causal Alpha V10."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config

_ROWS_SCHEMA: Final = "causal_alpha_v10_training_rows_v1"
_HORIZON_FIT_SCHEMA: Final = "causal_alpha_v10_horizon_fit_v1"
_DUAL_FIT_SCHEMA: Final = "causal_alpha_v10_dual_fit_v1"


def _readonly(value: object, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalAlphaV10TrainingRows:
    symbol: str
    decision_indices: np.ndarray
    fast_label_end_indices: np.ndarray
    slow_label_end_indices: np.ndarray
    features: np.ndarray
    feature_available: np.ndarray
    fast_labels: np.ndarray
    slow_labels: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str = _ROWS_SCHEMA

    def __post_init__(self) -> None:
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        fast_ends = _readonly(self.fast_label_end_indices, dtype=np.int64).reshape(-1)
        slow_ends = _readonly(self.slow_label_end_indices, dtype=np.int64).reshape(-1)
        features = _readonly(self.features, dtype=np.float64)
        available = _readonly(self.feature_available, dtype=np.bool_)
        fast_labels = _readonly(self.fast_labels, dtype=np.float64).reshape(-1)
        slow_labels = _readonly(self.slow_labels, dtype=np.float64).reshape(-1)
        names = tuple(self.feature_names)
        count = len(decisions)
        if not self.symbol or count == 0:
            raise ValueError("V10 training row identity is invalid")
        if any("symbol" in name.lower() for name in names):
            raise ValueError("V10 cannot use a symbol identity feature")
        if not names or len(set(names)) != len(names):
            raise ValueError("V10 feature names are invalid")
        if (
            features.shape != (count, len(names))
            or available.shape != features.shape
            or fast_ends.shape != (count,)
            or slow_ends.shape != (count,)
            or fast_labels.shape != (count,)
            or slow_labels.shape != (count,)
        ):
            raise ValueError("V10 training arrays are not aligned")
        if not np.isfinite(features).all() or np.any(np.diff(decisions) <= 0):
            raise ValueError("V10 training arrays are invalid")
        for ends, labels in ((fast_ends, fast_labels), (slow_ends, slow_labels)):
            valid = ends >= 0
            if (
                np.any(valid & ~np.isfinite(labels))
                or np.any(~valid & np.isfinite(labels))
                or np.any(valid & (ends <= decisions))
            ):
                raise ValueError("V10 training labels are invalid")
        if self.schema_version != _ROWS_SCHEMA:
            raise ValueError("unsupported V10 training row schema")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "fast_label_end_indices", fast_ends)
        object.__setattr__(self, "slow_label_end_indices", slow_ends)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_available", available)
        object.__setattr__(self, "fast_labels", fast_labels)
        object.__setattr__(self, "slow_labels", slow_labels)
        object.__setattr__(self, "feature_names", names)


@dataclass(frozen=True, slots=True)
class CausalAlphaV10HorizonFit:
    horizon: Literal["fast_4h", "slow_72h"]
    design_mode: Literal["raw_plus_relu", "relu_only"]
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
    schema_version: str = _HORIZON_FIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        mean = _readonly(self.feature_mean, dtype=np.float64).reshape(-1)
        scale = _readonly(self.feature_scale, dtype=np.float64).reshape(-1)
        weights = _readonly(self.hidden_weights, dtype=np.float64)
        bias = _readonly(self.hidden_bias, dtype=np.float64)
        coefficients = _readonly(self.coefficients, dtype=np.float64)
        width = len(self.feature_names)
        if self.horizon not in ("fast_4h", "slow_72h"):
            raise ValueError("V10 fit horizon is invalid")
        expected_mode = "raw_plus_relu" if self.horizon == "fast_4h" else "relu_only"
        if self.design_mode != expected_mode:
            raise ValueError("V10 fit design mode is invalid")
        if self.maximum_label_end_index >= self.knowledge_cutoff or self.training_row_count <= 0:
            raise ValueError("V10 fit causal range is invalid")
        if mean.shape != (width,) or scale.shape != (width,) or np.any(scale <= 0.0):
            raise ValueError("V10 fit normalization is invalid")
        if weights.ndim != 3 or weights.shape[:2] != (3, width):
            raise ValueError("V10 fit hidden weights are invalid")
        if bias.shape != (3, weights.shape[2]):
            raise ValueError("V10 fit hidden bias is invalid")
        design_width = (
            width + weights.shape[2]
            if self.design_mode == "raw_plus_relu"
            else weights.shape[2]
        )
        if coefficients.shape != (3, design_width):
            raise ValueError("V10 fit coefficients are invalid")
        if self.schema_version != _HORIZON_FIT_SCHEMA:
            raise ValueError("unsupported V10 horizon fit schema")
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
            raise ValueError("V10 fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    def predict_heads(self, features: object) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("V10 prediction features are invalid")
        if not np.isfinite(values).all():
            raise ValueError("V10 prediction features must be finite")
        normalized = (values - self.feature_mean) / self.feature_scale
        result = np.empty((3, len(values)), dtype=np.float64)
        for head in range(3):
            hidden = np.maximum(
                0.0,
                normalized @ self.hidden_weights[head] + self.hidden_bias[head],
            )
            design = (
                np.column_stack((normalized, hidden))
                if self.design_mode == "raw_plus_relu"
                else hidden
            )
            result[head] = design @ self.coefficients[head]
        result.setflags(write=False)
        return result

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "design_mode": self.design_mode,
            "feature_names": self.feature_names,
            "horizon": self.horizon,
            "knowledge_cutoff": self.knowledge_cutoff,
            "maximum_label_end_index": self.maximum_label_end_index,
            "schema_version": self.schema_version,
            "training_row_count": self.training_row_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV10DualFit:
    fast: CausalAlphaV10HorizonFit
    slow: CausalAlphaV10HorizonFit
    config_digest: str
    schema_version: str = _DUAL_FIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.fast.horizon != "fast_4h" or self.slow.horizon != "slow_72h":
            raise ValueError("V10 dual fit horizons are invalid")
        if self.fast.knowledge_cutoff != self.slow.knowledge_cutoff:
            raise ValueError("V10 dual fit cutoffs differ")
        if self.fast.config_digest != self.config_digest or self.slow.config_digest != self.config_digest:
            raise ValueError("V10 dual fit config identity drifted")
        if self.schema_version != _DUAL_FIT_SCHEMA:
            raise ValueError("unsupported V10 dual fit schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 dual fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "fast_fit_digest": self.fast.digest,
            "schema_version": self.schema_version,
            "slow_fit_digest": self.slow.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _fit_horizon(
    records: Mapping[str, CausalAlphaV10TrainingRows],
    *,
    knowledge_cutoff: int,
    config: CausalAlphaV10Config,
    horizon: Literal["fast_4h", "slow_72h"],
) -> CausalAlphaV10HorizonFit:
    fast = horizon == "fast_4h"
    lookback = config.fast_lookback_decisions if fast else config.slow_lookback_decisions
    horizon_decisions = config.fast_horizon_decisions if fast else config.slow_horizon_decisions
    start = knowledge_cutoff - lookback
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    end_blocks: list[np.ndarray] = []
    for record in records.values():
        ends = record.fast_label_end_indices if fast else record.slow_label_end_indices
        labels = record.fast_labels if fast else record.slow_labels
        selected = (
            (record.decision_indices >= start)
            & (ends >= 0)
            & (ends < knowledge_cutoff)
            & ((knowledge_cutoff - record.decision_indices) % horizon_decisions == 0)
            & np.all(record.feature_available, axis=1)
        )
        if not np.any(selected):
            raise ValueError(f"V10 {record.symbol} has no causal {horizon} rows")
        feature_blocks.append(record.features[selected])
        label_blocks.append(labels[selected])
        end_blocks.append(ends[selected])
    features = np.concatenate(feature_blocks)
    labels = np.concatenate(label_blocks)
    ends = np.concatenate(end_blocks)
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, dtype=np.float64)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (features - mean) / scale
    hidden_feature_count = (
        config.hidden_feature_count if fast else config.slow_hidden_feature_count
    )
    if not fast and len(features) < 2 * hidden_feature_count:
        raise ValueError("V10 slow fit requires two rows per hidden coefficient")
    weights = np.stack(
        tuple(
            np.random.default_rng(seed).normal(
                size=(features.shape[1], hidden_feature_count)
            )
            / np.sqrt(features.shape[1])
            for seed in config.head_seeds
        )
    )
    bias = np.stack(
        tuple(
            np.random.default_rng(seed).normal(size=hidden_feature_count)
            for seed in config.bias_seeds
        )
    )
    coefficients: list[np.ndarray] = []
    for head in range(3):
        hidden = np.maximum(0.0, normalized @ weights[head] + bias[head])
        design = np.column_stack((normalized, hidden)) if fast else hidden
        gram = design.T @ design
        coefficients.append(
            np.linalg.solve(
                gram + config.ridge_strength * np.eye(gram.shape[0]),
                design.T @ labels,
            )
        )
    return CausalAlphaV10HorizonFit(
        horizon=horizon,
        design_mode="raw_plus_relu" if fast else "relu_only",
        knowledge_cutoff=knowledge_cutoff,
        maximum_label_end_index=int(np.max(ends)),
        feature_names=next(iter(records.values())).feature_names,
        feature_mean=mean,
        feature_scale=scale,
        hidden_weights=weights,
        hidden_bias=bias,
        coefficients=np.stack(coefficients),
        training_row_count=len(features),
        config_digest=config.digest,
    )


def fit_causal_alpha_v10(
    rows: Mapping[str, CausalAlphaV10TrainingRows],
    *,
    knowledge_cutoff: int,
    config: CausalAlphaV10Config,
) -> CausalAlphaV10DualFit:
    """Fit deterministic fast and slow heads using only causal non-overlapping rows."""

    records = dict(rows)
    if not records or tuple(sorted(records)) != tuple(
        sorted(record.symbol for record in records.values())
    ):
        raise ValueError("V10 training row symbols are invalid")
    names = next(iter(records.values())).feature_names
    if any(record.feature_names != names for record in records.values()):
        raise ValueError("V10 feature schema drifted across symbols")
    fast = _fit_horizon(
        records,
        knowledge_cutoff=knowledge_cutoff,
        config=config,
        horizon="fast_4h",
    )
    slow = _fit_horizon(
        records,
        knowledge_cutoff=knowledge_cutoff,
        config=config,
        horizon="slow_72h",
    )
    return CausalAlphaV10DualFit(fast=fast, slow=slow, config_digest=config.digest)


__all__ = [
    "CausalAlphaV10DualFit",
    "CausalAlphaV10HorizonFit",
    "CausalAlphaV10TrainingRows",
    "fit_causal_alpha_v10",
]
