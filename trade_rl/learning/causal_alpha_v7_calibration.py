"""Purged shared fast calibration for Causal Alpha V7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    fit_causal_alpha_ridge,
)
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
)

_ROWS_SCHEMA: Final = "causal_alpha_v7_calibration_rows_v1"
_FIT_SCHEMA: Final = "causal_alpha_v7_calibration_fit_v1"


def _readonly_array(value: object, *, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    if array.size == 0:
        raise ValueError(f"{field} must be non-empty")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CalibrationRows:
    """One symbol's preselected causal calibration rows."""

    symbol: str
    decision_indices: np.ndarray
    label_end_indices: np.ndarray
    features: np.ndarray
    feature_available: np.ndarray
    realized_returns: np.ndarray
    range_digest: str
    schema_version: str = _ROWS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("V7 calibration rows symbol must be non-empty")
        require_sha256(self.range_digest, field="V7 calibration rows range digest")
        decisions = _readonly_array(
            self.decision_indices,
            dtype=np.int64,
            field="V7 calibration decisions",
        ).reshape(-1)
        ends = _readonly_array(
            self.label_end_indices,
            dtype=np.int64,
            field="V7 calibration label ends",
        ).reshape(-1)
        realized = _readonly_array(
            self.realized_returns,
            dtype=np.float64,
            field="V7 calibration realized returns",
        ).reshape(-1)
        features = _readonly_array(
            self.features,
            dtype=np.float64,
            field="V7 calibration features",
        )
        available = _readonly_array(
            self.feature_available,
            dtype=np.bool_,
            field="V7 calibration feature availability",
        )
        rows = decisions.size
        if (
            ends.shape != (rows,)
            or realized.shape != (rows,)
            or features.shape != (rows, len(CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES))
            or available.shape != features.shape
        ):
            raise ValueError("V7 calibration rows are not aligned")
        if np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V7 calibration decisions must be strictly increasing")
        if np.any(ends < decisions):
            raise ValueError("V7 calibration label ends precede decisions")
        if self.schema_version != _ROWS_SCHEMA:
            raise ValueError("unsupported V7 calibration rows schema")
        for name, array in (
            ("decision_indices", decisions),
            ("label_end_indices", ends),
            ("features", features),
            ("feature_available", available),
            ("realized_returns", realized),
        ):
            object.__setattr__(self, name, array)
        expected = content_and_arrays_digest(
            {
                "range_digest": self.range_digest,
                "schema_version": self.schema_version,
                "symbol": self.symbol,
            },
            (
                ("decision_indices", decisions),
                ("label_end_indices", ends),
                ("features", features),
                ("feature_available", available),
                ("realized_returns", realized),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V7 calibration rows digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CalibrationFit:
    """Two shared ridge heads for calibrated gross return and direction."""

    calibration_range: CausalAlphaV7CalibrationRange
    config: CausalAlphaV7CalibrationConfig
    return_model: CausalAlphaRidgeModel
    direction_model: CausalAlphaRidgeModel
    per_symbol_support: tuple[tuple[str, int], ...]
    positive_direction_support: int
    negative_direction_support: int
    rows_digests: tuple[tuple[str, str], ...]
    schema_version: str = _FIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_range, CausalAlphaV7CalibrationRange):
            raise TypeError("V7 calibration fit range is invalid")
        if not isinstance(self.config, CausalAlphaV7CalibrationConfig):
            raise TypeError("V7 calibration fit config is invalid")
        for model in (self.return_model, self.direction_model):
            if not isinstance(model, CausalAlphaRidgeModel):
                raise TypeError("V7 calibration fit models are invalid")
            if model.feature_names != CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES:
                raise ValueError("V7 calibration model feature schema drifted")
            if model.knowledge_cutoff != self.calibration_range.train_stop:
                raise ValueError("V7 calibration model cutoff drifted")
        support = tuple(self.per_symbol_support)
        digests = tuple(self.rows_digests)
        if (
            not support
            or support != tuple(sorted(support))
            or digests != tuple(sorted(digests))
            or tuple(symbol for symbol, _ in support)
            != tuple(symbol for symbol, _ in digests)
        ):
            raise ValueError("V7 calibration symbol evidence is invalid")
        if any(count < self.config.minimum_symbol_support for _, count in support):
            raise ValueError("V7 calibration symbol support is insufficient")
        pooled = sum(count for _, count in support)
        if pooled < self.config.minimum_pooled_support:
            raise ValueError("V7 calibration pooled support is insufficient")
        if self.return_model.sample_count != pooled or self.direction_model.sample_count != pooled:
            raise ValueError("V7 calibration model support drifted")
        minimum_direction = self.config.minimum_symbol_support
        if (
            self.positive_direction_support < minimum_direction
            or self.negative_direction_support < minimum_direction
            or self.positive_direction_support + self.negative_direction_support > pooled
        ):
            raise ValueError("V7 calibration requires both realized directions")
        for _symbol, digest in digests:
            require_sha256(digest, field="V7 calibration rows digest")
        if self.schema_version != _FIT_SCHEMA:
            raise ValueError("unsupported V7 calibration fit schema")
        object.__setattr__(self, "per_symbol_support", support)
        object.__setattr__(self, "rows_digests", digests)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 calibration fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def pooled_support(self) -> int:
        return sum(count for _, count in self.per_symbol_support)

    def predict(
        self,
        features: object,
        *,
        feature_available: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float64)
        available = np.asarray(feature_available, dtype=np.bool_)
        if (
            values.ndim != 2
            or values.shape[1] != len(CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES)
            or available.shape != values.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError("V7 calibration prediction inputs are invalid")
        calibrated_return = np.empty(values.shape[0], dtype=np.float64)
        reliability = np.empty(values.shape[0], dtype=np.float64)
        block_rows = self.config.working_memory_rows
        for start in range(0, values.shape[0], block_rows):
            stop = min(start + block_rows, values.shape[0])
            block = slice(start, stop)
            calibrated_return[block] = self.return_model.predict(
                values[block], feature_available=available[block]
            )
            reliability[block] = self.direction_model.predict(
                values[block], feature_available=available[block]
            )
        return calibrated_return, reliability

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config.digest,
            "direction_model_digest": self.direction_model.digest,
            "negative_direction_support": self.negative_direction_support,
            "per_symbol_support": self.per_symbol_support,
            "pooled_support": self.pooled_support,
            "positive_direction_support": self.positive_direction_support,
            "range_digest": self.calibration_range.digest,
            "return_model_digest": self.return_model.digest,
            "rows_digests": self.rows_digests,
            "schema_version": self.schema_version,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def fit_causal_alpha_v7_calibration(
    *,
    rows: Mapping[str, CausalAlphaV7CalibrationRows],
    calibration_range: CausalAlphaV7CalibrationRange,
    config: CausalAlphaV7CalibrationConfig,
) -> CausalAlphaV7CalibrationFit:
    """Fit deterministic shared V7 calibration heads on causal rows."""

    if not isinstance(calibration_range, CausalAlphaV7CalibrationRange):
        raise TypeError("V7 calibration range is invalid")
    if not isinstance(config, CausalAlphaV7CalibrationConfig):
        raise TypeError("V7 calibration config is invalid")
    ordered = tuple(sorted(rows))
    if not ordered or set(ordered) != set(rows):
        raise ValueError("V7 calibration rows mapping is invalid")
    records: list[CausalAlphaV7CalibrationRows] = []
    weights: list[np.ndarray] = []
    for symbol in ordered:
        record = rows[symbol]
        if not isinstance(record, CausalAlphaV7CalibrationRows) or record.symbol != symbol:
            raise ValueError("V7 calibration row identity drifted")
        if record.range_digest != calibration_range.digest:
            raise ValueError("V7 calibration range digest drifted")
        if (
            np.any(record.decision_indices < calibration_range.calibration_start)
            or np.any(record.decision_indices >= calibration_range.train_stop)
        ):
            raise ValueError("V7 calibration decisions are outside the causal range")
        if np.any(record.label_end_indices >= calibration_range.train_stop):
            raise ValueError("V7 calibration labels must end strictly before train stop")
        record_weights = causal_alpha_overlap_uniqueness_weights(
            record.decision_indices,
            record.label_end_indices,
            knowledge_cutoff=calibration_range.train_stop,
        )
        if int(np.count_nonzero(record_weights)) < config.minimum_symbol_support:
            raise ValueError("V7 calibration symbol support is insufficient")
        records.append(record)
        weights.append(record_weights)
    pooled_features = np.concatenate(tuple(record.features for record in records))
    pooled_available = np.concatenate(tuple(record.feature_available for record in records))
    pooled_realized = np.concatenate(tuple(record.realized_returns for record in records))
    pooled_ends = np.concatenate(tuple(record.label_end_indices for record in records))
    pooled_weights = np.concatenate(tuple(weights))
    positive = int(np.count_nonzero((pooled_realized > 0.0) & (pooled_weights > 0.0)))
    negative = int(np.count_nonzero((pooled_realized < 0.0) & (pooled_weights > 0.0)))
    if positive < config.minimum_symbol_support or negative < config.minimum_symbol_support:
        raise ValueError("V7 calibration requires both realized directions")
    ridge = CausalAlphaRidgeConfig(ridge_strength=config.ridge_strength)
    def fit_head(labels: np.ndarray) -> CausalAlphaRidgeModel:
        return fit_causal_alpha_ridge(
            features=pooled_features,
            labels=labels,
            feature_available=pooled_available,
            label_end_indices=pooled_ends,
            knowledge_cutoff=calibration_range.train_stop,
            feature_names=CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
            config=ridge,
            sample_weights=pooled_weights,
            normalize_objective=True,
            working_memory_rows=config.working_memory_rows,
        )

    return_model = fit_head(pooled_realized)
    direction_model = fit_head(np.sign(pooled_realized))
    return CausalAlphaV7CalibrationFit(
        calibration_range=calibration_range,
        config=config,
        return_model=return_model,
        direction_model=direction_model,
        per_symbol_support=tuple(
            (record.symbol, int(np.count_nonzero(weight)))
            for record, weight in zip(records, weights, strict=True)
        ),
        positive_direction_support=positive,
        negative_direction_support=negative,
        rows_digests=tuple((record.symbol, record.digest) for record in records),
    )


__all__ = [
    "CausalAlphaV7CalibrationFit",
    "CausalAlphaV7CalibrationRows",
    "fit_causal_alpha_v7_calibration",
]
