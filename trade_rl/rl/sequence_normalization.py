"""Train-range-only normalization for structured native-timeframe features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.rl.sequence_observations import SequenceObservationBuilder

SEQUENCE_NORMALIZER_SCHEMA = "sequence_feature_normalizer_v1"
_EPSILON = 1e-12


def _readonly_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy(order="C")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SequenceFeatureNormalizer:
    """Robust per-channel statistics fitted only on newly available train events."""

    feature_names: Mapping[str, tuple[str, ...]]
    center: Mapping[str, np.ndarray]
    scale: Mapping[str, np.ndarray]
    train_start: int
    train_end: int
    dataset_id: str
    source_dataset_id: str
    sequence_schema_digest: str
    clip: float = 10.0
    epsilon: float = 1e-8
    schema_version: str = SEQUENCE_NORMALIZER_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        clocks = tuple(self.feature_names)
        if clocks != ("15m", "1h", "4h", "1d"):
            raise ValueError("sequence normalizer requires ordered maintained clocks")
        if tuple(self.center) != clocks or tuple(self.scale) != clocks:
            raise ValueError("sequence normalizer statistics must match feature clocks")
        resolved_center: dict[str, np.ndarray] = {}
        resolved_scale: dict[str, np.ndarray] = {}
        resolved_names: dict[str, tuple[str, ...]] = {}
        for timeframe in clocks:
            names = tuple(self.feature_names[timeframe])
            if (
                not names
                or len(set(names)) != len(names)
                or any(not name for name in names)
            ):
                raise ValueError("sequence normalizer feature names must be unique")
            center = _readonly_vector(
                self.center[timeframe], field=f"{timeframe}.center"
            )
            scale = _readonly_vector(self.scale[timeframe], field=f"{timeframe}.scale")
            if center.shape != scale.shape or center.size != len(names):
                raise ValueError("sequence normalizer channel statistics mismatch")
            if np.any(scale <= 0.0):
                raise ValueError("sequence normalizer scale must be positive")
            resolved_names[timeframe] = names
            resolved_center[timeframe] = center
            resolved_scale[timeframe] = scale
        if (
            isinstance(self.train_start, bool)
            or isinstance(self.train_end, bool)
            or not isinstance(self.train_start, int)
            or not isinstance(self.train_end, int)
            or self.train_start < 0
            or self.train_end <= self.train_start
        ):
            raise ValueError("sequence normalizer train range is invalid")
        if not math.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("sequence normalizer clip must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("sequence normalizer epsilon must be positive")
        require_sha256(self.dataset_id, field="sequence_normalizer.dataset_id")
        require_sha256(
            self.source_dataset_id, field="sequence_normalizer.source_dataset_id"
        )
        require_sha256(
            self.sequence_schema_digest,
            field="sequence_normalizer.sequence_schema_digest",
        )
        if self.schema_version != SEQUENCE_NORMALIZER_SCHEMA:
            raise ValueError("unsupported sequence normalizer schema")
        object.__setattr__(self, "feature_names", MappingProxyType(resolved_names))
        object.__setattr__(self, "center", MappingProxyType(resolved_center))
        object.__setattr__(self, "scale", MappingProxyType(resolved_scale))
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("sequence normalizer digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "center": {
                key: tuple(float(value) for value in self.center[key])
                for key in self.feature_names
            },
            "clip": self.clip,
            "dataset_id": self.dataset_id,
            "epsilon": self.epsilon,
            "feature_names": dict(self.feature_names),
            "scale": {
                key: tuple(float(value) for value in self.scale[key])
                for key in self.feature_names
            },
            "schema_version": self.schema_version,
            "sequence_schema_digest": self.sequence_schema_digest,
            "source_dataset_id": self.source_dataset_id,
            "train_end": self.train_end,
            "train_start": self.train_start,
        }

    @classmethod
    def fit(
        cls,
        dataset: MarketDataset,
        builder: SequenceObservationBuilder,
        *,
        train_start: int,
        train_end: int,
        source_dataset_id: str | None = None,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> SequenceFeatureNormalizer:
        if (
            isinstance(train_start, bool)
            or isinstance(train_end, bool)
            or not isinstance(train_start, int)
            or not isinstance(train_end, int)
            or not 0 <= train_start < train_end <= dataset.n_bars
        ):
            raise ValueError("sequence normalizer train range is outside dataset")
        payload = builder.schema_payload(dataset)
        windows = cast(tuple[dict[str, object], ...], payload["windows"])
        feature_names: dict[str, tuple[str, ...]] = {}
        centers: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        ages = dataset.resolved_array("feature_staleness_hours")
        for raw_window in windows:
            window = dict(raw_window)
            timeframe = str(window["timeframe"])
            raw_names = window["feature_names"]
            if not isinstance(raw_names, (tuple, list)):
                raise ValueError("sequence normalizer feature names must be ordered")
            names = tuple(str(name) for name in raw_names)
            columns = np.asarray(
                [dataset.feature_names.index(name) for name in names], dtype=np.int64
            )
            values = dataset.features[train_start:train_end, :, columns]
            available = dataset.feature_available[train_start:train_end, :, columns]
            current_age = ages[train_start:train_end, :, columns]
            previous_available = np.zeros_like(available)
            previous_age = np.full_like(current_age, np.inf, dtype=np.float64)
            if train_start > 0:
                previous_available[0] = dataset.feature_available[train_start - 1][
                    :, columns
                ]
                previous_age[0] = ages[train_start - 1][:, columns]
            if len(available) > 1:
                previous_available[1:] = available[:-1]
                previous_age[1:] = current_age[:-1]
            new_event = available & (
                ~previous_available
                | (current_age <= _EPSILON)
                | (current_age < previous_age - _EPSILON)
            )
            center = np.zeros(len(names), dtype=np.float64)
            scale = np.ones(len(names), dtype=np.float64)
            for feature_index in range(len(names)):
                sample = np.asarray(
                    values[:, :, feature_index][new_event[:, :, feature_index]],
                    dtype=np.float64,
                )
                sample = sample[np.isfinite(sample)]
                if sample.size == 0:
                    continue
                median = float(np.median(sample))
                q25, q75 = np.quantile(sample, (0.25, 0.75))
                robust_scale = float((q75 - q25) / 1.349)
                if robust_scale <= epsilon:
                    robust_scale = float(np.std(sample))
                center[feature_index] = median
                scale[feature_index] = robust_scale if robust_scale > epsilon else 1.0
            feature_names[timeframe] = names
            centers[timeframe] = center
            scales[timeframe] = scale
        return cls(
            feature_names=feature_names,
            center=centers,
            scale=scales,
            train_start=train_start,
            train_end=train_end,
            dataset_id=dataset.dataset_id,
            source_dataset_id=source_dataset_id or dataset.dataset_id,
            sequence_schema_digest=builder.layout_digest(dataset),
            clip=clip,
            epsilon=epsilon,
        )

    def transform(
        self,
        timeframe: str,
        values: np.ndarray,
        available: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        if timeframe not in self.feature_names:
            raise ValueError("sequence normalizer timeframe is unknown")
        if tuple(feature_names) != self.feature_names[timeframe]:
            raise ValueError("sequence feature order does not match normalizer")
        array = np.asarray(values, dtype=np.float64)
        mask = np.asarray(available, dtype=np.bool_)
        if array.shape != mask.shape or array.shape[-1] != len(feature_names):
            raise ValueError("sequence values and availability shapes differ")
        if not np.isfinite(array).all():
            raise ValueError("sequence values must be finite")
        result = np.clip(
            (array - self.center[timeframe]) / self.scale[timeframe],
            -self.clip,
            self.clip,
        )
        result = np.where(mask, result, 0.0)
        return np.asarray(result, dtype=np.float32)


__all__ = ["SEQUENCE_NORMALIZER_SCHEMA", "SequenceFeatureNormalizer"]
