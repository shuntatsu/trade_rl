"""Frozen, fit-scope-only preprocessing for optional PPO local feature values."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.dataset_scope import (
    validated_feature_indices,
    validated_symbol_indices,
)

SCHEMA = "ppo_feature_standardization_v1"


def _indices(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in result
        )
    ):
        raise ValueError(f"{name} must be nonempty unique nonnegative integers")
    return result


def _window(start: int, stop: int) -> None:
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, stop)
        )
        or not 0 <= start < stop
    ):
        raise ValueError("training scope must be a nonempty integer window")


@dataclass(frozen=True, slots=True)
class PPOFeatureNormalizer:
    source_dataset_id: str
    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    fit_symbol_indices: tuple[int, ...]
    start_index: int
    stop_index: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    usable_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        require_sha256(self.source_dataset_id, field="source_dataset_id")
        indices = _indices(self.feature_indices, "feature indices")
        symbols = _indices(self.fit_symbol_indices, "fit symbols")
        _window(self.start_index, self.stop_index)
        names = tuple(self.feature_names)
        if (
            len(names) != len(indices)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("feature names must match the ordered unique features")
        mean, scale = tuple(self.mean), tuple(self.scale)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in mean + scale
        ):
            raise ValueError("mean and scale must contain real numbers, not booleans")
        mean, scale = (
            tuple(float(value) for value in mean),
            tuple(float(value) for value in scale),
        )
        if len(mean) != len(indices) or len(scale) != len(indices):
            raise ValueError("mean and scale must match feature indices")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or any(value <= 0 for value in scale)
        ):
            raise ValueError("mean/scale must be finite and scale positive")
        counts = tuple(tuple(row) for row in self.usable_counts)
        if len(counts) != len(symbols) or any(
            len(row) != len(indices)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= self.stop_index - self.start_index
                for value in row
            )
            for row in counts
        ):
            raise ValueError(
                "usable counts must match fit scope and feature dimensions"
            )
        for name, value in (
            ("feature_indices", indices),
            ("feature_names", names),
            ("fit_symbol_indices", symbols),
            ("mean", mean),
            ("scale", scale),
            ("usable_counts", counts),
        ):
            object.__setattr__(self, name, value)

    def validate_features(self, feature_indices: tuple[int, ...]) -> None:
        if tuple(feature_indices) != self.feature_indices:
            raise ValueError("normalizer feature ordering differs from policy")

    def validate_training_scope(
        self, dataset: MarketDataset, symbols: tuple[int, ...], start: int, stop: int
    ) -> None:
        if (
            dataset.dataset_id != self.source_dataset_id
            or start != self.start_index
            or stop != self.stop_index
            or not set(symbols).issubset(self.fit_symbol_indices)
            or tuple(dataset.feature_names[i] for i in self.feature_indices)
            != self.feature_names
        ):
            raise ValueError("normalizer training scope differs from environment")

    def transform(self, selected: np.ndarray, usable: np.ndarray) -> np.ndarray:
        values = np.asarray(selected, dtype=np.float64)
        mask = np.asarray(usable, dtype=np.bool_)
        if values.shape != (len(self.feature_indices),) or mask.shape != values.shape:
            raise ValueError("normalizer input must match feature dimensions")
        if not np.isfinite(values[mask]).all():
            raise ValueError("usable normalized input must be finite")
        result = np.zeros_like(values)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result[mask] = (values[mask] - np.asarray(self.mean)[mask]) / np.asarray(
                self.scale
            )[mask]
        if not np.isfinite(result).all() or np.any(
            np.abs(result) > np.finfo(np.float32).max
        ):
            raise ValueError("normalized values must be finite float32")
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source_dataset_id": self.source_dataset_id,
            "feature_indices": list(self.feature_indices),
            "feature_names": list(self.feature_names),
            "fit_symbol_indices": list(self.fit_symbol_indices),
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "usable_counts": [list(row) for row in self.usable_counts],
        }

    @classmethod
    def from_payload(cls, value: Any) -> PPOFeatureNormalizer:
        keys = {
            "schema",
            "source_dataset_id",
            "feature_indices",
            "feature_names",
            "fit_symbol_indices",
            "start_index",
            "stop_index",
            "mean",
            "scale",
            "usable_counts",
        }
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or value["schema"] != SCHEMA
        ):
            raise ValueError("invalid normalizer schema or fields")
        sequence_fields = (
            "feature_indices",
            "feature_names",
            "fit_symbol_indices",
            "mean",
            "scale",
            "usable_counts",
        )
        if any(not isinstance(value[field], list) for field in sequence_fields) or any(
            not isinstance(row, list) for row in value["usable_counts"]
        ):
            raise ValueError("normalizer sequences and count rows must be JSON arrays")
        try:
            return cls(
                **{key: payload for key, payload in value.items() if key != "schema"}
            )
        except (TypeError, OverflowError) as error:
            raise ValueError("invalid normalizer metadata") from error


def fit_ppo_feature_normalizer(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    start_index: int,
    stop_index: int,
) -> PPOFeatureNormalizer:
    indices = validated_feature_indices(dataset, feature_indices)
    symbols = validated_symbol_indices(dataset, fit_symbol_indices)
    _window(start_index, stop_index)
    if stop_index >= dataset.n_bars:
        raise ValueError("training scope needs a terminal observation within dataset")
    means, scales = [], []
    counts = [[0 for _ in indices] for _ in symbols]
    for column, feature in enumerate(indices):
        per_symbol = []
        for row, symbol in enumerate(symbols):
            values = np.asarray(
                dataset.features[start_index:stop_index, symbol, feature],
                dtype=np.float64,
            )
            mask = dataset.feature_available[
                start_index:stop_index, symbol, feature
            ] & np.isfinite(values)
            usable = values[mask]
            if not usable.size:
                raise ValueError(
                    "every fit symbol/feature requires usable training values"
                )
            counts[row][column] = int(usable.size)
            per_symbol.append(usable)
        symbol_means = np.asarray([np.mean(values) for values in per_symbol])
        mean = float(np.mean(symbol_means))
        variance = float(
            np.mean([np.mean((values - mean) ** 2) for values in per_symbol])
        )
        scale = float(np.sqrt(variance))
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return PPOFeatureNormalizer(
        source_dataset_id=dataset.dataset_id,
        feature_indices=indices,
        feature_names=tuple(dataset.feature_names[i] for i in indices),
        fit_symbol_indices=symbols,
        start_index=start_index,
        stop_index=stop_index,
        mean=tuple(means),
        scale=tuple(scales),
        usable_counts=tuple(tuple(row) for row in counts),
    )
