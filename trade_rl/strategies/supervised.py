"""Shared causal supervised rows for forecast strategy families."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset


def validated_feature_indices(
    dataset: MarketDataset,
    feature_indices: tuple[int, ...],
) -> tuple[int, ...]:
    indices = tuple(feature_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("feature_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("feature_indices must contain non-negative integers")
    if max(indices) >= dataset.n_features:
        raise ValueError("feature index is outside dataset features")
    return indices


def validated_symbol_indices(
    dataset: MarketDataset,
    symbol_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve an optional fit scope while rejecting ambiguous symbol indices."""

    if symbol_indices is None:
        return tuple(range(dataset.n_symbols))
    indices = tuple(symbol_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("symbol_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("symbol_indices must contain non-negative integers")
    if max(indices) >= dataset.n_symbols:
        raise ValueError("symbol index is outside dataset symbols")
    return indices


@dataclass(frozen=True, slots=True)
class CausalForecastTrainingSet:
    """Frozen fit-prefix rows with complete forward labels and fit-only weights."""

    feature_indices: tuple[int, ...]
    features: np.ndarray
    labels: np.ndarray
    label_end_times: np.ndarray
    sample_weights: np.ndarray
    fit_cutoff: np.datetime64
    horizon_hours: int

    def __post_init__(self) -> None:
        indices = tuple(self.feature_indices)
        features = np.asarray(self.features, dtype=np.float64).copy()
        labels = np.asarray(self.labels, dtype=np.float64).reshape(-1).copy()
        label_end_times = (
            np.asarray(self.label_end_times, dtype="datetime64[ns]").reshape(-1).copy()
        )
        sample_weights = (
            np.asarray(self.sample_weights, dtype=np.float64).reshape(-1).copy()
        )
        cutoff = np.datetime64(self.fit_cutoff, "ns")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional array")
        if features.shape[1] != len(indices):
            raise ValueError("features must match feature_indices")
        if labels.shape != (features.shape[0],):
            raise ValueError("labels must match training rows")
        if label_end_times.shape != labels.shape:
            raise ValueError("label_end_times must match training rows")
        if sample_weights.shape != labels.shape:
            raise ValueError("sample_weights must match training rows")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError("training features and labels must be finite")
        if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0.0):
            raise ValueError("sample_weights must be finite and positive")
        if np.any(label_end_times >= cutoff):
            raise ValueError("all label_end_times must be strictly before fit_cutoff")
        if (
            isinstance(self.horizon_hours, bool)
            or not isinstance(self.horizon_hours, int)
            or self.horizon_hours <= 0
        ):
            raise ValueError("horizon_hours must be a positive integer")
        features.setflags(write=False)
        labels.setflags(write=False)
        label_end_times.setflags(write=False)
        sample_weights.setflags(write=False)
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_end_times", label_end_times)
        object.__setattr__(self, "sample_weights", sample_weights)
        object.__setattr__(self, "fit_cutoff", cutoff)

    @property
    def n_samples(self) -> int:
        return int(self.labels.size)


def build_causal_forecast_training_set(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
) -> CausalForecastTrainingSet:
    """Pool exact-horizon rows without adding symbol identity to model features."""

    indices = validated_feature_indices(dataset, feature_indices)
    symbols = validated_symbol_indices(dataset, fit_symbol_indices)
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, int)
        or horizon_hours <= 0
    ):
        raise ValueError("horizon_hours must be a positive integer")

    cutoff = np.datetime64(fit_cutoff, "ns")
    cutoff_ns = int(cutoff.astype(np.int64))
    timestamps = dataset.timestamps.astype("datetime64[ns]")
    timestamps_ns = timestamps.astype(np.int64)
    horizon_ns = int(
        np.timedelta64(horizon_hours, "h").astype("timedelta64[ns]").astype(np.int64)
    )
    time_to_index = {int(value): index for index, value in enumerate(timestamps_ns)}

    rows_by_symbol: list[list[np.ndarray]] = [[] for _ in range(dataset.n_symbols)]
    labels_by_symbol: list[list[float]] = [[] for _ in range(dataset.n_symbols)]
    ends_by_symbol: list[list[np.datetime64]] = [[] for _ in range(dataset.n_symbols)]
    for symbol_index in symbols:
        close = np.asarray(dataset.close[:, symbol_index], dtype=np.float64)
        availability = np.asarray(
            dataset.feature_available[:, symbol_index],
            dtype=np.bool_,
        )
        symbol_rows: list[np.ndarray] = []
        symbol_labels: list[float] = []
        symbol_ends: list[np.datetime64] = []
        for start_index, start_ns in enumerate(timestamps_ns):
            end_ns = int(start_ns) + horizon_ns
            if end_ns >= cutoff_ns:
                continue
            end_index = time_to_index.get(end_ns)
            if end_index is None or end_index <= start_index:
                continue
            if not bool(np.all(availability[start_index, list(indices)])):
                continue
            selected = np.asarray(
                dataset.features[start_index, symbol_index, list(indices)],
                dtype=np.float64,
            )
            if not np.isfinite(selected).all():
                continue
            start_price = float(close[start_index])
            end_price = float(close[end_index])
            if (
                not math.isfinite(start_price)
                or not math.isfinite(end_price)
                or start_price <= 0.0
                or end_price <= 0.0
            ):
                continue
            symbol_rows.append(selected)
            symbol_labels.append(math.log(end_price / start_price))
            symbol_ends.append(timestamps[end_index])
        rows_by_symbol[symbol_index] = symbol_rows
        labels_by_symbol[symbol_index] = symbol_labels
        ends_by_symbol[symbol_index] = symbol_ends

    if fit_symbol_indices is not None:
        missing_symbols = [index for index in symbols if not rows_by_symbol[index]]
        if missing_symbols:
            names = ", ".join(dataset.symbols[index] for index in missing_symbols)
            raise ValueError(f"fit symbol has no eligible training rows: {names}")

    active_symbols = [index for index in symbols if rows_by_symbol[index]]
    total_rows = sum(len(rows_by_symbol[index]) for index in active_symbols)
    if total_rows < 2:
        raise ValueError(
            "forecast fitting requires at least two eligible training rows"
        )

    per_symbol_weight = total_rows / len(active_symbols)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    label_end_times: list[np.datetime64] = []
    sample_weights: list[float] = []
    for symbol_index in active_symbols:
        symbol_rows = rows_by_symbol[symbol_index]
        row_weight = per_symbol_weight / len(symbol_rows)
        rows.extend(symbol_rows)
        labels.extend(labels_by_symbol[symbol_index])
        label_end_times.extend(ends_by_symbol[symbol_index])
        sample_weights.extend([row_weight] * len(symbol_rows))

    return CausalForecastTrainingSet(
        feature_indices=indices,
        features=np.stack(rows, axis=0),
        labels=np.asarray(labels, dtype=np.float64),
        label_end_times=np.asarray(label_end_times, dtype="datetime64[ns]"),
        sample_weights=np.asarray(sample_weights, dtype=np.float64),
        fit_cutoff=cutoff,
        horizon_hours=horizon_hours,
    )


__all__ = [
    "CausalForecastTrainingSet",
    "build_causal_forecast_training_set",
    "validated_feature_indices",
    "validated_symbol_indices",
]
