"""Shared causal supervised rows for forecast strategy families."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.dataset_scope import (
    validated_feature_indices,
    validated_symbol_indices,
    validated_training_scope,
)

_NANOSECONDS_PER_HOUR = 3_600_000_000_000
MAX_SUPERVISED_FORECAST_HORIZON_HOURS = ((2**63) - 1) // _NANOSECONDS_PER_HOUR


def validate_supervised_forecast_horizon_hours(
    value: object,
    *,
    field: str = "horizon_hours",
) -> int:
    """Validate that an hour horizon fits the signed datetime64[ns] range."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if value > MAX_SUPERVISED_FORECAST_HORIZON_HOURS:
        raise ValueError(
            f"{field} must be representable as a signed int64 nanosecond "
            f"duration (maximum {MAX_SUPERVISED_FORECAST_HORIZON_HOURS} hours)"
        )
    return value


def _validated_fit_cutoff_ns(value: np.datetime64) -> np.datetime64:
    """Normalize an exact, finite cutoff without datetime64 precision loss."""

    try:
        source = np.datetime64(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "fit_cutoff must be exactly representable as datetime64[ns]"
        ) from error
    if np.isnat(source):
        raise ValueError("fit_cutoff must not be NaT")
    try:
        cutoff_ns = int(
            _datetime64_values_to_ns_exact(np.asarray(source)).astype(np.int64)
        )
        cutoff = np.datetime64(cutoff_ns, "ns")
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "fit_cutoff must be exactly representable as datetime64[ns]"
        ) from error
    if np.isnat(cutoff):
        raise ValueError("fit_cutoff must be exactly representable as datetime64[ns]")
    return cutoff


def _immutable_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | str,
) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


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
        cutoff = _validated_fit_cutoff_ns(self.fit_cutoff)
        indices = tuple(self.feature_indices)
        features = _immutable_array(self.features, dtype=np.dtype(np.float64))
        labels = _immutable_array(
            np.asarray(self.labels).reshape(-1),
            dtype=np.dtype(np.float64),
        )
        raw_label_end_times = np.asarray(self.label_end_times).reshape(-1)
        if raw_label_end_times.dtype.kind != "M":
            raise ValueError("label_end_times must use a datetime64 dtype")
        if np.isnat(raw_label_end_times).any():
            raise ValueError("label_end_times must not contain NaT")
        try:
            normalized_label_end_times = _datetime64_values_to_ns_exact(
                raw_label_end_times
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "label_end_times must be exactly representable in nanoseconds"
            ) from error
        label_end_times = _immutable_array(
            normalized_label_end_times,
            dtype=np.dtype("datetime64[ns]"),
        )
        sample_weights = _immutable_array(
            np.asarray(self.sample_weights).reshape(-1),
            dtype=np.dtype(np.float64),
        )
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
        horizon_hours = validate_supervised_forecast_horizon_hours(self.horizon_hours)
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_end_times", label_end_times)
        object.__setattr__(self, "sample_weights", sample_weights)
        object.__setattr__(self, "fit_cutoff", cutoff)
        object.__setattr__(self, "horizon_hours", horizon_hours)

    @property
    def n_samples(self) -> int:
        return int(self.labels.size)


@dataclass(frozen=True, slots=True)
class _SelectedForecastRows:
    """Internal numeric rows and exact source coordinates, without return units."""

    feature_indices: tuple[int, ...]
    symbol_indices: tuple[int, ...]
    features: np.ndarray
    sample_weights: np.ndarray
    source_rows: tuple[tuple[int, int, int], ...]


def _select_causal_forecast_rows(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None,
    cutoff: np.datetime64,
    horizon_hours: int,
    strict_publication: bool,
) -> _SelectedForecastRows:
    # Callers validate the exact cutoff and horizon before entering the selector.
    indices, symbols = validated_training_scope(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
    )
    cutoff_ns = int(cutoff.astype(np.int64))
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    price_available_at = dataset.resolved_array("available_at")
    information_available = (
        dataset.resolved_array("information_available") if strict_publication else None
    )
    horizon_ns = horizon_hours * _NANOSECONDS_PER_HOUR
    time_to_index = {int(value): index for index, value in enumerate(timestamps_ns)}
    rows_by_symbol: list[list[np.ndarray]] = [[] for _ in range(dataset.n_symbols)]
    pairs_by_symbol: list[list[tuple[int, int]]] = [
        [] for _ in range(dataset.n_symbols)
    ]
    feature_columns = list(indices)
    for symbol_index in symbols:
        close = np.asarray(dataset.close[:, symbol_index], dtype=np.float64)
        availability = np.asarray(
            dataset.feature_available[:, symbol_index], dtype=np.bool_
        )
        for start_index, start_ns in enumerate(timestamps_ns):
            end_ns = int(start_ns) + horizon_ns
            if end_ns >= cutoff_ns:
                continue
            end_index = time_to_index.get(end_ns)
            if end_index is None or end_index <= start_index:
                continue
            start_available = price_available_at[start_index, symbol_index]
            end_available = price_available_at[end_index, symbol_index]
            if strict_publication:
                # Task18's original simple-return selector admitted only
                # information available on its source row and published strictly
                # before the fit cutoff. Do not change legacy log-label admission.
                assert information_available is not None
                if (
                    not information_available[start_index, symbol_index]
                    or not information_available[end_index, symbol_index]
                    or start_available >= cutoff
                    or end_available >= cutoff
                ):
                    continue
            elif start_available > cutoff or end_available > cutoff:
                continue
            if not bool(np.all(availability[start_index, feature_columns])):
                continue
            selected = np.asarray(
                dataset.features[start_index, symbol_index, feature_columns],
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
            rows_by_symbol[symbol_index].append(selected)
            pairs_by_symbol[symbol_index].append((start_index, end_index))

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
    sample_weights: list[float] = []
    source_rows: list[tuple[int, int, int]] = []
    for symbol_index in active_symbols:
        symbol_rows = rows_by_symbol[symbol_index]
        row_weight = per_symbol_weight / len(symbol_rows)
        rows.extend(symbol_rows)
        sample_weights.extend([row_weight] * len(symbol_rows))
        source_rows.extend(
            (symbol_index, start, end) for start, end in pairs_by_symbol[symbol_index]
        )
    return _SelectedForecastRows(
        feature_indices=indices,
        symbol_indices=symbols,
        features=np.stack(rows, axis=0),
        sample_weights=np.asarray(sample_weights, dtype=np.float64),
        source_rows=tuple(source_rows),
    )


def build_causal_forecast_training_set(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
) -> CausalForecastTrainingSet:
    """Pool exact-horizon log-return rows without symbol identity features."""

    horizon_hours = validate_supervised_forecast_horizon_hours(horizon_hours)
    cutoff = _validated_fit_cutoff_ns(fit_cutoff)
    rows = _select_causal_forecast_rows(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        cutoff=cutoff,
        horizon_hours=horizon_hours,
        strict_publication=False,
    )
    labels = [
        math.log(
            float(dataset.close[end, symbol]) / float(dataset.close[start, symbol])
        )
        for symbol, start, end in rows.source_rows
    ]
    timestamps = dataset.timestamps.astype("datetime64[ns]")
    return CausalForecastTrainingSet(
        feature_indices=rows.feature_indices,
        features=rows.features,
        labels=np.asarray(labels, dtype=np.float64),
        label_end_times=np.asarray(
            [timestamps[end] for _, _, end in rows.source_rows], dtype="datetime64[ns]"
        ),
        sample_weights=rows.sample_weights,
        fit_cutoff=cutoff,
        horizon_hours=horizon_hours,
    )


__all__ = [
    "CausalForecastTrainingSet",
    "build_causal_forecast_training_set",
    "validated_feature_indices",
    "validated_symbol_indices",
]
