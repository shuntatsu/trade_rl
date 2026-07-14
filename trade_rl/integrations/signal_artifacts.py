"""Validated providers backed by point-in-time signal artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trade_rl.artifacts.signals import (
    SignalArrayManifest,
    SignalArrays,
    load_signal_artifact,
)
from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class LoadedAlphaArtifact:
    manifest: SignalArrayManifest
    arrays: SignalArrays
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def values(self) -> np.ndarray:
        return self.arrays.values

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.arrays.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )

    @property
    def dataset_id(self) -> str:
        return self.bound_dataset_id or self.manifest.dataset_id

    def for_view(
        self, *, start: int, stop: int, dataset_id: str
    ) -> LoadedAlphaArtifact:
        _validate_view(self.arrays, start=start, stop=stop)
        return LoadedAlphaArtifact(
            manifest=self.manifest,
            arrays=self.arrays,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.arrays,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        if not np.all(self.arrays.valid[source_index]):
            raise ValueError("signal prediction is invalid at the requested index")
        return np.asarray(self.arrays.values[source_index], dtype=np.float64).copy()


@dataclass(frozen=True, slots=True)
class LoadedFactorArtifact:
    manifest: SignalArrayManifest
    arrays: SignalArrays
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def values(self) -> np.ndarray:
        return self.arrays.values

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.arrays.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )

    @property
    def dataset_id(self) -> str:
        return self.bound_dataset_id or self.manifest.dataset_id

    @property
    def n_factors(self) -> int:
        return len(self.manifest.names)

    def for_view(
        self, *, start: int, stop: int, dataset_id: str
    ) -> LoadedFactorArtifact:
        _validate_view(self.arrays, start=start, stop=stop)
        return LoadedFactorArtifact(
            manifest=self.manifest,
            arrays=self.arrays,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.arrays,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        if not np.all(self.arrays.valid[source_index]):
            raise ValueError("signal prediction is invalid at the requested index")
        return np.asarray(self.arrays.values[source_index], dtype=np.float64).copy()


def _validate_view(arrays: SignalArrays, *, start: int, stop: int) -> None:
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= arrays.shape[0]
    ):
        raise ValueError("signal artifact view is outside the source values")


def _minimum_index(
    valid_mask: np.ndarray,
    *,
    offset: int,
    bound_bars: int | None,
) -> int:
    stop = valid_mask.shape[0] if bound_bars is None else offset + bound_bars
    available = np.flatnonzero(valid_mask[offset:stop])
    if available.size == 0:
        raise ValueError("signal artifact has no valid predictions in the bound range")
    return int(available[0])


def _decision_value(
    dataset: MarketDataset,
    index: int,
    source_index: int,
    available_at: np.ndarray,
) -> np.datetime64 | int:
    if np.issubdtype(available_at.dtype, np.datetime64):
        return np.datetime64(dataset.timestamps[index], "ns")
    return source_index


def _validate_dataset_and_index(
    manifest: SignalArrayManifest,
    arrays: SignalArrays,
    dataset: MarketDataset,
    index: int,
    *,
    dataset_id: str,
    offset: int,
    bound_bars: int | None,
) -> int:
    if dataset.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    expected_bars = arrays.shape[0] if bound_bars is None else bound_bars
    if dataset.n_bars != expected_bars:
        raise ValueError("signal artifact bar count does not match dataset")
    if not 0 <= index < expected_bars:
        raise ValueError("signal artifact is unavailable at the requested index")
    source_index = index + offset
    if not np.all(arrays.valid[source_index]):
        if np.any(arrays.valid[source_index]):
            raise ValueError("signal prediction is invalid at the requested index")
        raise ValueError("signal artifact is unavailable at the requested index")
    if arrays.knowledge_cutoff[source_index] < 0:
        raise ValueError("signal artifact is unavailable at the requested index")
    decision = _decision_value(dataset, index, source_index, arrays.available_at)
    if arrays.available_at[source_index] > decision:
        raise ValueError("signal prediction is not available at the decision timestamp")
    if arrays.knowledge_cutoff[source_index] >= source_index:
        raise ValueError("signal artifact violates its point-in-time knowledge cutoff")
    return source_index


def _validate_common(
    manifest: SignalArrayManifest,
    arrays: SignalArrays,
    *,
    dataset_id: str,
    evaluation_start: int | None,
) -> None:
    if manifest.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    if evaluation_start is not None:
        if evaluation_start < 0 or evaluation_start >= arrays.shape[0]:
            raise ValueError("signal evaluation start is outside the artifact")
        if not np.any(arrays.valid_mask[evaluation_start:]):
            raise ValueError("signal artifact has no valid predictions for evaluation")


def load_alpha_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> LoadedAlphaArtifact:
    manifest, arrays = load_signal_artifact(root, expected_kind="alpha")
    _validate_common(
        manifest,
        arrays,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_symbols is not None and manifest.names != expected_symbols:
        raise ValueError("alpha symbol names do not match the dataset")
    return LoadedAlphaArtifact(manifest=manifest, arrays=arrays)


def load_factor_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_names: tuple[str, ...] | None = None,
    expected_symbols: int | None = None,
) -> LoadedFactorArtifact:
    manifest, arrays = load_signal_artifact(root, expected_kind="factor")
    _validate_common(
        manifest,
        arrays,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_names is not None and manifest.names != expected_names:
        raise ValueError("factor names do not match the action specification")
    if expected_symbols is not None and arrays.shape[2] != expected_symbols:
        raise ValueError("factor artifact symbol count does not match the dataset")
    return LoadedFactorArtifact(manifest=manifest, arrays=arrays)


__all__ = [
    "LoadedAlphaArtifact",
    "LoadedFactorArtifact",
    "load_alpha_artifact",
    "load_factor_artifact",
]
