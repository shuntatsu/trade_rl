"""Validated point-in-time alpha and factor artifact providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trade_rl.artifacts.signals import (
    SignalArrayManifest,
    load_signal_artifact_payload,
)
from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class LoadedAlphaArtifact:
    manifest: SignalArrayManifest
    values: np.ndarray
    valid_mask: np.ndarray
    knowledge_cutoff: np.ndarray
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )

    @property
    def dataset_id(self) -> str:
        return self.bound_dataset_id or self.manifest.dataset_id

    def for_view(
        self, *, start: int, stop: int, dataset_id: str
    ) -> LoadedAlphaArtifact:
        _validate_view(self.values, start=start, stop=stop)
        return LoadedAlphaArtifact(
            manifest=self.manifest,
            values=self.values,
            valid_mask=self.valid_mask,
            knowledge_cutoff=self.knowledge_cutoff,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.values,
            self.valid_mask,
            self.knowledge_cutoff,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        return np.asarray(self.values[source_index], dtype=np.float64).copy()


@dataclass(frozen=True, slots=True)
class LoadedFactorArtifact:
    manifest: SignalArrayManifest
    values: np.ndarray
    valid_mask: np.ndarray
    knowledge_cutoff: np.ndarray
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.valid_mask,
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
        _validate_view(self.values, start=start, stop=stop)
        return LoadedFactorArtifact(
            manifest=self.manifest,
            values=self.values,
            valid_mask=self.valid_mask,
            knowledge_cutoff=self.knowledge_cutoff,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.values,
            self.valid_mask,
            self.knowledge_cutoff,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        return np.asarray(self.values[source_index], dtype=np.float64).copy()


def _validate_view(values: np.ndarray, *, start: int, stop: int) -> None:
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= values.shape[0]
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


def _validate_dataset_and_index(
    manifest: SignalArrayManifest,
    values: np.ndarray,
    valid_mask: np.ndarray,
    knowledge_cutoff: np.ndarray,
    dataset: MarketDataset,
    index: int,
    *,
    dataset_id: str,
    offset: int,
    bound_bars: int | None,
) -> int:
    if dataset.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    expected_bars = values.shape[0] if bound_bars is None else bound_bars
    if dataset.n_bars != expected_bars:
        raise ValueError("signal artifact bar count does not match dataset")
    source_index = index + offset
    if not 0 <= index < expected_bars or not valid_mask[source_index]:
        raise ValueError("signal artifact is unavailable at the requested index")
    if knowledge_cutoff[source_index] >= source_index:
        raise ValueError("signal artifact violates its point-in-time knowledge cutoff")
    return source_index


def _validate_common(
    manifest: SignalArrayManifest,
    valid_mask: np.ndarray,
    *,
    dataset_id: str,
    evaluation_start: int | None,
) -> None:
    if manifest.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    if evaluation_start is not None:
        if evaluation_start < 0 or evaluation_start >= valid_mask.shape[0]:
            raise ValueError("signal evaluation start is outside the artifact")
        if not np.any(valid_mask[evaluation_start:]):
            raise ValueError("signal artifact has no valid predictions for evaluation")


def load_alpha_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> LoadedAlphaArtifact:
    manifest, payload = load_signal_artifact_payload(root, expected_kind="alpha")
    _validate_common(
        manifest,
        payload.valid_mask,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_symbols is not None and manifest.names != expected_symbols:
        raise ValueError("alpha symbol names do not match the dataset")
    return LoadedAlphaArtifact(
        manifest=manifest,
        values=payload.values,
        valid_mask=payload.valid_mask,
        knowledge_cutoff=payload.knowledge_cutoff,
    )


def load_factor_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_names: tuple[str, ...] | None = None,
    expected_symbols: int | None = None,
) -> LoadedFactorArtifact:
    manifest, payload = load_signal_artifact_payload(root, expected_kind="factor")
    _validate_common(
        manifest,
        payload.valid_mask,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_names is not None and manifest.names != expected_names:
        raise ValueError("factor names do not match the action specification")
    if expected_symbols is not None and payload.values.shape[2] != expected_symbols:
        raise ValueError("factor artifact symbol count does not match the dataset")
    return LoadedFactorArtifact(
        manifest=manifest,
        values=payload.values,
        valid_mask=payload.valid_mask,
        knowledge_cutoff=payload.knowledge_cutoff,
    )


__all__ = [
    "LoadedAlphaArtifact",
    "LoadedFactorArtifact",
    "load_alpha_artifact",
    "load_factor_artifact",
]
