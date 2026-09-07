"""Canonical U0 source artifacts materialized as exact U2 FIT dataset views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np

from trade_rl.data.artifact import load_market_dataset_artifact
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS

U2SourceArtifactLocator: TypeAlias = str | Path
U2SourceArtifactLoader: TypeAlias = Callable[[U2SourceArtifactLocator], MarketDataset]


def _require_exact_locator_closure(
    *,
    closure: U2TrainingSourceClosure,
    artifact_locators: Mapping[str, U2SourceArtifactLocator],
) -> dict[str, U2SourceArtifactLocator]:
    if not isinstance(artifact_locators, Mapping):
        raise TypeError("U2 source artifact locators must be a mapping")
    expected_symbols = tuple(source.symbol for source in closure.sources)
    observed_symbols = tuple(artifact_locators)
    if len(observed_symbols) != len(expected_symbols) or set(observed_symbols) != set(
        expected_symbols
    ):
        raise ValueError("U2 source artifact locator closure must equal Train symbols")

    resolved: dict[str, U2SourceArtifactLocator] = {}
    for symbol in expected_symbols:
        locator = artifact_locators[symbol]
        if not isinstance(locator, str | Path):
            raise TypeError("U2 source artifact locator must be a string or Path")
        resolved[symbol] = locator
    return resolved


def _source_timestamps_ns(dataset: MarketDataset) -> np.ndarray:
    timestamps = (
        np.asarray(dataset.timestamps).astype("datetime64[ns]").astype(np.int64)
    )
    if timestamps.ndim != 1 or timestamps.size != dataset.n_bars:
        raise ValueError("U2 source dataset timestamp layout is invalid")
    return timestamps


def _require_source_dataset(
    *,
    source: U2TrainingSource,
    dataset: MarketDataset,
) -> MarketDataset:
    if not isinstance(dataset, MarketDataset):
        raise TypeError("U2 source artifact loader must return a MarketDataset")
    if not dataset.identity_verified:
        raise ValueError("U2 source dataset must have verified canonical identity")
    if dataset.dataset_id != source.dataset_digest:
        raise ValueError("U2 source dataset identity mismatch")
    if dataset.symbols != (source.symbol,):
        raise ValueError("U2 source dataset symbol mismatch")
    if dataset.n_bars != source.source_row_count:
        raise ValueError("U2 source dataset row count mismatch")

    timestamps_ns = _source_timestamps_ns(dataset)
    expected = source.source_first_timestamp_ns + np.arange(
        source.source_row_count,
        dtype=np.int64,
    ) * np.int64(U2_DECISION_STEP_NS)
    if not np.array_equal(timestamps_ns, expected):
        raise ValueError("U2 source dataset timestamps differ from frozen source grid")
    return dataset


def _materialize_fit_dataset(
    *,
    source: U2TrainingSource,
    dataset: MarketDataset,
) -> MarketDataset:
    view = MarketDatasetView(
        dataset=dataset,
        start=source.fit_start_index,
        stop=source.fit_stop_index,
    )
    fit_dataset = view.materialize()
    if fit_dataset.dataset_id != view.identity:
        raise ValueError("U2 FIT dataset view identity mismatch")
    if fit_dataset.n_bars != source.fit_bar_count:
        raise ValueError("U2 FIT dataset bar count mismatch")

    timestamps_ns = _source_timestamps_ns(fit_dataset)
    if int(timestamps_ns[0]) != source.fit_first_timestamp_ns:
        raise ValueError("U2 FIT dataset first timestamp mismatch")
    if int(timestamps_ns[-1]) != source.fit_last_timestamp_ns:
        raise ValueError("U2 FIT dataset last timestamp mismatch")
    return fit_dataset


def load_universal_trade_rl_u2_fit_datasets(
    *,
    closure: U2TrainingSourceClosure,
    artifact_locators: Mapping[str, U2SourceArtifactLocator],
    loader: U2SourceArtifactLoader = load_market_dataset_artifact,
) -> dict[str, MarketDataset]:
    """Load exact frozen Train sources and materialize their preregistered FIT views."""

    if not isinstance(closure, U2TrainingSourceClosure):
        raise TypeError("U2 FIT dataset loading requires a verified source closure")
    if not callable(loader):
        raise TypeError("U2 source artifact loader must be callable")
    locators = _require_exact_locator_closure(
        closure=closure,
        artifact_locators=artifact_locators,
    )

    fit_datasets: dict[str, MarketDataset] = {}
    for source in closure.sources:
        loaded = loader(locators[source.symbol])
        dataset = _require_source_dataset(source=source, dataset=loaded)
        fit_datasets[source.symbol] = _materialize_fit_dataset(
            source=source,
            dataset=dataset,
        )
    return fit_datasets


__all__ = [
    "U2SourceArtifactLoader",
    "U2SourceArtifactLocator",
    "load_universal_trade_rl_u2_fit_datasets",
]
