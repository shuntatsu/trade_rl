"""Exact frozen-source loading for Universal Trade RL U2 Development evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from trade_rl.data.artifacts import MarketDatasetView, load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseEntry,
    UniversalTradeRLUniverseManifest,
)

U2EvaluationSourceArtifactLocator: TypeAlias = str | Path
U2EvaluationSourceArtifactLoader: TypeAlias = Callable[
    [U2EvaluationSourceArtifactLocator], MarketDataset
]


@dataclass(frozen=True, slots=True)
class _EvaluationDatasetContract:
    symbol: str
    role: UniversalTradeRLSymbolRole
    source_dataset_digest: str
    evaluation_dataset_digest: str
    source_start: int
    source_stop: int


def _default_loader(locator: U2EvaluationSourceArtifactLocator) -> MarketDataset:
    return load_market_dataset_artifact(Path(locator))


def _required_dataset_contracts(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
) -> tuple[_EvaluationDatasetContract, ...]:
    if scope_closure.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 Development scope closure universe identity mismatch")

    by_symbol: dict[str, _EvaluationDatasetContract] = {}
    for scope in scope_closure.scopes:
        if not isinstance(scope, UniversalTradeRLU2EvaluationScope):
            raise TypeError("U2 Development scope closure contains an invalid scope")
        try:
            entry = manifest.entry_for(scope.concrete_symbol)
        except KeyError as error:
            raise ValueError(
                "U2 Development scope references an unknown symbol"
            ) from error
        if entry.role is UniversalTradeRLSymbolRole.ADMISSION:
            raise PermissionError(
                "U2 Development scope cannot reference an Admission symbol"
            )
        if entry.role not in {
            UniversalTradeRLSymbolRole.TRAIN,
            UniversalTradeRLSymbolRole.DEVELOPMENT,
        }:
            raise PermissionError(
                "U2 Development scope references a non-evaluation role"
            )
        if scope.symbol_role is not entry.role:
            raise PermissionError(
                "U2 Development scope symbol role mismatches the manifest"
            )
        if scope.source_dataset_digest != entry.dataset_digest:
            raise ValueError("U2 Development scope source dataset identity mismatch")

        source_start, source_stop = scope.evaluation_source_range
        candidate = _EvaluationDatasetContract(
            symbol=scope.concrete_symbol,
            role=scope.symbol_role,
            source_dataset_digest=scope.source_dataset_digest,
            evaluation_dataset_digest=scope.evaluation_dataset_digest,
            source_start=source_start,
            source_stop=source_stop,
        )
        existing = by_symbol.get(candidate.symbol)
        if existing is None:
            by_symbol[candidate.symbol] = candidate
        elif existing != candidate:
            raise ValueError(
                "U2 Development scopes disagree on one symbol evaluation dataset"
            )

    if not by_symbol:
        raise ValueError("U2 Development scope closure contains no evaluation symbols")
    return tuple(
        by_symbol[entry.symbol]
        for entry in manifest.entries
        if entry.symbol in by_symbol
    )


def _require_exact_locators(
    *,
    contracts: tuple[_EvaluationDatasetContract, ...],
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
) -> dict[str, U2EvaluationSourceArtifactLocator]:
    if not isinstance(artifact_locators, Mapping):
        raise TypeError("U2 Development source artifact locators must be a mapping")
    required_symbols = tuple(contract.symbol for contract in contracts)
    observed_symbols = tuple(artifact_locators)
    if len(observed_symbols) != len(required_symbols) or set(observed_symbols) != set(
        required_symbols
    ):
        raise ValueError(
            "U2 Development source artifact locator closure must equal evaluation symbols"
        )

    resolved: dict[str, U2EvaluationSourceArtifactLocator] = {}
    for symbol in required_symbols:
        locator = artifact_locators[symbol]
        if not isinstance(locator, str | Path):
            raise TypeError(
                "U2 Development source artifact locator must be string or Path"
            )
        resolved[symbol] = locator
    return resolved


def _timestamps_ns(dataset: MarketDataset) -> np.ndarray:
    timestamps = (
        np.asarray(dataset.timestamps).astype("datetime64[ns]").astype(np.int64)
    )
    if timestamps.ndim != 1 or timestamps.size != dataset.n_bars:
        raise ValueError("U2 Development source timestamp layout is invalid")
    return timestamps


def _require_canonical_source(
    *,
    entry: UniversalTradeRLUniverseEntry,
    dataset: MarketDataset,
) -> MarketDataset:
    if not isinstance(dataset, MarketDataset):
        raise TypeError("U2 Development source loader must return a MarketDataset")
    if not dataset.identity_verified:
        raise ValueError("U2 Development source must have verified canonical identity")
    if dataset.dataset_id != entry.dataset_digest:
        raise ValueError("U2 Development source dataset identity mismatch")
    if dataset.symbols != (entry.symbol,):
        raise ValueError("U2 Development source symbol mismatch")
    if dataset.n_bars != entry.row_count:
        raise ValueError("U2 Development source row count mismatch")

    timestamps = _timestamps_ns(dataset)
    if int(timestamps[0]) != entry.first_timestamp_ns:
        raise ValueError("U2 Development source first timestamp mismatch")
    if int(timestamps[-1]) != entry.last_timestamp_ns:
        raise ValueError("U2 Development source last timestamp mismatch")
    expected = entry.first_timestamp_ns + np.arange(
        entry.row_count,
        dtype=np.int64,
    ) * np.int64(U2_DECISION_STEP_NS)
    if not np.array_equal(timestamps, expected):
        raise ValueError("U2 Development source differs from the frozen dense 15m grid")
    return dataset


def _materialize_evaluation_dataset(
    *,
    source: MarketDataset,
    contract: _EvaluationDatasetContract,
) -> MarketDataset:
    view = MarketDatasetView(
        source,
        contract.source_start,
        contract.source_stop,
    )
    if view.identity != contract.evaluation_dataset_digest:
        raise ValueError("U2 Development common-view dataset identity mismatch")
    materialized = view.materialize()
    if materialized.dataset_id != contract.evaluation_dataset_digest:
        raise ValueError("U2 Development materialized dataset identity mismatch")
    if materialized.n_bars != contract.source_stop - contract.source_start:
        raise ValueError("U2 Development materialized common-view size mismatch")
    return materialized


def load_universal_trade_rl_u2_development_evaluation_datasets(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
    loader: U2EvaluationSourceArtifactLoader = _default_loader,
) -> dict[str, MarketDataset]:
    """Load each exact Train/Development source once and expose only its common view."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 Development dataset loading requires a U0 manifest")
    if not isinstance(scope_closure, UniversalTradeRLU2DevelopmentScopeClosure):
        raise TypeError("U2 Development dataset loading requires a scope closure")
    if not callable(loader):
        raise TypeError("U2 Development source artifact loader must be callable")

    # Everything above the loader boundary is validated first so malformed scopes,
    # Admission references, and locator drift cannot trigger numeric source access.
    contracts = _required_dataset_contracts(
        manifest=manifest,
        scope_closure=scope_closure,
    )
    locators = _require_exact_locators(
        contracts=contracts,
        artifact_locators=artifact_locators,
    )

    datasets: dict[str, MarketDataset] = {}
    for contract in contracts:
        entry = manifest.entry_for(contract.symbol)
        loaded = loader(locators[contract.symbol])
        source = _require_canonical_source(entry=entry, dataset=loaded)
        datasets[contract.symbol] = _materialize_evaluation_dataset(
            source=source,
            contract=contract,
        )
    return datasets


__all__ = [
    "U2EvaluationSourceArtifactLoader",
    "U2EvaluationSourceArtifactLocator",
    "load_universal_trade_rl_u2_development_evaluation_datasets",
]
