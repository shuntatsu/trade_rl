"""Immutable publication of catalog-bound universal-instrument artifacts."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from trade_rl.catalog.stored_instrument_catalog import (
    StoredIndicatorSourceInventory,
    StoredInstrumentCatalog,
    build_stored_instrument_catalog,
    load_stored_instrument_catalog,
    write_stored_instrument_catalog,
)
from trade_rl.workflows.symbol_disjoint_manifest import (
    SymbolDisjointManifest,
    load_symbol_disjoint_manifest,
    write_symbol_disjoint_manifest,
)
from trade_rl.workflows.universal_instrument_partition import (
    UniversalInstrumentPartition,
    build_universal_instrument_partition,
    load_universal_instrument_partition,
    write_universal_instrument_partition,
)

STORED_INSTRUMENTS_FILENAME = "stored-instruments.json"
SYMBOL_DISJOINT_FILENAME = "symbol-disjoint.json"
UNIVERSAL_INSTRUMENT_PARTITION_FILENAME = "universal-instrument-partition.json"
_REQUIRED_FILENAMES = frozenset(
    {
        STORED_INSTRUMENTS_FILENAME,
        SYMBOL_DISJOINT_FILENAME,
        UNIVERSAL_INSTRUMENT_PARTITION_FILENAME,
    }
)
_LOAD_ERRORS = (
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class UniversalInstrumentArtifactBundle:
    """Three immutable contracts whose identities must close exactly."""

    catalog: StoredInstrumentCatalog
    symbol_disjoint_manifest: SymbolDisjointManifest
    partition: UniversalInstrumentPartition

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, StoredInstrumentCatalog):
            raise TypeError("catalog must be StoredInstrumentCatalog")
        if not isinstance(self.symbol_disjoint_manifest, SymbolDisjointManifest):
            raise TypeError(
                "symbol_disjoint_manifest must be SymbolDisjointManifest"
            )
        if not isinstance(self.partition, UniversalInstrumentPartition):
            raise TypeError("partition must be UniversalInstrumentPartition")
        if self.partition.catalog_digest != self.catalog.digest:
            raise ValueError("universal artifact catalog digest mismatch")
        if self.partition.symbol_disjoint_manifest != self.symbol_disjoint_manifest:
            raise ValueError("universal artifact manifest and partition mismatch")
        if (
            self.partition.symbol_disjoint_manifest_digest
            != self.symbol_disjoint_manifest.digest
        ):
            raise ValueError("universal artifact manifest digest mismatch")
        expected_universe = tuple(sorted(self.catalog.eligible_symbols))
        if self.symbol_disjoint_manifest.source_universe != expected_universe:
            raise ValueError("universal artifact catalog symbol closure mismatch")


@dataclass(frozen=True, slots=True)
class UniversalInstrumentArtifactPaths:
    """Canonical paths for one dedicated universal-instrument bundle."""

    root: Path
    stored_instruments: Path
    symbol_disjoint: Path
    universal_partition: Path

    def __post_init__(self) -> None:
        root = Path(self.root)
        stored_instruments = Path(self.stored_instruments)
        symbol_disjoint = Path(self.symbol_disjoint)
        universal_partition = Path(self.universal_partition)
        if stored_instruments != root / STORED_INSTRUMENTS_FILENAME:
            raise ValueError("stored instrument artifact path mismatch")
        if symbol_disjoint != root / SYMBOL_DISJOINT_FILENAME:
            raise ValueError("symbol-disjoint artifact path mismatch")
        if universal_partition != root / UNIVERSAL_INSTRUMENT_PARTITION_FILENAME:
            raise ValueError("universal partition artifact path mismatch")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "stored_instruments", stored_instruments)
        object.__setattr__(self, "symbol_disjoint", symbol_disjoint)
        object.__setattr__(self, "universal_partition", universal_partition)

    @classmethod
    def for_root(cls, root: str | Path) -> UniversalInstrumentArtifactPaths:
        resolved = Path(root)
        return cls(
            root=resolved,
            stored_instruments=resolved / STORED_INSTRUMENTS_FILENAME,
            symbol_disjoint=resolved / SYMBOL_DISJOINT_FILENAME,
            universal_partition=(
                resolved / UNIVERSAL_INSTRUMENT_PARTITION_FILENAME
            ),
        )


def build_universal_instrument_artifact_bundle(
    source: StoredIndicatorSourceInventory,
    *,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    seed: int,
) -> UniversalInstrumentArtifactBundle:
    """Build one catalog and its deterministic symbol-disjoint partition."""

    catalog = build_stored_instrument_catalog(
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
    )
    partition = build_universal_instrument_partition(catalog, seed=seed)
    return UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        partition=partition,
    )


def _require_bundle_root(root: Path) -> UniversalInstrumentArtifactPaths:
    if root.is_symlink():
        raise ValueError("universal instrument artifact root must not be a symlink")
    if not root.exists():
        raise ValueError("universal instrument artifact root does not exist")
    if not root.is_dir():
        raise ValueError("universal instrument artifact root must be a directory")

    paths = UniversalInstrumentArtifactPaths.for_root(root)
    observed_names = {entry.name for entry in root.iterdir()}
    if observed_names != _REQUIRED_FILENAMES:
        raise ValueError("universal instrument artifact filename closure mismatch")
    required_paths = (
        paths.stored_instruments,
        paths.symbol_disjoint,
        paths.universal_partition,
    )
    if any(path.is_symlink() or not path.is_file() for path in required_paths):
        raise ValueError("universal instrument artifacts must be regular files")
    return paths


def load_universal_instrument_artifact_bundle(
    root: str | Path,
) -> UniversalInstrumentArtifactBundle:
    """Strict-load all three contracts and revalidate their cross-bindings."""

    paths = _require_bundle_root(Path(root))
    try:
        catalog = load_stored_instrument_catalog(paths.stored_instruments)
    except _LOAD_ERRORS as error:
        raise ValueError(f"stored instrument catalog is invalid: {error}") from error
    try:
        manifest = load_symbol_disjoint_manifest(paths.symbol_disjoint)
    except _LOAD_ERRORS as error:
        raise ValueError(f"symbol-disjoint manifest is invalid: {error}") from error
    try:
        partition = load_universal_instrument_partition(
            paths.universal_partition,
            catalog=catalog,
            symbol_disjoint_manifest=manifest,
        )
    except _LOAD_ERRORS as error:
        raise ValueError(f"universal instrument partition is invalid: {error}") from error
    return UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=manifest,
        partition=partition,
    )


def _existing_bundle_paths(
    root: Path,
    expected: UniversalInstrumentArtifactBundle,
) -> UniversalInstrumentArtifactPaths:
    if root.is_symlink():
        raise FileExistsError(
            "universal instrument artifact root exists as a symlink"
        )
    if not root.is_dir():
        raise FileExistsError(
            "universal instrument artifact root exists and is not a directory"
        )
    try:
        observed = load_universal_instrument_artifact_bundle(root)
    except _LOAD_ERRORS as error:
        raise FileExistsError(
            "universal instrument artifact root exists with partial or different content"
        ) from error
    if observed != expected:
        raise FileExistsError(
            "universal instrument artifact root exists with different content"
        )
    return UniversalInstrumentArtifactPaths.for_root(root)


def _fsync_file(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging_root(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def write_universal_instrument_artifact_bundle(
    root: str | Path,
    bundle: UniversalInstrumentArtifactBundle,
) -> UniversalInstrumentArtifactPaths:
    """Publish all contracts together or require exact immutable reuse."""

    if not isinstance(bundle, UniversalInstrumentArtifactBundle):
        raise TypeError("bundle must be UniversalInstrumentArtifactBundle")
    output_root = Path(root)
    if output_root.is_symlink():
        raise FileExistsError(
            "universal instrument artifact root exists as a symlink"
        )
    if output_root.exists():
        return _existing_bundle_paths(output_root, bundle)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / (
        f".{output_root.name}.staging-{os.getpid()}-{uuid4().hex}"
    )
    staging_paths = UniversalInstrumentArtifactPaths.for_root(staging_root)
    published = False
    try:
        staging_root.mkdir(mode=0o700)
        write_stored_instrument_catalog(
            staging_paths.stored_instruments,
            bundle.catalog,
        )
        write_symbol_disjoint_manifest(
            staging_paths.symbol_disjoint,
            bundle.symbol_disjoint_manifest,
        )
        write_universal_instrument_partition(
            staging_paths.universal_partition,
            bundle.partition,
        )
        staged_bundle = load_universal_instrument_artifact_bundle(staging_root)
        if staged_bundle != bundle:
            raise ValueError("staged universal instrument artifact bundle mismatch")

        for path in (
            staging_paths.stored_instruments,
            staging_paths.symbol_disjoint,
            staging_paths.universal_partition,
        ):
            _fsync_file(path)
        _fsync_directory(staging_root)
        try:
            os.rename(staging_root, output_root)
        except OSError:
            if output_root.exists() or output_root.is_symlink():
                return _existing_bundle_paths(output_root, bundle)
            raise
        published = True
        _fsync_directory(output_root.parent)
        return UniversalInstrumentArtifactPaths.for_root(output_root)
    finally:
        if not published:
            _remove_staging_root(staging_root)


def materialize_universal_instrument_artifacts(
    root: str | Path,
    source: StoredIndicatorSourceInventory,
    *,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    seed: int,
) -> UniversalInstrumentArtifactPaths:
    """Build a complete valid bundle before touching the output filesystem."""

    bundle = build_universal_instrument_artifact_bundle(
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
        seed=seed,
    )
    return write_universal_instrument_artifact_bundle(root, bundle)


__all__ = [
    "STORED_INSTRUMENTS_FILENAME",
    "SYMBOL_DISJOINT_FILENAME",
    "UNIVERSAL_INSTRUMENT_PARTITION_FILENAME",
    "UniversalInstrumentArtifactBundle",
    "UniversalInstrumentArtifactPaths",
    "build_universal_instrument_artifact_bundle",
    "load_universal_instrument_artifact_bundle",
    "materialize_universal_instrument_artifacts",
    "write_universal_instrument_artifact_bundle",
]
