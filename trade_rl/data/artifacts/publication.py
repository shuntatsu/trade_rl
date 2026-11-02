"""Exclusive atomic publication wrapper for canonical market dataset artifacts."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_rename_directory
from trade_rl.data.artifacts.codec import (
    DATASET_ARRAYS_NAME as MARKET_ARTIFACT_ARRAYS,
)
from trade_rl.data.artifacts.codec import (
    DATASET_ARTIFACT_SCHEMA as MARKET_ARTIFACT_SCHEMA,
)
from trade_rl.data.artifacts.codec import (
    DATASET_ARTIFACT_SCHEMA_V4 as MARKET_ARTIFACT_SCHEMA_V4,
)
from trade_rl.data.artifacts.codec import (
    DATASET_MANIFEST_NAME as MARKET_ARTIFACT_MANIFEST,
)
from trade_rl.data.artifacts.codec import (
    DatasetArtifactFiles,
    _load_dataset_files_with_digest,
    load_dataset_files,
    write_market_dataset_files,
)
from trade_rl.data.market import MarketDataset


def _write_arrays(path: Path, dataset: MarketDataset) -> DatasetArtifactFiles:
    """Compatibility seam used by atomic-publication failure tests."""

    return write_market_dataset_files(path, dataset)


@dataclass(frozen=True, slots=True)
class PublishedDatasetArtifact:
    """Identity and paths of an exclusively published dataset artifact."""

    root: Path
    manifest_path: Path
    arrays_path: Path
    artifact_digest: str
    schema_version: str = MARKET_ARTIFACT_SCHEMA


def inspect_published_market_dataset_artifact(
    root: str | Path,
) -> PublishedDatasetArtifact:
    """Validate and reconstruct the typed identity of an existing artifact."""

    artifact, _ = load_published_market_dataset_artifact(root)
    return artifact


def load_published_market_dataset_artifact(
    root: str | Path,
) -> tuple[PublishedDatasetArtifact, MarketDataset]:
    """Load a verified artifact identity and the Dataset it identifies.

    Both values come from one manifest parse and its verified array payload.
    This binds the returned pair to one loader pass; it does not lock the
    directory or guarantee an operating-system-level atomic filesystem snapshot.
    """

    artifact_root = Path(root)
    dataset, artifact_digest, schema_version = _load_dataset_files_with_digest(
        artifact_root
    )
    manifest_path = artifact_root / MARKET_ARTIFACT_MANIFEST
    return (
        PublishedDatasetArtifact(
            root=artifact_root,
            manifest_path=manifest_path,
            arrays_path=artifact_root / MARKET_ARTIFACT_ARRAYS,
            artifact_digest=artifact_digest,
            schema_version=schema_version,
        ),
        dataset,
    )


def publish_market_dataset_artifact(
    root: str | Path, dataset: MarketDataset
) -> PublishedDatasetArtifact:
    """Atomically publish one immutable filesystem artifact."""

    if not dataset.identity_verified:
        raise ValueError("market dataset publication requires a canonical identity")
    output = Path(root)
    if output.exists():
        raise FileExistsError(f"market artifact destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        files = _write_arrays(staging, dataset)
        loaded = load_dataset_files(staging)
        if loaded.dataset_id != dataset.dataset_id:
            raise ValueError("staged market artifact changed dataset identity")
        atomic_rename_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedDatasetArtifact(
        root=output,
        manifest_path=output / MARKET_ARTIFACT_MANIFEST,
        arrays_path=output / MARKET_ARTIFACT_ARRAYS,
        artifact_digest=files.artifact_digest,
        schema_version=files.schema_version,
    )


def load_market_dataset_artifact(root: str | Path) -> MarketDataset:
    return load_dataset_files(Path(root))


__all__ = [
    "MARKET_ARTIFACT_ARRAYS",
    "MARKET_ARTIFACT_MANIFEST",
    "MARKET_ARTIFACT_SCHEMA",
    "MARKET_ARTIFACT_SCHEMA_V4",
    "PublishedDatasetArtifact",
    "inspect_published_market_dataset_artifact",
    "load_market_dataset_artifact",
    "load_published_market_dataset_artifact",
    "publish_market_dataset_artifact",
]
