"""Exclusive atomic publication wrapper for canonical market dataset artifacts."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

from trade_rl.data.artifact_codec import (
    DATASET_ARRAYS_NAME as MARKET_ARTIFACT_ARRAYS,
)
from trade_rl.data.artifact_codec import (
    DATASET_ARTIFACT_SCHEMA as MARKET_ARTIFACT_SCHEMA,
)
from trade_rl.data.artifact_codec import (
    DATASET_MANIFEST_NAME as MARKET_ARTIFACT_MANIFEST,
)
from trade_rl.data.artifact_codec import (
    DatasetArtifactFiles,
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


def publish_market_dataset_artifact(
    root: str | Path, dataset: MarketDataset
) -> PublishedDatasetArtifact:
    """Atomically publish one immutable artifact into a new destination."""

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
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedDatasetArtifact(
        root=output,
        manifest_path=output / MARKET_ARTIFACT_MANIFEST,
        arrays_path=output / MARKET_ARTIFACT_ARRAYS,
        artifact_digest=files.artifact_digest,
    )


def write_market_dataset_artifact(root: str | Path, dataset: MarketDataset) -> str:
    """Deprecated compatibility wrapper returning only the artifact digest."""

    warnings.warn(
        "write_market_dataset_artifact is deprecated; use "
        "publish_market_dataset_artifact",
        DeprecationWarning,
        stacklevel=2,
    )
    return publish_market_dataset_artifact(root, dataset).artifact_digest


def load_market_dataset_artifact(root: str | Path) -> MarketDataset:
    return load_dataset_files(Path(root))


__all__ = [
    "MARKET_ARTIFACT_ARRAYS",
    "MARKET_ARTIFACT_MANIFEST",
    "MARKET_ARTIFACT_SCHEMA",
    "PublishedDatasetArtifact",
    "load_market_dataset_artifact",
    "publish_market_dataset_artifact",
    "write_market_dataset_artifact",
]
