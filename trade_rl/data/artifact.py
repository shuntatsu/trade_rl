"""Exclusive atomic publication wrapper for canonical market dataset artifacts."""

from __future__ import annotations

import shutil
import tempfile
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
    load_dataset_files,
    write_dataset_files,
)
from trade_rl.data.market import MarketDataset


def _write_arrays(path: Path, dataset: MarketDataset) -> tuple[Path, str]:
    """Compatibility seam used by atomic-publication failure tests."""

    return write_dataset_files(path, dataset)


def write_market_dataset_artifact(root: str | Path, dataset: MarketDataset) -> str:
    """Atomically publish one immutable artifact into a new destination."""

    output = Path(root)
    if output.exists():
        raise FileExistsError(f"market artifact destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        _, artifact_digest = _write_arrays(staging, dataset)
        loaded = load_dataset_files(staging)
        if loaded.dataset_id != dataset.dataset_id:
            raise ValueError("staged market artifact changed dataset identity")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return artifact_digest


def load_market_dataset_artifact(root: str | Path) -> MarketDataset:
    return load_dataset_files(Path(root))


__all__ = [
    "MARKET_ARTIFACT_ARRAYS",
    "MARKET_ARTIFACT_MANIFEST",
    "MARKET_ARTIFACT_SCHEMA",
    "load_market_dataset_artifact",
    "write_market_dataset_artifact",
]
