"""Explicit retryable catalog synchronization for published market datasets."""

from __future__ import annotations

from pathlib import Path

from trade_rl.catalog.contracts import ArtifactCatalog, ArtifactRecord
from trade_rl.catalog.service import market_dataset_registration
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)


def reconcile_market_dataset_catalog(
    artifact_root: str | Path,
    catalog: ArtifactCatalog,
) -> ArtifactRecord:
    """Validate an existing dataset artifact and register its immutable identity."""

    published = inspect_published_market_dataset_artifact(artifact_root)
    dataset = load_market_dataset_artifact(published.root)
    registration = market_dataset_registration(published, dataset)
    return catalog.register(registration)


__all__ = ["reconcile_market_dataset_catalog"]
