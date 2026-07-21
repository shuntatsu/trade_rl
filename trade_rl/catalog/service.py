"""Optional environment-gated catalog registration helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from trade_rl.catalog.contracts import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRegistration,
)


def catalog_factory(database_url: str) -> Any:
    from trade_rl.catalog.postgres import PostgresArtifactCatalog

    return PostgresArtifactCatalog(database_url)


def register_artifact_if_configured(
    registration: ArtifactRegistration,
    *,
    environ: Mapping[str, str] | None = None,
) -> ArtifactRecord | None:
    environment = os.environ if environ is None else environ
    database_url = environment.get("TRADE_RL_DATABASE_URL")
    if database_url is None or not database_url.strip():
        return None
    return catalog_factory(database_url).register(registration)


def market_dataset_registration(published: Any, dataset: Any) -> ArtifactRegistration:
    raw_identity = getattr(dataset, "identity_payload_json", None)
    if not isinstance(raw_identity, str):
        raise ValueError("market dataset must expose canonical identity_payload_json")
    try:
        identity = json.loads(raw_identity)
    except json.JSONDecodeError as error:
        raise ValueError(
            "market dataset identity payload must be valid JSON"
        ) from error
    if not isinstance(identity, dict):
        raise ValueError("market dataset identity payload must be a JSON object")
    root = published.root.resolve()
    size_bytes = int(published.manifest_path.stat().st_size) + int(
        published.arrays_path.stat().st_size
    )
    return ArtifactRegistration(
        artifact_digest=published.artifact_digest,
        artifact_kind=ArtifactKind.MARKET_DATASET,
        schema_version=published.schema_version,
        dataset_id=dataset.dataset_id,
        cache_key=identity,
        metadata={
            "arrays_file": published.arrays_path.name,
            "manifest_file": published.manifest_path.name,
            "n_bars": int(dataset.n_bars),
            "n_features": int(dataset.n_features),
            "n_symbols": int(dataset.n_symbols),
            "symbols": list(dataset.symbols),
        },
        location=str(root),
        size_bytes=size_bytes,
    )


__all__ = [
    "catalog_factory",
    "market_dataset_registration",
    "register_artifact_if_configured",
]
