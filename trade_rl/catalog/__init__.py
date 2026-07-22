"""Searchable metadata catalog for immutable research artifacts."""

from trade_rl.catalog import contracts as _contracts
from trade_rl.catalog.contracts import (
    ArtifactCatalog,
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStatus,
    cache_key_digest,
)
from trade_rl.domain.canonical_json import canonical_json_bytes

setattr(_contracts, "canonical_json_bytes", canonical_json_bytes)

__all__ = [
    "ArtifactCatalog",
    "ArtifactKind",
    "ArtifactQuery",
    "ArtifactRecord",
    "ArtifactRegistration",
    "ArtifactStatus",
    "cache_key_digest",
]
