"""Searchable metadata catalog for immutable research artifacts."""

from typing import Any

from trade_rl.catalog import contracts as _contracts
from trade_rl.catalog import postgres as _postgres
from trade_rl.catalog.contracts import (
    ArtifactCatalog,
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStatus,
    cache_key_digest,
)
from trade_rl.catalog.postgres_sealed_test import PostgresSealedTestReservationStore
from trade_rl.domain.canonical_json import canonical_json_bytes

setattr(_contracts, "canonical_json_bytes", canonical_json_bytes)


def _reserve_sealed_test_access(self: Any, record: Any) -> None:
    """Compatibility delegate retained for existing workflow construction."""

    PostgresSealedTestReservationStore(
        self._database_url,
        connection_factory=self._connection_factory,
    ).reserve_sealed_test_access(record)


setattr(
    _postgres.PostgresArtifactCatalog,
    "reserve_sealed_test_access",
    _reserve_sealed_test_access,
)

__all__ = [
    "ArtifactCatalog",
    "ArtifactKind",
    "ArtifactQuery",
    "ArtifactRecord",
    "ArtifactRegistration",
    "ArtifactStatus",
    "PostgresSealedTestReservationStore",
    "cache_key_digest",
]
