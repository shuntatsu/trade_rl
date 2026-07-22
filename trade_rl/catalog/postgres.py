"""psycopg-backed artifact catalog implementation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from trade_rl.catalog.contracts import (
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStatus,
    cache_key_digest,
    thaw_json,
)
from trade_rl.catalog.migrations import apply_migrations
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestAccessRecord

_ARTIFACT_COLUMNS = """
artifact_digest, artifact_kind, schema_version, dataset_id, cache_key_digest,
cache_key, metadata, location, size_bytes, status, created_at, last_seen_at
"""


class CatalogConflictError(ValueError):
    """Raised when one exact cache identity resolves to incompatible metadata."""


def _import_psycopg() -> Any | None:
    try:
        import psycopg
    except ImportError:
        return None
    return psycopg


def _default_connection_factory(database_url: str) -> Any:
    psycopg = _import_psycopg()
    if psycopg is None:
        raise RuntimeError(
            "PostgreSQL catalog support requires the 'postgres' optional dependency"
        )
    return psycopg.connect(database_url)


def _record_from_row(row: tuple[Any, ...]) -> ArtifactRecord:
    if len(row) != 12:
        raise ValueError("catalog artifact row has an unexpected shape")
    registration = ArtifactRegistration(
        artifact_digest=str(row[0]),
        artifact_kind=ArtifactKind(str(row[1])),
        schema_version=str(row[2]),
        dataset_id=None if row[3] is None else str(row[3]),
        cache_key=dict(row[5]),
        metadata=dict(row[6]),
        location=str(row[7]),
        size_bytes=int(row[8]),
        status=ArtifactStatus(str(row[9])),
    )
    if registration.cache_key_digest != str(row[4]):
        raise ValueError("catalog cache key digest does not match stored JSON")
    return ArtifactRecord(
        registration=registration,
        created_at=row[10],
        last_seen_at=row[11],
    )


def _registration_payload(registration: ArtifactRegistration) -> tuple[object, ...]:
    return (
        registration.artifact_digest,
        registration.artifact_kind.value,
        registration.schema_version,
        registration.dataset_id,
        registration.cache_key_digest,
        json.dumps(
            thaw_json(registration.cache_key),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        json.dumps(
            thaw_json(registration.metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        registration.location,
        registration.size_bytes,
        registration.status.value,
    )


class PostgresArtifactCatalog:
    """Open short-lived PostgreSQL connections for catalog operations."""

    def __init__(
        self,
        database_url: str,
        *,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be non-empty")
        self._database_url = database_url
        self._connection_factory = connection_factory or _default_connection_factory

    def _connect(self) -> Any:
        return self._connection_factory(self._database_url)

    def migrate(self) -> tuple[int, ...]:
        with self._connect() as connection:
            return apply_migrations(connection)

    def health(self) -> Mapping[str, object]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user, version()")
                database_row = cursor.fetchone()
                cursor.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM catalog_schema_migrations"
                )
                migration_row = cursor.fetchone()
        if database_row is None or migration_row is None:
            raise RuntimeError("PostgreSQL catalog health query returned no rows")
        return {
            "database": str(database_row[0]),
            "user": str(database_row[1]),
            "server_version": str(database_row[2]),
            "migration_version": int(migration_row[0]),
            "status": "ok",
        }

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT {_ARTIFACT_COLUMNS}
                        FROM catalog_artifacts
                        WHERE artifact_digest = %s
                           OR (artifact_kind = %s AND cache_key_digest = %s)
                        FOR UPDATE
                        """,
                        (
                            registration.artifact_digest,
                            registration.artifact_kind.value,
                            registration.cache_key_digest,
                        ),
                    )
                    existing_rows = cursor.fetchall()
                    if existing_rows:
                        if len(existing_rows) != 1:
                            raise CatalogConflictError(
                                "artifact digest and cache key resolve to different records"
                            )
                        existing = _record_from_row(existing_rows[0])
                        if existing.registration != registration:
                            if (
                                existing.registration.artifact_kind
                                == registration.artifact_kind
                                and existing.registration.cache_key_digest
                                == registration.cache_key_digest
                            ):
                                raise CatalogConflictError(
                                    "artifact cache key already belongs to a different artifact"
                                )
                            raise CatalogConflictError(
                                "artifact digest already exists with different metadata"
                            )
                        cursor.execute(
                            f"""
                            UPDATE catalog_artifacts
                            SET last_seen_at = CURRENT_TIMESTAMP
                            WHERE artifact_digest = %s
                            RETURNING {_ARTIFACT_COLUMNS}
                            """,
                            (registration.artifact_digest,),
                        )
                        updated = cursor.fetchone()
                        if updated is None:
                            raise RuntimeError(
                                "catalog idempotent update returned no row"
                            )
                        return _record_from_row(updated)
                    cursor.execute(
                        f"""
                        INSERT INTO catalog_artifacts (
                            artifact_digest, artifact_kind, schema_version, dataset_id,
                            cache_key_digest, cache_key, metadata, location, size_bytes,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                        RETURNING {_ARTIFACT_COLUMNS}
                        """,
                        _registration_payload(registration),
                    )
                    inserted = cursor.fetchone()
                    if inserted is None:
                        raise RuntimeError("catalog insert returned no row")
                    return _record_from_row(inserted)

    def find(
        self, artifact_kind: ArtifactKind, cache_key: Mapping[str, object]
    ) -> ArtifactRecord | None:
        resolved_kind = ArtifactKind(artifact_kind)
        digest = cache_key_digest(cache_key)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {_ARTIFACT_COLUMNS}
                    FROM catalog_artifacts
                    WHERE artifact_kind = %s AND cache_key_digest = %s
                    """,
                    (resolved_kind.value, digest),
                )
                row = cursor.fetchone()
        return None if row is None else _record_from_row(row)

    def list(
        self, query: ArtifactQuery = ArtifactQuery()
    ) -> tuple[ArtifactRecord, ...]:
        clauses: list[str] = []
        params: list[object] = []
        if query.artifact_kind is not None:
            clauses.append("artifact_kind = %s")
            params.append(query.artifact_kind.value)
        if query.dataset_id is not None:
            clauses.append("dataset_id = %s")
            params.append(query.dataset_id)
        if query.status is not None:
            clauses.append("status = %s")
            params.append(query.status.value)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        params.append(query.limit)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {_ARTIFACT_COLUMNS}
                    FROM catalog_artifacts
                    {where}
                    ORDER BY created_at DESC, artifact_digest
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def reserve_sealed_test_access(self, record: SealedTestAccessRecord) -> None:
        from trade_rl.catalog.postgres_sealed_test import (
            PostgresSealedTestReservationStore,
        )

        PostgresSealedTestReservationStore(
            self._database_url,
            connection_factory=self._connection_factory,
        ).reserve_sealed_test_access(record)

    def add_dependency(self, parent_digest: str, child_digest: str, role: str) -> None:
        ArtifactRegistration(
            artifact_digest=parent_digest,
            artifact_kind=ArtifactKind.RESEARCH_RUN,
            schema_version="validation_only",
            cache_key={"digest": parent_digest},
            metadata={},
            location="validation_only",
            size_bytes=0,
        )
        ArtifactRegistration(
            artifact_digest=child_digest,
            artifact_kind=ArtifactKind.RESEARCH_RUN,
            schema_version="validation_only",
            cache_key={"digest": child_digest},
            metadata={},
            location="validation_only",
            size_bytes=0,
        )
        if not isinstance(role, str) or not role.strip():
            raise ValueError("dependency role must be non-empty")
        if parent_digest == child_digest:
            raise ValueError("artifact dependency cannot reference itself")
        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO catalog_artifact_dependencies (
                            parent_digest, child_digest, dependency_role
                        )
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (parent_digest, child_digest, role),
                    )


__all__ = ["CatalogConflictError", "PostgresArtifactCatalog"]
