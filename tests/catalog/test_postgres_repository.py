from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from trade_rl.catalog import (
    ArtifactKind,
    ArtifactQuery,
    ArtifactRegistration,
    ArtifactStatus,
)
from trade_rl.catalog.postgres import CatalogConflictError, PostgresArtifactCatalog

_COLUMNS = (
    "artifact_digest",
    "artifact_kind",
    "schema_version",
    "dataset_id",
    "cache_key_digest",
    "cache_key",
    "metadata",
    "location",
    "size_bytes",
    "status",
    "created_at",
    "last_seen_at",
)


class FakeTransaction:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.rows: list[tuple[Any, ...]] = []
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(sql.split())
        self.executed.append((normalized, params))
        self.connection.executed.append((normalized, params))
        if "WHERE artifact_digest = %s OR" in normalized:
            assert params is not None
            digest, kind, cache_digest = params
            self.rows = [
                row
                for row in self.connection.artifacts.values()
                if row[0] == digest or (row[1] == kind and row[4] == cache_digest)
            ]
        elif normalized.startswith("INSERT INTO catalog_artifacts"):
            assert params is not None
            now = datetime(2026, 7, 21, tzinfo=UTC)
            row = (
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                json.loads(str(params[5])),
                json.loads(str(params[6])),
                params[7],
                params[8],
                params[9],
                now,
                now,
            )
            self.connection.artifacts[str(params[0])] = row
            self.rows = [row]
        elif normalized.startswith("UPDATE catalog_artifacts SET last_seen_at"):
            assert params is not None
            digest = str(params[0])
            old = self.connection.artifacts[digest]
            row = (*old[:-1], datetime(2026, 7, 22, tzinfo=UTC))
            self.connection.artifacts[digest] = row
            self.rows = [row]
        elif "WHERE artifact_kind = %s AND cache_key_digest = %s" in normalized:
            assert params is not None
            kind, cache_digest = params
            self.rows = [
                row
                for row in self.connection.artifacts.values()
                if row[1] == kind and row[4] == cache_digest
            ]
        elif normalized.startswith("SELECT artifact_digest"):
            values = list(self.connection.artifacts.values())
            if params:
                if "artifact_kind = %s" in normalized:
                    values = [row for row in values if row[1] == params[0]]
                if "dataset_id = %s" in normalized:
                    dataset = params[-2] if "LIMIT %s" in normalized else params[-1]
                    values = [row for row in values if row[3] == dataset]
                if "status = %s" in normalized:
                    status = params[-2] if "LIMIT %s" in normalized else params[-1]
                    values = [row for row in values if row[9] == status]
                limit = int(params[-1])
                values = values[:limit]
            self.rows = values
        elif normalized.startswith("INSERT INTO catalog_artifact_dependencies"):
            assert params is not None
            self.connection.dependencies.add(
                (str(params[0]), str(params[1]), str(params[2]))
            )
            self.rows = []
        else:
            self.rows = []

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        return None if not self.rows else self.rows[0]


class FakeConnection:
    def __init__(self) -> None:
        self.artifacts: dict[str, tuple[Any, ...]] = {}
        self.dependencies: set[tuple[str, str, str]] = set()
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


@pytest.fixture
def catalog() -> tuple[PostgresArtifactCatalog, FakeConnection]:
    connection = FakeConnection()
    return (
        PostgresArtifactCatalog(
            "postgresql://user:secret@localhost/trade_rl",
            connection_factory=lambda _: connection,
        ),
        connection,
    )


def _registration(
    digest: str = "a" * 64,
    *,
    cache_value: str = "dataset-a",
    kind: ArtifactKind = ArtifactKind.MARKET_DATASET,
) -> ArtifactRegistration:
    return ArtifactRegistration(
        artifact_digest=digest,
        artifact_kind=kind,
        schema_version="artifact_v1",
        cache_key={"identity": cache_value},
        metadata={"rows": 10},
        location="/tmp/artifact",
        size_bytes=123,
        dataset_id="b" * 64,
    )


def test_register_is_idempotent_and_find_uses_exact_cache_key(catalog) -> None:
    repository, connection = catalog
    registration = _registration()

    first = repository.register(registration)
    second = repository.register(registration)
    found = repository.find(ArtifactKind.MARKET_DATASET, {"identity": "dataset-a"})

    assert first.registration == registration
    assert second.registration == registration
    assert second.last_seen_at > first.last_seen_at
    assert found == second
    assert len(connection.artifacts) == 1
    assert all("%s" in sql or not params for sql, params in connection.executed)


def test_register_rejects_same_cache_key_with_different_digest(catalog) -> None:
    repository, _ = catalog
    repository.register(_registration())

    with pytest.raises(CatalogConflictError, match="cache key"):
        repository.register(_registration("c" * 64))


def test_list_filters_and_dependency_insert(catalog) -> None:
    repository, connection = catalog
    parent = repository.register(_registration())
    child = repository.register(
        _registration("c" * 64, cache_value="model-a", kind=ArtifactKind.MODEL)
    )

    records = repository.list(
        ArtifactQuery(artifact_kind=ArtifactKind.MODEL, status=ArtifactStatus.READY)
    )
    repository.add_dependency(
        parent.registration.artifact_digest,
        child.registration.artifact_digest,
        "source_dataset",
    )

    assert records == (child,)
    assert connection.dependencies == {("a" * 64, "c" * 64, "source_dataset")}


def test_missing_psycopg_dependency_has_focused_error(monkeypatch) -> None:
    from trade_rl.catalog import postgres

    monkeypatch.setattr(postgres, "_import_psycopg", lambda: None)
    repository = PostgresArtifactCatalog("postgresql://localhost/trade_rl")

    with pytest.raises(RuntimeError, match="postgres"):
        repository.health()
