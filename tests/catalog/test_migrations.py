from __future__ import annotations

import pytest

from trade_rl.catalog.migrations import apply_migrations, load_migrations


class RecordingCursor:
    def __init__(self, applied: dict[int, str] | None = None) -> None:
        self.applied = dict(applied or {})
        self.executed: list[tuple[str, object | None]] = []
        self._rows: list[tuple[object, ...]] = []

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object | None = None) -> None:
        self.executed.append((sql, params))
        if "SELECT version, checksum" in sql:
            self._rows = [
                (version, checksum)
                for version, checksum in sorted(self.applied.items())
            ]
        elif "INSERT INTO catalog_schema_migrations" in sql:
            assert isinstance(params, tuple)
            self.applied[int(params[0])] = str(params[2])

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class Transaction:
    def __init__(self, connection: "RecordingConnection") -> None:
        self.connection = connection

    def __enter__(self) -> None:
        self.connection.transactions += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is not None:
            self.connection.rollbacks += 1


class RecordingConnection:
    def __init__(self, applied: dict[int, str] | None = None) -> None:
        self.cursor_value = RecordingCursor(applied)
        self.transactions = 0
        self.rollbacks = 0

    def cursor(self) -> RecordingCursor:
        return self.cursor_value

    def transaction(self) -> Transaction:
        return Transaction(self)


def test_migrations_are_ordered_and_apply_once() -> None:
    migrations = load_migrations()
    assert [item.version for item in migrations] == sorted(
        item.version for item in migrations
    )
    connection = RecordingConnection()

    applied = apply_migrations(connection)
    reapplied = apply_migrations(connection)

    assert applied == tuple(item.version for item in migrations)
    assert reapplied == ()
    assert connection.transactions == 2
    assert any(
        "pg_advisory_xact_lock" in sql for sql, _ in connection.cursor_value.executed
    )


def test_applied_migration_checksum_mismatch_fails_closed() -> None:
    migration = load_migrations()[0]
    connection = RecordingConnection({migration.version: "f" * 64})

    with pytest.raises(ValueError, match="checksum"):
        apply_migrations(connection)

    assert connection.rollbacks == 1
