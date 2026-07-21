"""Transactional PostgreSQL schema migration support."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

_MIGRATION_RE = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z0-9_]+)\.sql$")
_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS catalog_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def load_migrations() -> tuple[Migration, ...]:
    migrations: list[Migration] = []
    root = files("trade_rl.catalog.sql")
    for entry in root.iterdir():
        match = _MIGRATION_RE.fullmatch(entry.name)
        if match is None:
            continue
        sql = entry.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    migrations.sort(key=lambda item: item.version)
    versions = [item.version for item in migrations]
    if not migrations or len(set(versions)) != len(versions):
        raise RuntimeError("catalog migrations must have unique ordered versions")
    return tuple(migrations)


def apply_migrations(connection: Any) -> tuple[int, ...]:
    migrations = load_migrations()
    applied_now: list[int] = []
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("trade_rl_catalog_migrations_v1",),
            )
            cursor.execute(_BOOTSTRAP_SQL)
            cursor.execute(
                "SELECT version, checksum FROM catalog_schema_migrations ORDER BY version"
            )
            applied = {int(row[0]): str(row[1]) for row in cursor.fetchall()}
            for migration in migrations:
                existing = applied.get(migration.version)
                if existing is not None:
                    if existing != migration.checksum:
                        raise ValueError(
                            f"catalog migration {migration.version} checksum mismatch"
                        )
                    continue
                cursor.execute(migration.sql)
                cursor.execute(
                    """
                    INSERT INTO catalog_schema_migrations (version, name, checksum)
                    VALUES (%s, %s, %s)
                    """,
                    (migration.version, migration.name, migration.checksum),
                )
                applied_now.append(migration.version)
    return tuple(applied_now)


__all__ = ["Migration", "apply_migrations", "load_migrations"]
