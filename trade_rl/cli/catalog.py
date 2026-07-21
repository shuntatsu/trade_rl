"""Command-line operations for the PostgreSQL artifact catalog."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import TextIO

from trade_rl.catalog import (
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStatus,
)
from trade_rl.catalog.contracts import thaw_json
from trade_rl.catalog.postgres import PostgresArtifactCatalog

catalog_factory = PostgresArtifactCatalog


def _write_json(stdout: TextIO, payload: object) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stdout.write("\n")


def _database_url(args: argparse.Namespace) -> str:
    value = getattr(args, "database_url", None) or os.environ.get(
        "TRADE_RL_DATABASE_URL"
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "provide --database-url or set TRADE_RL_DATABASE_URL for catalog commands"
        )
    return value


def _json_object(value: str, *, field: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object")
    return parsed


def _record_payload(record: ArtifactRecord) -> dict[str, object]:
    registration = record.registration
    return {
        "artifact_digest": registration.artifact_digest,
        "artifact_kind": registration.artifact_kind.value,
        "cache_key": thaw_json(registration.cache_key),
        "cache_key_digest": registration.cache_key_digest,
        "created_at": record.created_at.isoformat(),
        "dataset_id": registration.dataset_id,
        "last_seen_at": record.last_seen_at.isoformat(),
        "location": registration.location,
        "metadata": thaw_json(registration.metadata),
        "schema_version": registration.schema_version,
        "size_bytes": registration.size_bytes,
        "status": registration.status.value,
    }


def _migrate(args: argparse.Namespace, stdout: TextIO) -> int:
    applied = catalog_factory(_database_url(args)).migrate()
    _write_json(
        stdout,
        {
            "applied_versions": list(applied),
            "schema": "artifact_catalog_migration_result_v1",
        },
    )
    return 0


def _health(args: argparse.Namespace, stdout: TextIO) -> int:
    payload = dict(catalog_factory(_database_url(args)).health())
    payload["schema"] = "artifact_catalog_health_v1"
    _write_json(stdout, payload)
    return 0


def _register(args: argparse.Namespace, stdout: TextIO) -> int:
    registration = ArtifactRegistration(
        artifact_digest=args.artifact_digest,
        artifact_kind=ArtifactKind(args.kind),
        schema_version=args.schema_version,
        cache_key=_json_object(args.cache_key_json, field="cache-key-json"),
        metadata=_json_object(args.metadata_json, field="metadata-json"),
        location=args.location,
        size_bytes=args.size_bytes,
        dataset_id=args.dataset_id,
        status=ArtifactStatus(args.status),
    )
    record = catalog_factory(_database_url(args)).register(registration)
    _write_json(
        stdout,
        {**_record_payload(record), "schema": "artifact_catalog_record_v1"},
    )
    return 0


def _find(args: argparse.Namespace, stdout: TextIO) -> int:
    record = catalog_factory(_database_url(args)).find(
        ArtifactKind(args.kind),
        _json_object(args.cache_key_json, field="cache-key-json"),
    )
    _write_json(
        stdout,
        {
            "artifact": None if record is None else _record_payload(record),
            "schema": "artifact_catalog_find_result_v1",
        },
    )
    return 0


def _list(args: argparse.Namespace, stdout: TextIO) -> int:
    query = ArtifactQuery(
        artifact_kind=None if args.kind is None else ArtifactKind(args.kind),
        dataset_id=args.dataset_id,
        status=None if args.status is None else ArtifactStatus(args.status),
        limit=args.limit,
    )
    records = catalog_factory(_database_url(args)).list(query)
    _write_json(
        stdout,
        {
            "artifacts": [_record_payload(record) for record in records],
            "schema": "artifact_catalog_list_result_v1",
        },
    )
    return 0


def _add_database_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        help="PostgreSQL DSN; defaults to TRADE_RL_DATABASE_URL",
    )


def add_catalog_parser(subparsers: argparse._SubParsersAction) -> None:
    catalog = subparsers.add_parser(
        "catalog", help="PostgreSQL research artifact catalog"
    )
    commands = catalog.add_subparsers(dest="catalog_command", required=True)

    migrate = commands.add_parser("migrate", help="apply catalog schema migrations")
    _add_database_url(migrate)
    migrate.set_defaults(handler=_migrate)

    health = commands.add_parser("health", help="verify catalog connectivity")
    _add_database_url(health)
    health.set_defaults(handler=_health)

    register = commands.add_parser("register", help="register one immutable artifact")
    _add_database_url(register)
    register.add_argument("--artifact-digest", required=True)
    register.add_argument(
        "--kind", choices=tuple(item.value for item in ArtifactKind), required=True
    )
    register.add_argument("--schema-version", required=True)
    register.add_argument("--cache-key-json", required=True)
    register.add_argument("--metadata-json", default="{}")
    register.add_argument("--location", required=True)
    register.add_argument("--size-bytes", type=int, required=True)
    register.add_argument("--dataset-id")
    register.add_argument(
        "--status",
        choices=tuple(item.value for item in ArtifactStatus),
        default=ArtifactStatus.READY.value,
    )
    register.set_defaults(handler=_register)

    find = commands.add_parser("find", help="find an exact reusable artifact")
    _add_database_url(find)
    find.add_argument(
        "--kind", choices=tuple(item.value for item in ArtifactKind), required=True
    )
    find.add_argument("--cache-key-json", required=True)
    find.set_defaults(handler=_find)

    listing = commands.add_parser("list", help="list recent catalog artifacts")
    _add_database_url(listing)
    listing.add_argument("--kind", choices=tuple(item.value for item in ArtifactKind))
    listing.add_argument("--dataset-id")
    listing.add_argument(
        "--status", choices=tuple(item.value for item in ArtifactStatus)
    )
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(handler=_list)


__all__ = ["add_catalog_parser", "catalog_factory"]
