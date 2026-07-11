"""SQLite-backed audit events and replay protection for serving requests."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class AuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS claimed_requests (
                    request_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    claimed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    request_id TEXT,
                    model_version TEXT,
                    bundle_digest TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def claim_request(self, request_id: str, payload_hash: str) -> bool:
        if not request_id or not payload_hash:
            raise ValueError("request_id and payload_hash are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_hash FROM claimed_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is not None:
                if row["payload_hash"] != payload_hash:
                    raise ValueError(
                        "request_id was already used with a different payload"
                    )
                return False
            connection.execute(
                "INSERT INTO claimed_requests(request_id, payload_hash, claimed_at) "
                "VALUES (?, ?, ?)",
                (request_id, payload_hash, time.time()),
            )
            return True

    def append_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        request_id: str | None = None,
        model_version: str | None = None,
        bundle_digest: str | None = None,
    ) -> None:
        if not event_type:
            raise ValueError("event_type is required")
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_events(event_type, request_id, model_version, "
                "bundle_digest, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_type,
                    request_id,
                    model_version,
                    bundle_digest,
                    payload_json,
                    time.time(),
                ),
            )

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_type, request_id, model_version, bundle_digest, "
                "payload_json, created_at FROM audit_events "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "request_id": row["request_id"],
                "model_version": row["model_version"],
                "bundle_digest": row["bundle_digest"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
