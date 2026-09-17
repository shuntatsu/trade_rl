"""Protocol-bound, transactional, append-only paper event evidence.

This module assigns no financial meaning to events. The paper engine owns
decisions and composes the existing canonical account; this is persistence only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes

_PROTOCOL = "paper_journal_protocol_v1"
_EVENT = "paper_journal_event_v1"
_DDL = """
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY CHECK(sequence >= 0),
    event_key TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL UNIQUE,
    body BLOB NOT NULL
);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'paper events are immutable'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'paper events are immutable'); END;
CREATE TRIGGER events_contiguous BEFORE INSERT ON events
WHEN NEW.sequence != (SELECT COALESCE(MAX(sequence) + 1, 0) FROM events)
  OR EXISTS(SELECT 1 FROM events WHERE event_key=NEW.event_key OR sha256=NEW.sha256)
BEGIN SELECT RAISE(ABORT, 'paper events must append contiguously'); END;
"""


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("paper digest must be a lowercase SHA-256")


def _identity(key: str, kind: str, payload: object) -> None:
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", key) is None:
        raise ValueError("paper idempotency key is invalid")
    if not isinstance(kind, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", kind) is None:
        raise ValueError("paper event kind is invalid")
    if not isinstance(payload, dict):
        raise ValueError("paper event payload must be an object")


def _record(row: sqlite3.Row) -> dict[str, Any]:
    body = row["body"]
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("paper event digest mismatch")
    event = json.loads(body)
    if (
        not isinstance(event, dict)
        or set(event) != {"schema", "sequence", "parent", "key", "kind", "payload"}
        or event.get("schema") != _EVENT
    ):
        raise ValueError("paper event schema mismatch")
    if (
        type(event["sequence"]) is not int
        or event["sequence"] != row["sequence"]
        or event["key"] != row["event_key"]
    ):
        raise ValueError("paper event row identity mismatch")
    _identity(event["key"], event["kind"], event["payload"])
    _digest(event["parent"])
    if canonical_json_bytes(event) != body:
        raise ValueError("paper event bytes are not canonical")
    return {"sha256": row["sha256"], "event": event}


class PaperJournal:
    """A committed event prefix; reopening validates its entire hash chain."""

    @staticmethod
    def initialize(root: str | Path, protocol: dict[str, Any]) -> str:
        if not isinstance(protocol, dict):
            raise ValueError("paper protocol must be an object")
        raw = canonical_json_bytes({"schema": _PROTOCOL, "protocol": protocol})
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=False)
        database = sqlite3.connect(destination / "journal.sqlite", isolation_level=None)
        try:
            database.execute("PRAGMA journal_mode=DELETE")
            database.execute("PRAGMA synchronous=FULL")
            database.executescript(_DDL)
        finally:
            database.close()
        with (destination / "protocol.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(raw).hexdigest()

    def __init__(self, root: str | Path, *, expected_protocol_sha256: str) -> None:
        _digest(expected_protocol_sha256)
        self.root = Path(root).absolute()
        self.protocol_sha256 = expected_protocol_sha256
        # A read-only connection cannot recover a hot rollback journal. Verify
        # the protocol first, then allow SQLite to restore its committed prefix.
        with self._connection(writable=True) as connection:
            connection.execute("SELECT sequence FROM events LIMIT 1").fetchone()
        self.events()

    def _protocol(self) -> None:
        path = self.root / "protocol.json"
        if (
            self.root.is_symlink()
            or not self.root.is_dir()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("paper protocol is missing or not a regular file")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.protocol_sha256:
            raise ValueError("paper protocol digest mismatch")
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "protocol"}
            or value["schema"] != _PROTOCOL
            or not isinstance(value["protocol"], dict)
            or canonical_json_bytes(value) != raw
        ):
            raise ValueError("paper protocol schema or canonical bytes mismatch")

    @contextmanager
    def _connection(self, *, writable: bool) -> Iterator[sqlite3.Connection]:
        self._protocol()
        path = self.root / "journal.sqlite"
        if path.is_symlink() or not path.is_file():
            raise ValueError("paper database is missing or not a regular file")
        mode = "rw" if writable else "ro"
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode={mode}", uri=True, isolation_level=None, timeout=5
        )
        connection.row_factory = sqlite3.Row
        try:
            if writable:
                connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    def events(self) -> list[dict[str, Any]]:
        """Read and verify the contiguous committed prefix without creating files."""
        result: list[dict[str, Any]] = []
        parent = self.protocol_sha256
        with self._connection(writable=False) as connection:
            for row in connection.execute("SELECT * FROM events ORDER BY sequence"):
                record = _record(row)
                event = record["event"]
                if event["sequence"] != len(result) or event["parent"] != parent:
                    raise ValueError(
                        "paper event chain is gapped or has a wrong parent"
                    )
                result.append(record)
                parent = record["sha256"]
        return result

    def append(
        self,
        key: str,
        kind: str,
        payload: dict[str, Any],
        *,
        expected_parent: str,
    ) -> dict[str, Any]:
        """Compare-and-append once; an exact retry returns its original record."""
        _identity(key, kind, payload)
        _digest(expected_parent)
        payload = json.loads(canonical_json_bytes(payload))
        with self._connection(writable=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._protocol()
            existing = connection.execute(
                "SELECT * FROM events WHERE event_key=?", (key,)
            ).fetchone()
            if existing is not None:
                record = _record(existing)
                event = record["event"]
                if (
                    event["parent"] != expected_parent
                    or event["kind"] != kind
                    or canonical_json_bytes(event["payload"])
                    != canonical_json_bytes(payload)
                ):
                    raise ValueError(
                        "paper idempotency key was reused with different contents"
                    )
                connection.commit()
                return record
            tail = connection.execute(
                "SELECT * FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            last = None if tail is None else _record(tail)
            parent = self.protocol_sha256 if last is None else last["sha256"]
            if parent != expected_parent:
                raise ValueError("paper append parent is stale")
            sequence = 0 if last is None else last["event"]["sequence"] + 1
            event = dict(
                schema=_EVENT,
                sequence=sequence,
                parent=parent,
                key=key,
                kind=kind,
                payload=payload,
            )
            body = canonical_json_bytes(event)
            digest = hashlib.sha256(body).hexdigest()
            connection.execute(
                "INSERT INTO events(sequence,event_key,sha256,body) VALUES(?,?,?,?)",
                (sequence, key, digest, body),
            )
            connection.commit()
            return {"sha256": digest, "event": event}
