"""Durable collection-cycle acknowledgements, separate from financial events."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

ATTEMPT_OUTCOMES = ("failed", "unused")


class CollectionControl:
    """The caller holds the exclusive collector lock for this object's lifetime."""

    def __init__(self, root: Path, protocol_sha256: str) -> None:
        path = root / "collection-control.sqlite"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("collection control must be a regular file")
        new = not path.exists()
        if new and (root / "sources").exists():
            raise RuntimeError("interrupted or untracked collection sources")
        self.connection = sqlite3.connect(path, isolation_level=None)
        try:
            self.connection.execute("PRAGMA synchronous=FULL")
            if new:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("CREATE TABLE identity(digest TEXT NOT NULL)")
                self.connection.execute(
                    "INSERT INTO identity VALUES(?)", (protocol_sha256,)
                )
                self.connection.execute(
                    "CREATE TABLE cycles(id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, complete INTEGER NOT NULL CHECK(complete IN (0,1)))"
                )
                self.connection.commit()
            if self.connection.execute("SELECT digest FROM identity").fetchall() != [
                (protocol_sha256,)
            ]:
                raise ValueError("collection control protocol digest mismatch")
            # The registry of failed or unused captures and of skipped slots lives beside
            # the cycle acknowledgements: an unregistered failure cannot pass as an
            # allowed skip. Created for new and existing databases alike.
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS attempts("
                "id INTEGER PRIMARY KEY, cycle_id INTEGER NOT NULL, "
                "slot INTEGER NOT NULL, ordinal INTEGER NOT NULL, "
                "purpose TEXT NOT NULL, path TEXT NOT NULL UNIQUE, "
                "outcome TEXT NOT NULL CHECK(outcome IN ('failed','unused')), "
                "error_type TEXT NOT NULL, error TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS skips("
                "slot INTEGER PRIMARY KEY, cycle_id INTEGER NOT NULL, "
                "reason TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS cycles_incomplete_idx "
                "ON cycles(id) WHERE complete=0"
            )
            if (
                self.connection.execute(
                    "SELECT id FROM cycles WHERE complete=0 LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise RuntimeError("interrupted collection cycle; collection is halted")
        except BaseException:
            self.connection.close()
            raise

    def begin(self, at: datetime) -> int:
        if (
            self.connection.execute(
                "SELECT id FROM cycles WHERE complete=0 LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise RuntimeError("interrupted collection cycle; collection is halted")
        row = self.connection.execute(
            "INSERT INTO cycles(started_at,complete) VALUES(?,0)", (at.isoformat(),)
        )
        assert row.lastrowid is not None
        return row.lastrowid

    def finish(self, cycle: int) -> None:
        result = self.connection.execute(
            "UPDATE cycles SET complete=1 WHERE id=? AND complete=0", (cycle,)
        )
        if result.rowcount != 1:
            raise ValueError("collection cycle acknowledgement is invalid")

    def previous_cycle_start(self, cycle: int) -> datetime | None:
        row = self.connection.execute(
            "SELECT started_at FROM cycles WHERE id<? ORDER BY id DESC LIMIT 1",
            (cycle,),
        ).fetchone()
        return None if row is None else datetime.fromisoformat(row[0])

    def register_attempt(
        self,
        cycle: int,
        *,
        slot: int,
        purpose: str,
        path: str,
        outcome: str,
        error_type: str,
        error: str,
    ) -> int:
        """Durably register one failed or unused capture directory exactly once.

        Returns the attempt's ordinal inside its cycle. A directory registered twice is
        a bug, never a second attempt.
        """
        if outcome not in ATTEMPT_OUTCOMES:
            raise ValueError("unsupported capture attempt outcome")
        ordinal = self.connection.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM attempts WHERE cycle_id=?",
            (cycle,),
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO attempts(cycle_id,slot,ordinal,purpose,path,outcome,"
            "error_type,error) VALUES(?,?,?,?,?,?,?,?)",
            (cycle, slot, ordinal, purpose, path, outcome, error_type, error[:500]),
        )
        return int(ordinal)

    def register_skips(
        self, cycle: int, slots: Sequence[int], reason: str
    ) -> list[int]:
        """Register skipped slots (first registration of a slot wins); return all."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for slot in slots:
                self.connection.execute(
                    "INSERT OR IGNORE INTO skips(slot,cycle_id,reason) VALUES(?,?,?)",
                    (slot, cycle, reason),
                )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return self.skipped_slots()

    def skipped_slots(self) -> list[int]:
        found = self.connection.execute("SELECT slot FROM skips ORDER BY slot")
        return [int(row[0]) for row in found.fetchall()]

    def close(self) -> None:
        self.connection.close()


def incomplete_cycle_exists(root: Path) -> bool:
    """Read-only probe for a begun-but-unacknowledged cycle (no schema creation).

    Owned here so the cycles table layout has one definition. Raises on a symlink,
    an unreadable database or a missing ``cycles`` table; absent file means False.
    """
    path = root / "collection-control.sqlite"
    if path.is_symlink():
        raise ValueError("collection control must be a regular file")
    if not path.exists():
        return False
    with contextlib.closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=0)
    ) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM cycles WHERE complete=0 LIMIT 1"
            ).fetchone()
            is not None
        )
