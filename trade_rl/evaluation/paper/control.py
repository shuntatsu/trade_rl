"""Durable collection-cycle acknowledgements, separate from financial events."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


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

    def close(self) -> None:
        self.connection.close()
