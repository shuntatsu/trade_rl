"""The single advisory collector lock, shared by the collector and its watchdog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

LOCK_NAME = "collector.sqlite"


class CollectorLockHeld(RuntimeError):
    """Another live process owns this collection; nothing was written or halted."""


def _lock_path(root: Path) -> Path:
    path = root / LOCK_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("collector lock must be a regular file")
    return path


def acquire_collector_lock(root: Path) -> sqlite3.Connection:
    """Take the exclusive lock (process death releases it) or raise CollectorLockHeld."""
    connection = sqlite3.connect(_lock_path(root), timeout=0, isolation_level=None)
    try:
        connection.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as error:
        connection.close()
        raise CollectorLockHeld("another collector holds this collection") from error
    return connection


def collector_lock_held(root: Path) -> bool:
    """Non-destructive probe: True while a live collector holds the lock.

    Never creates the lock file and releases it at once when it was free, so it
    writes nothing and cannot invalidate a screen; a collector starting in the
    same instant is told the lock is held and exits without recording a failure.
    """
    path = _lock_path(root)
    if not path.exists():
        return False
    try:
        connection = acquire_collector_lock(root)
    except CollectorLockHeld:
        return True
    connection.rollback()
    connection.close()
    return False
