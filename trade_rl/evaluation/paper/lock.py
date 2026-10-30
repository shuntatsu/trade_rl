"""The single advisory collector lock, shared by the collector and its watchdog."""

from __future__ import annotations

import secrets
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

LOCK_NAME = "collector.sqlite"
LOCK_PATIENCE_SECONDS = 1.5


class CollectorLockHeld(RuntimeError):
    """Another live process owns this collection; nothing was written or halted."""


def _lock_path(root: Path) -> Path:
    path = root / LOCK_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("collector lock must be a regular file")
    return path


_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


def _is_lock_busy(error: sqlite3.OperationalError) -> bool:
    """True only for SQLITE_BUSY/SQLITE_LOCKED (including their extended codes).

    Python 3.11+ exposes ``sqlite_errorcode``; the message is a fallback for an error
    without a code. Every other OperationalError is a real error, not contention.
    """
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        return code & 0xFF in {_SQLITE_BUSY, _SQLITE_LOCKED}
    return "database is locked" in str(error).lower()


def acquire_collector_lock(
    root: Path,
    *,
    patience_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> sqlite3.Connection:
    """Take the exclusive lock (process death releases it) or raise CollectorLockHeld.

    Only a busy/locked database is contention; any other sqlite error propagates.
    SQLite's zero-timeout ``BEGIN EXCLUSIVE`` can fail for *both* of two processes
    that start at the same instant (each briefly blocks the other's pending lock),
    which would leave a root with no owner. A busy result is therefore retried with
    a small random backoff for ``patience_seconds`` (default
    :data:`LOCK_PATIENCE_SECONDS`) before it is reported as held; a lock that a live
    collector really owns stays busy for that whole window, so the owner is never
    displaced and the loser is told, after at most that delay, that it is held.
    """
    patience = LOCK_PATIENCE_SECONDS if patience_seconds is None else patience_seconds
    deadline = monotonic() + patience
    path = _lock_path(root)
    while True:
        connection = sqlite3.connect(path, timeout=0, isolation_level=None)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            return connection
        except sqlite3.OperationalError as error:
            connection.close()
            if not _is_lock_busy(error):
                # An I/O failure, a corrupt or unreadable file, or anything else is
                # not proof that a live collector owns the root; surface the error.
                raise
            if monotonic() >= deadline:
                raise CollectorLockHeld(
                    "another collector holds this collection"
                ) from error
        sleep(0.02 + secrets.randbelow(80) / 1000)


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
