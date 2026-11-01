"""The single permanent collection-failure marker writer (no market or journal I/O)."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes

FAILURE_NAME = "collection-failure.json"


def _now() -> datetime:
    return datetime.now(UTC)


def write_collection_failure(
    root: Path,
    *,
    protocol_sha256: str,
    error_type: str,
    error: str,
    extra: dict[str, Any] | None = None,
    clock: Callable[[], datetime] = _now,
) -> dict[str, Any] | None:
    """Create the permanent failure marker once; return None if one already exists.

    The marker is the single operational fact that rejects a screen. It is written
    exclusively and fsynced, never replaced, so the first recorded cause is kept.
    The collector, the watchdog and the termination hooks all use this one writer.
    """
    failure = Path(root) / FAILURE_NAME
    if failure.exists() or failure.is_symlink():
        return None
    record: dict[str, Any] = dict(
        schema="paper_collection_failure_v1",
        protocol_sha256=protocol_sha256,
        recorded_at=clock().isoformat(),
        error_type=error_type,
        error=error[:2000],
        production_eligible=False,
    )
    if extra:
        record.update(extra)
    try:
        with failure.open("xb") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return None
    return record


__all__ = ["FAILURE_NAME", "write_collection_failure"]
