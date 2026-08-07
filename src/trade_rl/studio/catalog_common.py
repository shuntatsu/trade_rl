"""Shared helpers for focused Studio catalogs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def stat_fingerprint(path: Path) -> tuple[bool, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (False, 0, 0)
    return (True, stat.st_size, stat.st_mtime_ns)


def fingerprint_identity(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
