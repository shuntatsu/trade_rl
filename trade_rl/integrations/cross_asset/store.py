"""Raw-file store and append-only manifest for cross-asset downloads.

Layout under the (git-ignored) root::

    manifest.jsonl                         one canonical JSON record per fetch
    raw/<source>/<key>/<UTC stamp>_<sha8>.<ext>   exact bytes as served

A record's ``status`` is ``ok`` (new vintage), ``unchanged`` (same bytes as the
previous vintage; no new file), ``unavailable`` (source blocked or missing) or
``error``. Raw files are immutable; a revised series becomes a second file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.hashing import content_digest

MANIFEST_SCHEMA = "cross_asset_manifest_v1"
USABLE_STATUSES = frozenset({"ok", "unchanged"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


def format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0, tzinfo=None).isoformat() + "Z"


def parse_utc(text: str) -> datetime:
    if not text.endswith("Z"):
        raise ValueError(f"not a UTC timestamp: {text!r}")
    return datetime.fromisoformat(text[:-1]).replace(tzinfo=UTC)


def _component(value: str, label: str) -> str:
    if not _SAFE_COMPONENT.match(value):
        raise ValueError(f"unsafe {label} for a raw path: {value!r}")
    return value


class CrossAssetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def write_raw(
        self,
        source: str,
        key: str,
        fetched_at: datetime,
        payload: bytes,
        *,
        extension: str,
    ) -> tuple[str, str]:
        """Write immutable raw bytes and return (relative posix path, sha256)."""

        _component(source, "source")
        _component(key, "key")
        _component(extension, "extension")
        digest = hashlib.sha256(payload).hexdigest()
        stamp = format_utc(fetched_at).replace("-", "").replace(":", "")
        relative = f"raw/{source}/{key}/{stamp}_{digest[:8]}.{extension}"
        target = self.root / relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(
                    f"refusing to overwrite a different raw file: {relative}"
                )
            return relative, digest
        atomic_write_bytes(target, payload)
        return relative, digest

    def append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(
            dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_manifest(self) -> list[dict[str, object]]:
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, object]] = []
        text = self.manifest_path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"manifest line {number} is not JSON") from error
            if (
                not isinstance(record, dict)
                or record.get("schema") != MANIFEST_SCHEMA
                or not isinstance(record.get("key"), str)
                or not isinstance(record.get("fetched_at"), str)
            ):
                raise ValueError(f"manifest line {number} is not a valid record")
            records.append(record)
        return records

    def read_raw(self, record: Mapping[str, object]) -> bytes:
        path = record.get("path")
        if not isinstance(path, str):
            raise ValueError("record has no raw path")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError("raw path escapes the store root")
        return resolved.read_bytes()

    def verify_raw(self, record: Mapping[str, object]) -> bool:
        """True when the raw file exists and still matches the recorded digest."""

        try:
            payload = self.read_raw(record)
        except (OSError, ValueError):
            return False
        return hashlib.sha256(payload).hexdigest() == record.get("sha256")


def captured_at(record: Mapping[str, object]) -> str:
    """When the record's bytes were first captured (``fetched_at`` for old files).

    An ``unchanged`` re-check carries the capture time of the vintage it
    confirmed, never the time of the re-check.
    """

    return str(record.get("captured_at") or record["fetched_at"])


def latest_records(
    records: list[dict[str, object]], *, as_of: datetime | None = None
) -> dict[str, dict[str, object]]:
    """Return the latest usable vintage per series captured at or before ``as_of``."""

    cutoff = format_utc(as_of) if as_of is not None else None
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("status") not in USABLE_STATUSES:
            continue
        stamp = captured_at(record)
        if cutoff is not None and stamp > cutoff:
            continue
        key = str(record["key"])
        current = latest.get(key)
        if current is None or (stamp, str(record["fetched_at"])) >= (
            captured_at(current),
            str(current["fetched_at"]),
        ):
            latest[key] = record
    return latest


def manifest_digest(latest: Mapping[str, Mapping[str, object]]) -> str:
    """Digest of exactly which raw bytes feed a build (key and sha256 only)."""

    return content_digest(
        sorted([key, record["sha256"]] for key, record in latest.items())
    )
