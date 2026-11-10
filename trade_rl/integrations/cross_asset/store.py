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
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.hashing import content_digest

MANIFEST_SCHEMA = "cross_asset_manifest_v1"
USABLE_STATUSES = frozenset({"ok", "unchanged"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0, tzinfo=None).isoformat() + "Z"


def format_capture_utc(value: datetime) -> str:
    """Persist exact microseconds for new captures, including .000000 at S.

    Legacy second-only timestamps may have been truncated by older software.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return (
        value.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="microseconds")
        + "Z"
    )


def parse_utc(text: str) -> datetime:
    if not _UTC_STAMP.fullmatch(text):
        raise ValueError(f"not a canonical UTC timestamp: {text!r}")
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
        stamp = format_capture_utc(fetched_at).replace("-", "").replace(":", "")
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
        return _resolve_legacy_capture_times(records)

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


def _resolve_legacy_capture_times(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Give pre-``captured_at`` rows a capture time or fail closed.

    An ``ok`` row captured its own bytes when it was fetched. A legacy
    ``unchanged`` row was restamped by the re-check and is resolved to the ``ok``
    row that holds the same raw file; with no such row the capture time is
    unknown (a still-open session bar could be admitted) and the manifest is
    refused until the series is refetched.
    """

    origins: dict[tuple[object, object], str] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (record.get("path"), record.get("sha256"))
        stamp = str(record["fetched_at"])
        # A legacy raw file can be referenced by more than one "ok" entry.
        # The capture authority is the earliest occurrence, not the last one.
        if key not in origins or parse_utc(stamp) < parse_utc(origins[key]):
            origins[key] = stamp
    for number, record in enumerate(records, start=1):
        if record.get("status") not in USABLE_STATUSES or record.get("captured_at"):
            continue
        if record["status"] == "ok":
            record["captured_at"] = str(record["fetched_at"])
            continue
        origin = origins.get((record.get("path"), record.get("sha256")))
        if origin is None:
            raise ValueError(
                f"manifest record {number} ({record['key']}) is an unchanged re-check "
                "without captured_at and no originating ok record: capture time "
                "unknown; refetch the series"
            )
        record["captured_at"] = origin
    return records


def captured_at(record: Mapping[str, object]) -> str:
    """When the record's bytes were first captured.

    An ``unchanged`` re-check carries the capture time of the vintage it
    confirmed, never the time of the re-check.
    """

    return str(record["captured_at"])


def _stamp_eligible_at(stamp: str) -> datetime:
    """Legacy seconds are upper bounds, new six-digit fractions are exact."""

    instant = parse_utc(stamp)
    return instant if "." in stamp else instant + timedelta(seconds=1)


def _capture_eligible_at(record: Mapping[str, object]) -> datetime:
    """Conservative visibility of this vintage's raw bytes."""

    return _stamp_eligible_at(captured_at(record))


def latest_records(
    records: list[dict[str, object]], *, as_of: datetime | None = None
) -> dict[str, dict[str, object]]:
    """Return latest usable captured vintages under a precise UTC as-of cutoff."""

    if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
        raise ValueError("as_of must be timezone-aware")
    cutoff = as_of.astimezone(UTC) if as_of is not None else None
    latest: dict[str, dict[str, object]] = {}
    latest_clock: dict[str, tuple[datetime, datetime]] = {}
    for record in records:
        if record.get("status") not in USABLE_STATUSES:
            continue
        eligible = _capture_eligible_at(record)
        if cutoff is not None and eligible > cutoff:
            continue
        fetched = _stamp_eligible_at(str(record["fetched_at"]))
        # A future unchanged re-check proves nothing about status at this cutoff.
        if cutoff is not None and fetched > cutoff:
            continue
        key = str(record["key"])
        order = (eligible, fetched)
        if key not in latest_clock or order >= latest_clock[key]:
            latest[key] = record
            latest_clock[key] = order
    return latest


def manifest_digest(latest: Mapping[str, Mapping[str, object]]) -> str:
    """Digest the source bytes and the capture-time authority of every input."""

    return content_digest(
        {
            "schema": "cross_asset_input_provenance_v2",
            "capture_policy": "microsecond_exact_or_legacy_next_second",
            "records": sorted(
                [key, record["sha256"], captured_at(record)]
                for key, record in latest.items()
            ),
        }
    )
