"""Raw-file store and append-only manifest for cross-asset downloads.

Layout under the (git-ignored) root::

    manifest.jsonl                         one canonical JSON record per fetch
    raw/<source>/<key>/<UTC stamp>_<sha8>.<ext>   exact bytes as served

A record's ``status`` is ``ok`` (new vintage), ``unchanged`` (same bytes as the
previous vintage; no new file), ``unavailable`` (source blocked or missing) or
``error``. Raw files are immutable; a revised series becomes a second file.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest, open_regular_binary

MANIFEST_SCHEMA = "cross_asset_manifest_v1"
USABLE_STATUSES = frozenset({"ok", "unchanged"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_SHA256_LOWER_HEX = re.compile(r"^[0-9a-f]{64}$")
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


def _reject_linked_parent(path: Path) -> None:
    """Refuse a statically redirected destination directory chain.

    This is a static check, not a concurrent hostile-directory-swap guarantee.
    """

    absolute = path.absolute()
    for part in (absolute, *absolute.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError(
                "cross-asset raw parent must not contain a symlink or junction"
            )
        if part.exists() and not part.is_dir():
            raise ValueError("cross-asset raw parent must contain only directories")


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
        # Check the existing destination chain before observing/creating files.
        # A static linked root or ancestor would otherwise redirect publication
        # outside the requested raw store on the cache-miss path.
        _reject_linked_parent(target.parent)
        # The filename carries only 32 bits of the payload hash. Equal prefix
        # and capture timestamp do not prove equal full bytes; never follow a
        # symlink or accept a conflicting existing full digest.
        if target.exists() or target.is_symlink():
            if file_digest(target, field="cross-asset raw file") != digest:
                raise ValueError(
                    f"refusing to overwrite a different raw file: {relative}"
                )
            return relative, digest

        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck created directories before private staging. This still does
        # not make path operations atomic against a hostile concurrent swap.
        _reject_linked_parent(target.parent)
        # Atomic replacement would overwrite a competing capture that arrived
        # after the existence check. Each writer stages privately and claims the
        # destination with a same-filesystem hard link (exclusive on POSIX).
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.part-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                # Accept only the exact already-published winner; a 32-bit
                # filename-prefix collision must not replace earlier evidence.
                if file_digest(target, field="cross-asset raw file") != digest:
                    raise ValueError(
                        f"refusing to overwrite a different raw file: {relative}"
                    ) from None
            else:
                if os.name != "nt":
                    directory = os.open(target.parent, os.O_RDONLY)
                    try:
                        try:
                            os.fsync(directory)
                        except OSError as error:
                            # macOS and some POSIX filesystems cannot fsync a
                            # directory, even after exclusive link succeeded.
                            # Staged file bytes were fsynced before publication;
                            # directory durability is not guaranteed here.
                            if error.errno not in (
                                errno.EINVAL,
                                errno.ENOTSUP,
                                errno.EOPNOTSUPP,
                            ):
                                raise
                    finally:
                        os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return relative, digest

    def append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(
            dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        # Unavailable/error captures append manifest rows without write_raw().
        # Apply the same linked-ancestor boundary to these untrusted paths.
        _reject_linked_parent(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_linked_parent(self.root)
        destination = self.manifest_path
        if destination.is_symlink() or destination.is_junction():
            raise ValueError("cross-asset manifest must not be a symlink or junction")
        # O_NOFOLLOW closes the leaf-symlink swap between the static check
        # and opening the manifest on supporting platforms.
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(destination, flags, 0o666)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    "cross-asset manifest must not be a symlink or junction"
                ) from error
            raise
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("cross-asset manifest must be a regular file")
            if metadata.st_nlink != 1:
                raise ValueError(
                    "cross-asset manifest must not have multiple hard links"
                )
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as handle:
                descriptor = -1
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def read_manifest(self) -> list[dict[str, object]]:
        _reject_linked_parent(self.root)
        destination = self.manifest_path
        if not destination.exists() and not destination.is_symlink():
            return []
        # A pre-existing FIFO/device may block in os.open(O_RDONLY) before
        # the no-follow descriptor helper can fstat it. Reject statically
        # before entering the opener; later hostile swaps remain out of scope.
        if not destination.is_file():
            raise ValueError("cross-asset manifest must be a regular file")
        records: list[dict[str, object]] = []
        # The checked descriptor guards the entire read from leaf symlinks
        # and, on supported POSIX, pins directory ancestors during opening.
        with open_regular_binary(destination, field="cross-asset manifest") as handle:
            text = handle.read().decode("utf-8")
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
        relative = Path(path)
        # Manifest raw paths are store-relative; resolving first could follow a
        # symlinked configured root and make a foreign tree appear in-bounds.
        if not relative.parts or relative.anchor or ".." in relative.parts:
            raise ValueError("raw path must be relative with no parent traversal")
        # Open the literal path by the shared no-follow regular-file helper,
        # keeping descriptor-pinned ancestor protection on supported POSIX.
        # Portable fallback retains static, not atomic, ancestor checks.
        with open_regular_binary(
            self.root / relative, field="cross-asset raw file"
        ) as handle:
            return handle.read()

    def verify_raw(self, record: Mapping[str, object]) -> bool:
        """True when the raw file exists and still matches the recorded digest."""

        try:
            payload = self.read_raw(record)
        except (OSError, ValueError):
            return False
        return hashlib.sha256(payload).hexdigest() == record.get("sha256")


def _raw_identity(record: Mapping[str, object], number: int) -> tuple[str, str, str]:
    """Bind each immutable raw origin to exactly one catalog series."""

    key = record.get("key")
    path = record.get("path")
    digest = record.get("sha256")
    if (
        not isinstance(key, str)
        or not key.strip()
        or key != key.strip()
        or not isinstance(path, str)
        or not path.strip()
        or path != path.strip()
        or not isinstance(digest, str)
        or _SHA256_LOWER_HEX.fullmatch(digest) is None
    ):
        raise ValueError(f"manifest record {number} has incomplete raw provenance")
    return key, path, digest


def _resolve_legacy_capture_times(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Give pre-``captured_at`` rows a capture time or fail closed.

    New records with explicit ``captured_at`` preserve it; legacy ``ok``
    records sharing a series key, raw path and digest use the earliest capture.
    Legacy ``unchanged`` records inherit that origin, not their later recheck.
    Without a complete raw path/digest or any originating ``ok`` record, the
    historical capture time is unknown and the record is refused.
    """

    origins: dict[tuple[str, str, str], str] = {}
    for number, record in enumerate(records, start=1):
        if record.get("status") != "ok":
            continue
        origin_id = _raw_identity(record, number)
        # Prefer an already-recorded first capture over the time of a later
        # refetch of those same bytes in a mixed legacy/modern manifest.
        stamp = str(record.get("captured_at") or record["fetched_at"])
        # A legacy raw file can be referenced by more than one "ok" entry.
        # The capture authority is the earliest occurrence, not the last one.
        if origin_id not in origins or parse_utc(stamp) < parse_utc(origins[origin_id]):
            origins[origin_id] = stamp
    for number, record in enumerate(records, start=1):
        if record.get("status") not in USABLE_STATUSES:
            continue
        origin_id = _raw_identity(record, number)
        if record.get("captured_at"):
            continue
        origin = origins.get(origin_id)
        if record["status"] == "ok":
            # Repeated legacy "ok" rows can reference the same immutable raw
            # file. The first recorded capture is the authority for its bytes.
            record["captured_at"] = (
                origin if origin is not None else str(record["fetched_at"])
            )
            continue
        if origin is None:
            raise ValueError(
                f"manifest record {number} ({record['key']}) is an unchanged re-check "
                "without captured_at and no originating ok record: capture time "
                "unknown; refetch the series"
            )
        # A later first capture cannot justify an earlier unchanged recheck.
        # Legacy second-only stamps have unknown fractional time: compare their
        # conservative latest-possible instants rather than treating :00 as exact.
        if _stamp_eligible_at(origin) > _stamp_eligible_at(str(record["fetched_at"])):
            raise ValueError(
                f"manifest record {number} ({record['key']}) has capture origin "
                f"{origin} after fetched_at {record['fetched_at']}"
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
