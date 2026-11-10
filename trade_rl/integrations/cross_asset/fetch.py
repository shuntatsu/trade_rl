"""Download orchestration with reproducible raw capture and revision detection."""

from __future__ import annotations

import hashlib
import socket
import time as time_module
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Protocol

from trade_rl.integrations.cross_asset.catalog import SeriesSpec
from trade_rl.integrations.cross_asset.parsers import (
    PARSERS,
    ParsedSeries,
    SourceBlockedError,
    SourceDataError,
    diff_observations,
)
from trade_rl.integrations.cross_asset.store import (
    MANIFEST_SCHEMA,
    USABLE_STATUSES,
    CrossAssetStore,
    format_utc,
    latest_records,
)

USER_AGENT = "trade_rl-cross-asset/1.0 (research; key-less public data)"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_EXTENSION = {"fred": "csv", "stooq": "csv", "yahoo": "json"}


class TransportError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Transport(Protocol):
    def get(self, url: str) -> bytes: ...


class UrllibTransport:
    """Plain HTTPS GET with bounded retries. Never sends credentials."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def get(self, url: str) -> bytes:
        if not url.startswith("https://"):
            raise TransportError("only https URLs are allowed")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(  # noqa: S310 - https enforced above
                    request, timeout=self.timeout_seconds
                ) as response:
                    payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise TransportError("response exceeds the size ceiling")
                return payload
            except urllib.error.HTTPError as error:
                last = error
                if error.code < 500 and error.code != 429:
                    raise TransportError(
                        f"HTTP {error.code}", status=error.code
                    ) from error
            except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
                last = error
            if attempt + 1 < self.max_attempts:
                time_module.sleep(self.backoff_seconds * (2**attempt))
        status = last.code if isinstance(last, urllib.error.HTTPError) else None
        raise TransportError(f"request failed: {last}", status=status) from last


def _base_record(spec: SeriesSpec, fetched_at: datetime) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "key": spec.key,
        "source": spec.source,
        "symbol": spec.symbol,
        "url": spec.url(),
        "fetched_at": format_utc(fetched_at),
    }


def _summary(parsed: ParsedSeries) -> dict[str, object]:
    return {
        "rows_total": parsed.rows_total,
        "missing_rows": parsed.missing_rows,
        "n_obs": int(parsed.dates.size),
        "first_obs": str(parsed.dates[0]) if parsed.dates.size else None,
        "last_obs": str(parsed.dates[-1]) if parsed.dates.size else None,
        "last_row_date": (
            parsed.last_row_date.isoformat() if parsed.last_row_date else None
        ),
    }


def fetch_all(
    store: CrossAssetStore,
    specs: Iterable[SeriesSpec],
    transport: Transport,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time_module.sleep,
    delay_seconds: float = 0.4,
) -> list[dict[str, object]]:
    """Fetch every spec once, appending one manifest record per series."""

    blocked_sources: set[str] = set()
    records: list[dict[str, object]] = []
    first_request = True
    for spec in specs:
        if spec.source in blocked_sources:
            record = _base_record(spec, now())
            record.update(
                status="unavailable",
                reason="skipped: source answered with a bot check earlier in this run",
            )
            store.append(record)
            records.append(record)
            continue
        if not first_request:
            sleep(delay_seconds)
        first_request = False
        record = _fetch_one(store, spec, transport, now)
        if record.get("blocked"):
            blocked_sources.add(spec.source)
            del record["blocked"]
        store.append(record)
        records.append(record)
    return records


def _fetch_one(
    store: CrossAssetStore,
    spec: SeriesSpec,
    transport: Transport,
    now: Callable[[], datetime],
) -> dict[str, object]:
    parser_name, parser = PARSERS[spec.source]
    try:
        payload = transport.get(spec.url())
    except TransportError as error:
        record = _base_record(spec, now())
        record.update(status="error", reason=f"transport: {error}")
        if error.status == 404:
            record["status"] = "unavailable"
        return record
    fetched_at = now()
    record = _base_record(spec, fetched_at)
    try:
        parsed = parser(payload, spec.symbol)
    except SourceBlockedError as error:
        record.update(
            status="unavailable",
            reason=(f"bot-detection challenge, not bypassed: {error}"),
            blocked=True,
        )
        return record
    except SourceDataError as error:
        record.update(status="error", reason=f"parse: {error}")
        return record
    if parsed.dates.size == 0:
        record.update(status="error", reason="parse: no valid observations")
        return record
    previous = latest_records(store.read_manifest()).get(spec.key)
    record.update(parser=parser_name, **_summary(parsed))
    record["bytes"] = len(payload)
    digest = hashlib.sha256(payload).hexdigest()
    record["sha256"] = digest
    if previous is not None and previous.get("sha256") == digest:
        record.update(status="unchanged", path=previous["path"], revision=None)
        return record
    path, _digest = store.write_raw(
        spec.source, spec.key, fetched_at, payload, extension=_EXTENSION[spec.source]
    )
    revision = None
    if previous is not None and previous.get("status") in USABLE_STATUSES:
        try:
            old = parser(store.read_raw(previous), spec.symbol)
            revision = diff_observations(old, parsed)
        except (OSError, ValueError):
            revision = {"unreadable_previous": True}
    record.update(status="ok", path=path, revision=revision)
    return record


__all__ = [
    "MAX_RESPONSE_BYTES",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "fetch_all",
]
