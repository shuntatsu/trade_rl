"""Deterministic Binance Vision cache planning, validation, and synchronization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
)
from trade_rl.integrations.binance.vision import (
    plan_vision_kline_urls,
    vision_funding_url,
)

_VISION_PREFIX = "https://data.binance.vision/data/"
MISSING_ARCHIVE_SCHEMA = "binance_vision_missing_archive_v1"
_MISSING_ARCHIVE_FIELDS = frozenset(
    {"downloader", "http_status", "observed_at", "schema_version", "url"}
)


def missing_archive_evidence_path(cache_path: Path) -> Path:
    """Sidecar that records an HTTP 404 for the archive that would live at ``cache_path``.

    Consumers do not ignore it: the cache inspectors above look only at ``.bin`` files,
    but forward acquisition deliberately reports any unplanned cache member, this
    sidecar included, as ``archives_unexpected`` (fail closed). The trend-stop study is
    the consumer that reads it.
    """

    return cache_path.with_suffix(".missing.json")


def write_missing_archive_evidence(
    cache_path: Path, *, url: str, downloader: str, observed_at: datetime | None = None
) -> None:
    """Record that the venue answered HTTP 404 for ``url`` (URL, status, time)."""

    moment = (
        datetime.now(UTC)
        if observed_at is None
        else _aware_utc(observed_at, field="observed_at")
    )
    evidence = {
        "downloader": downloader,
        "http_status": 404,
        "observed_at": moment.isoformat(),
        "schema_version": MISSING_ARCHIVE_SCHEMA,
        "url": url,
    }
    path = missing_archive_evidence_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_missing_archive_evidence(
    cache_path: Path, url: str
) -> dict[str, object] | None:
    """Return the recorded HTTP 404 evidence for ``url``, or ``None`` if there is none.

    Malformed or mismatching evidence raises ``BinanceTransportError``: it is never
    read as a 404.
    """

    path = missing_archive_evidence_path(cache_path)
    if not path.is_file():
        return None
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BinanceTransportError(
            f"missing-archive evidence is unreadable: {path}"
        ) from error
    if (
        not isinstance(evidence, dict)
        or set(evidence) != _MISSING_ARCHIVE_FIELDS
        or evidence["schema_version"] != MISSING_ARCHIVE_SCHEMA
        or evidence["url"] != url
        or isinstance(evidence["http_status"], bool)
        or evidence["http_status"] != 404
        or not isinstance(evidence["observed_at"], str)
        or not isinstance(evidence["downloader"], str)
    ):
        raise BinanceTransportError(f"missing-archive evidence is invalid: {path}")
    try:
        observed = datetime.fromisoformat(evidence["observed_at"])
    except ValueError as error:
        raise BinanceTransportError(
            f"missing-archive evidence time is invalid: {path}"
        ) from error
    if observed.tzinfo is None:
        raise BinanceTransportError(
            f"missing-archive evidence time lacks a timezone: {path}"
        )
    return dict(evidence)


def validate_cached_vision_payload(url: str, cache_path: Path) -> bytes:
    """Load one Vision archive only when its sidecar proves exact content."""

    evidence_path = cache_path.with_suffix(".json")
    if not evidence_path.is_file():
        raise BinanceTransportError(
            f"cached Binance Vision archive lacks content evidence: {cache_path}"
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BinanceTransportError(
            "cached Binance Vision evidence is invalid"
        ) from error
    expected = {
        "acquired_at",
        "downloader",
        "etag",
        "last_modified",
        "schema_version",
        "sha256",
        "size_bytes",
        "url",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise BinanceTransportError("cached Binance Vision evidence fields are invalid")
    payload = cache_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if (
        evidence.get("schema_version") != "binance_vision_raw_cache_v1"
        or evidence.get("url") != url
        or evidence.get("size_bytes") != len(payload)
        or evidence.get("sha256") != digest
    ):
        raise BinanceTransportError(
            f"cached Binance Vision archive content digest or size mismatch: {cache_path}"
        )
    return payload


class _VisionArchiveTransport(Protocol):
    cache_root: Path | None

    def _request_bytes(self, url: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class BinanceVisionCachePlan:
    market: BinanceMarket
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    urls: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "end_time": self.end_time.isoformat(),
            "intervals": list(self.intervals),
            "market": self.market.value,
            "planned_count": len(self.urls),
            "start_time": self.start_time.isoformat(),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class BinanceVisionCacheReport:
    cache_root: Path
    planned_count: int
    cached_count: int
    downloaded_count: int
    missing_urls: tuple[str, ...]
    empty_urls: tuple[str, ...]
    invalid_urls: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_urls and not self.empty_urls and not self.invalid_urls

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_root": str(self.cache_root),
            "cached_count": self.cached_count,
            "complete": self.complete,
            "downloaded_count": self.downloaded_count,
            "empty_count": len(self.empty_urls),
            "invalid_count": len(self.invalid_urls),
            "missing_count": len(self.missing_urls),
            "planned_count": self.planned_count,
        }


def _ordered_nonempty(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in values))
    if not result or any(not value for value in result):
        raise ValueError(f"{field} must contain non-empty values")
    return result


def _official_urls(values: Sequence[str]) -> tuple[str, ...]:
    urls = _ordered_nonempty(values, field="urls")
    if any(not url.startswith(_VISION_PREFIX) for url in urls):
        raise ValueError("cache URL must be an official Binance Vision URL")
    return urls


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def plan_binance_vision_cache(
    *,
    market: BinanceMarket | str,
    symbols: Sequence[str],
    intervals: Sequence[str],
    start_time: datetime,
    end_time: datetime,
) -> BinanceVisionCachePlan:
    """Plan every immutable Vision archive required by one research range."""

    resolved_market = BinanceMarket(market)
    resolved_symbols = _ordered_nonempty(symbols, field="symbols")
    resolved_intervals = _ordered_nonempty(intervals, field="intervals")
    resolved_start = _aware_utc(start_time, field="start_time")
    resolved_end = _aware_utc(end_time, field="end_time")
    if resolved_end <= resolved_start:
        raise ValueError("end_time must be later than start_time")

    urls: list[str] = []
    for symbol in resolved_symbols:
        for interval in resolved_intervals:
            urls.extend(
                plan_vision_kline_urls(
                    resolved_market,
                    symbol,
                    interval,
                    resolved_start,
                    resolved_end,
                )
            )

    if resolved_market is not BinanceMarket.SPOT:
        month = resolved_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while _next_month(month) <= resolved_end:
            for symbol in resolved_symbols:
                urls.append(vision_funding_url(resolved_market, symbol, month))
            month = _next_month(month)

    return BinanceVisionCachePlan(
        market=resolved_market,
        symbols=resolved_symbols,
        intervals=resolved_intervals,
        start_time=resolved_start,
        end_time=resolved_end,
        urls=tuple(dict.fromkeys(urls)),
    )


def vision_cache_path(cache_root: str | Path, url: str) -> Path:
    """Return the path used by ``BinancePublicTransport`` for one Vision URL."""

    if not url.startswith(_VISION_PREFIX):
        raise ValueError("cache URL must be an official Binance Vision URL")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return Path(cache_root) / digest[:2] / f"{digest}.bin"


def inspect_binance_vision_urls(
    urls: Sequence[str],
    *,
    cache_root: str | Path,
) -> BinanceVisionCacheReport:
    """Validate an arbitrary authored set of official Vision URLs in shared cache."""

    resolved_urls = _official_urls(urls)
    root = Path(cache_root)
    cached: list[str] = []
    missing: list[str] = []
    empty: list[str] = []
    invalid: list[str] = []
    for url in resolved_urls:
        path = vision_cache_path(root, url)
        if not path.is_file():
            missing.append(url)
        elif path.stat().st_size <= 0:
            empty.append(url)
        else:
            try:
                validate_cached_vision_payload(url, path)
            except (BinanceTransportError, OSError):
                invalid.append(url)
            else:
                cached.append(url)
    return BinanceVisionCacheReport(
        cache_root=root,
        planned_count=len(resolved_urls),
        cached_count=len(cached),
        downloaded_count=0,
        missing_urls=tuple(missing),
        empty_urls=tuple(empty),
        invalid_urls=tuple(invalid),
    )


def inspect_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    cache_root: str | Path,
) -> BinanceVisionCacheReport:
    return inspect_binance_vision_urls(plan.urls, cache_root=cache_root)


def require_complete_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    cache_root: str | Path,
) -> BinanceVisionCacheReport:
    report = inspect_binance_vision_cache(plan, cache_root=cache_root)
    if not report.complete:
        raise FileNotFoundError(
            "incomplete Binance Vision cache: "
            f"missing={len(report.missing_urls)} empty={len(report.empty_urls)} "
            f"invalid={len(report.invalid_urls)}"
        )
    return report


def sync_binance_vision_urls(
    urls: Sequence[str],
    *,
    transport: _VisionArchiveTransport,
) -> BinanceVisionCacheReport:
    """Synchronize only missing/invalid members of an official authored URL set."""

    resolved_urls = _official_urls(urls)
    if transport.cache_root is None:
        raise ValueError("transport cache_root is required for Vision synchronization")
    root = Path(transport.cache_root)
    before = inspect_binance_vision_urls(resolved_urls, cache_root=root)
    required = (
        set(before.missing_urls) | set(before.empty_urls) | set(before.invalid_urls)
    )
    targets = tuple(url for url in resolved_urls if url in required)
    for url in targets:
        path = vision_cache_path(root, url)
        if path.exists():
            path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
        payload = transport._request_bytes(url)
        if not payload:
            raise RuntimeError(f"downloaded empty Binance Vision archive: {url}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                f"transport did not publish Binance Vision cache file: {path}"
            )
    after = inspect_binance_vision_urls(resolved_urls, cache_root=root)
    if not after.complete:
        raise FileNotFoundError(
            "Binance Vision synchronization incomplete: "
            f"missing={len(after.missing_urls)} empty={len(after.empty_urls)} "
            f"invalid={len(after.invalid_urls)}"
        )
    return BinanceVisionCacheReport(
        cache_root=root,
        planned_count=after.planned_count,
        cached_count=before.cached_count,
        downloaded_count=len(targets),
        missing_urls=after.missing_urls,
        empty_urls=after.empty_urls,
        invalid_urls=after.invalid_urls,
    )


def sync_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    transport: _VisionArchiveTransport,
) -> BinanceVisionCacheReport:
    """Download only absent or empty archives into the transport cache root."""

    return sync_binance_vision_urls(plan.urls, transport=transport)


__all__ = [
    "MISSING_ARCHIVE_SCHEMA",
    "BinanceVisionCachePlan",
    "BinanceVisionCacheReport",
    "inspect_binance_vision_cache",
    "inspect_binance_vision_urls",
    "missing_archive_evidence_path",
    "plan_binance_vision_cache",
    "read_missing_archive_evidence",
    "require_complete_binance_vision_cache",
    "sync_binance_vision_cache",
    "sync_binance_vision_urls",
    "vision_cache_path",
    "write_missing_archive_evidence",
]
