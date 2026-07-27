"""Deterministic planning and synchronization for shared Binance Vision archives."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    plan_vision_kline_urls,
    vision_funding_url,
)

_VISION_PREFIX = "https://data.binance.vision/data/"


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

    @property
    def complete(self) -> bool:
        return not self.missing_urls and not self.empty_urls

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_root": str(self.cache_root),
            "cached_count": self.cached_count,
            "complete": self.complete,
            "downloaded_count": self.downloaded_count,
            "empty_count": len(self.empty_urls),
            "missing_count": len(self.missing_urls),
            "planned_count": self.planned_count,
        }


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _ordered_nonempty(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in values))
    if not result or any(not value for value in result):
        raise ValueError(f"{field} must contain non-empty values")
    return result


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


def inspect_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    cache_root: str | Path,
) -> BinanceVisionCacheReport:
    root = Path(cache_root)
    cached: list[str] = []
    missing: list[str] = []
    empty: list[str] = []
    for url in plan.urls:
        path = vision_cache_path(root, url)
        if not path.is_file():
            missing.append(url)
        elif path.stat().st_size <= 0:
            empty.append(url)
        else:
            cached.append(url)
    return BinanceVisionCacheReport(
        cache_root=root,
        planned_count=len(plan.urls),
        cached_count=len(cached),
        downloaded_count=0,
        missing_urls=tuple(missing),
        empty_urls=tuple(empty),
    )


def require_complete_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    cache_root: str | Path,
) -> BinanceVisionCacheReport:
    report = inspect_binance_vision_cache(plan, cache_root=cache_root)
    if not report.complete:
        raise FileNotFoundError(
            "incomplete Binance Vision cache: "
            f"missing={len(report.missing_urls)} empty={len(report.empty_urls)}"
        )
    return report


def sync_binance_vision_cache(
    plan: BinanceVisionCachePlan,
    *,
    transport: _VisionArchiveTransport | BinancePublicTransport,
) -> BinanceVisionCacheReport:
    """Download only absent or empty archives into the transport cache root."""

    if transport.cache_root is None:
        raise ValueError("transport cache_root is required for Vision synchronization")
    root = Path(transport.cache_root)
    before = inspect_binance_vision_cache(plan, cache_root=root)
    required = set(before.missing_urls) | set(before.empty_urls)
    targets = tuple(url for url in plan.urls if url in required)

    for url in targets:
        path = vision_cache_path(root, url)
        if path.exists() and path.stat().st_size <= 0:
            path.unlink()
        payload = transport._request_bytes(url)
        if not payload:
            raise RuntimeError(f"downloaded empty Binance Vision archive: {url}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                f"transport did not publish Binance Vision cache file: {path}"
            )

    after = inspect_binance_vision_cache(plan, cache_root=root)
    if not after.complete:
        raise FileNotFoundError(
            "Binance Vision synchronization incomplete: "
            f"missing={len(after.missing_urls)} empty={len(after.empty_urls)}"
        )
    return BinanceVisionCacheReport(
        cache_root=root,
        planned_count=after.planned_count,
        cached_count=before.cached_count,
        downloaded_count=len(targets),
        missing_urls=after.missing_urls,
        empty_urls=after.empty_urls,
    )
