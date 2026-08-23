"""Coverage-only capability decision for Causal Alpha V4 derivative context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.binance_cache import inspect_binance_vision_urls
from trade_rl.integrations.binance_v4_context import vision_futures_metrics_url


@dataclass(frozen=True, slots=True)
class BinanceV4ProfileCapability:
    symbols: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    required_archive_count: int
    cached_archive_count: int
    missing_archive_count: int
    invalid_archive_count: int
    derivative_metrics_complete: bool
    profile_name: str
    source_digest: str

    def __post_init__(self) -> None:
        symbols = tuple(self.symbols)
        if not symbols or any(not symbol for symbol in symbols):
            raise ValueError("V4 capability symbols must be non-empty")
        if len(set(symbols)) != len(symbols):
            raise ValueError("V4 capability symbols must be unique")
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise ValueError("V4 capability range must be timezone-aware")
        if self.end_time <= self.start_time:
            raise ValueError("V4 capability range is invalid")
        for name in (
            "required_archive_count",
            "cached_archive_count",
            "missing_archive_count",
            "invalid_archive_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 capability {name} must be non-negative")
        if (
            self.cached_archive_count
            + self.missing_archive_count
            + self.invalid_archive_count
            > self.required_archive_count
        ):
            raise ValueError("V4 capability archive counts are inconsistent")
        expected_complete = (
            self.required_archive_count > 0
            and self.cached_archive_count == self.required_archive_count
            and self.missing_archive_count == 0
            and self.invalid_archive_count == 0
        )
        if self.derivative_metrics_complete != expected_complete:
            raise ValueError("V4 capability complete flag is inconsistent")
        expected_profile = (
            "cross_market_derivatives_v1"
            if expected_complete
            else "cross_market_core_v1"
        )
        if self.profile_name != expected_profile:
            raise ValueError("V4 capability profile_name is inconsistent")
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64:
            raise ValueError("V4 capability source_digest is invalid")
        object.__setattr__(self, "symbols", symbols)


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _daily_metrics_urls(
    *, symbols: tuple[str, ...], start_time: datetime, end_time: datetime
) -> tuple[str, ...]:
    day = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    urls: list[str] = []
    while day < end_time:
        for symbol in symbols:
            urls.append(vision_futures_metrics_url(symbol, day))
        day += timedelta(days=1)
    return tuple(urls)


def inspect_binance_v4_derivative_capability(
    *,
    symbols: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    cache_root: str | Path,
) -> BinanceV4ProfileCapability:
    """Resolve core vs derivatives profile from immutable archive coverage only."""

    resolved_symbols = tuple(
        dict.fromkeys(str(symbol).strip().upper() for symbol in symbols)
    )
    if not resolved_symbols or any(not symbol for symbol in resolved_symbols):
        raise ValueError("symbols must contain non-empty values")
    start = _aware_utc(start_time, field="start_time")
    end = _aware_utc(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    urls = _daily_metrics_urls(symbols=resolved_symbols, start_time=start, end_time=end)
    report = inspect_binance_vision_urls(urls, cache_root=cache_root)
    invalid_count = len(report.empty_urls) + len(report.invalid_urls)
    complete = report.complete and report.cached_count == len(urls)
    profile_name = "cross_market_derivatives_v1" if complete else "cross_market_core_v1"
    source_digest = content_digest(
        {
            "cached_urls": tuple(
                url
                for url in urls
                if url not in set(report.missing_urls)
                and url not in set(report.empty_urls)
                and url not in set(report.invalid_urls)
            ),
            "end_time": end.isoformat(),
            "invalid_urls": tuple((*report.empty_urls, *report.invalid_urls)),
            "missing_urls": report.missing_urls,
            "profile_name": profile_name,
            "schema_version": "binance_v4_derivative_capability_v1",
            "start_time": start.isoformat(),
            "symbols": resolved_symbols,
            "urls": urls,
        }
    )
    return BinanceV4ProfileCapability(
        symbols=resolved_symbols,
        start_time=start,
        end_time=end,
        required_archive_count=len(urls),
        cached_archive_count=report.cached_count,
        missing_archive_count=len(report.missing_urls),
        invalid_archive_count=invalid_count,
        derivative_metrics_complete=complete,
        profile_name=profile_name,
        source_digest=source_digest,
    )


__all__ = [
    "BinanceV4ProfileCapability",
    "inspect_binance_v4_derivative_capability",
]
