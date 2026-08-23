from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.binance_cache import (
    inspect_binance_vision_urls,
    sync_binance_vision_urls,
    vision_cache_path,
)
from trade_rl.integrations.binance_v4_context_capability import (
    inspect_binance_v4_derivative_capability,
)


def _evidence(url: str, payload: bytes) -> dict[str, object]:
    return {
        "acquired_at": "2026-08-23T00:00:00+00:00",
        "downloader": "test",
        "etag": None,
        "last_modified": None,
        "schema_version": "binance_vision_raw_cache_v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "url": url,
    }


def _publish(cache_root: Path, url: str, payload: bytes = b"payload") -> None:
    path = vision_cache_path(cache_root, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_suffix(".json").write_text(
        json.dumps(_evidence(url, payload), sort_keys=True),
        encoding="utf-8",
    )


def test_generic_vision_url_inspection_reports_missing_and_cached(
    tmp_path: Path,
) -> None:
    first = (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    )
    second = (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-01-02.zip"
    )
    _publish(tmp_path, first)

    report = inspect_binance_vision_urls((first, second), cache_root=tmp_path)
    assert report.planned_count == 2
    assert report.cached_count == 1
    assert report.missing_urls == (second,)
    assert not report.complete


def test_generic_vision_url_inspection_rejects_non_official_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official Binance Vision"):
        inspect_binance_vision_urls(
            ("https://example.com/fake.zip",),
            cache_root=tmp_path,
        )


class _PublishingTransport:
    def __init__(self, root: Path) -> None:
        self.cache_root = root
        self.calls: list[str] = []

    def _request_bytes(self, url: str) -> bytes:
        self.calls.append(url)
        payload = f"payload:{url}".encode()
        _publish(self.cache_root, url, payload)
        return payload


def test_generic_vision_url_sync_downloads_only_missing(tmp_path: Path) -> None:
    first = (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    )
    second = (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-01-02.zip"
    )
    _publish(tmp_path, first)
    transport = _PublishingTransport(tmp_path)

    report = sync_binance_vision_urls((first, second), transport=transport)
    assert report.complete
    assert report.downloaded_count == 1
    assert transport.calls == [second]


def test_derivative_capability_requires_complete_symbol_day_coverage(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 3, tzinfo=UTC)
    symbols = ("BTCUSDT", "ETHUSDT")
    urls = tuple(
        (
            "https://data.binance.vision/data/futures/um/daily/metrics/"
            f"{symbol}/{symbol}-metrics-2026-01-{day:02d}.zip"
        )
        for symbol in symbols
        for day in (1, 2)
    )
    for url in urls[:-1]:
        _publish(tmp_path, url)

    incomplete = inspect_binance_v4_derivative_capability(
        symbols=symbols,
        start_time=start,
        end_time=end,
        cache_root=tmp_path,
    )
    assert incomplete.required_archive_count == 4
    assert incomplete.cached_archive_count == 3
    assert incomplete.missing_archive_count == 1
    assert not incomplete.derivative_metrics_complete
    assert incomplete.profile_name == "cross_market_core_v1"

    _publish(tmp_path, urls[-1])
    complete = inspect_binance_v4_derivative_capability(
        symbols=symbols,
        start_time=start,
        end_time=end,
        cache_root=tmp_path,
    )
    assert complete.derivative_metrics_complete
    assert complete.profile_name == "cross_market_derivatives_v1"
    assert complete.missing_archive_count == 0
    assert len(complete.source_digest) == 64
