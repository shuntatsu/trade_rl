from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.binance_cache import (
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
    sync_binance_vision_cache,
    vision_cache_path,
)


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_plan_is_deterministic_and_includes_completed_funding_months() -> None:
    plan = plan_binance_vision_cache(
        market="usds-m",
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("15m", "1h"),
        start_time=_utc(2026, 1, 1),
        end_time=_utc(2026, 3, 1),
    )

    assert plan.urls == tuple(dict.fromkeys(plan.urls))
    assert any("BTCUSDT-15m-2026-01.zip" in url for url in plan.urls)
    assert any("ETHUSDT-1h-2026-02.zip" in url for url in plan.urls)
    assert any("BTCUSDT-fundingRate-2026-01.zip" in url for url in plan.urls)
    assert any("ETHUSDT-fundingRate-2026-02.zip" in url for url in plan.urls)


def test_plan_excludes_incomplete_trailing_funding_month() -> None:
    plan = plan_binance_vision_cache(
        market="usds-m",
        symbols=("BTCUSDT",),
        intervals=("1h",),
        start_time=_utc(2026, 1, 1),
        end_time=_utc(2026, 2, 15),
    )

    funding_urls = tuple(url for url in plan.urls if "fundingRate" in url)
    assert len(funding_urls) == 1
    assert "2026-01.zip" in funding_urls[0]


class _FakeTransport:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.calls: list[str] = []

    def _request_bytes(self, url: str) -> bytes:
        self.calls.append(url)
        payload = f"payload:{url}".encode()
        path = vision_cache_path(self.cache_root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish_cache_entry(path, url=url, payload=payload)
        return payload


def _publish_cache_entry(path: Path, *, url: str, payload: bytes) -> None:
    path.write_bytes(payload)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "acquired_at": "2026-08-09T00:00:00+00:00",
                "downloader": "test",
                "etag": None,
                "last_modified": None,
                "schema_version": "binance_vision_raw_cache_v1",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "url": url,
            }
        ),
        encoding="utf-8",
    )


def test_sync_downloads_only_missing_archives(tmp_path: Path) -> None:
    plan = plan_binance_vision_cache(
        market="spot",
        symbols=("BTCUSDT",),
        intervals=("1h",),
        start_time=_utc(2026, 1, 1),
        end_time=_utc(2026, 3, 1),
    )
    first_url, second_url = plan.urls
    first_path = vision_cache_path(tmp_path, first_url)
    first_path.parent.mkdir(parents=True)
    _publish_cache_entry(first_path, url=first_url, payload=b"already-cached")
    transport = _FakeTransport(tmp_path)

    report = sync_binance_vision_cache(plan, transport=transport)

    assert transport.calls == [second_url]
    assert report.planned_count == 2
    assert report.cached_count == 1
    assert report.downloaded_count == 1
    assert report.missing_urls == ()
    assert report.empty_urls == ()


def test_sync_redownloads_archive_without_content_evidence(tmp_path: Path) -> None:
    plan = plan_binance_vision_cache(
        market="spot",
        symbols=("BTCUSDT",),
        intervals=("1h",),
        start_time=_utc(2026, 1, 1),
        end_time=_utc(2026, 2, 1),
    )
    url = plan.urls[0]
    cache_path = vision_cache_path(tmp_path, url)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"legacy-without-evidence")
    transport = _FakeTransport(tmp_path)

    report = sync_binance_vision_cache(plan, transport=transport)

    assert transport.calls == [url]
    assert report.cached_count == 0
    assert report.downloaded_count == 1
    assert report.complete is True


def test_complete_check_rejects_empty_cache_file(tmp_path: Path) -> None:
    plan = plan_binance_vision_cache(
        market="spot",
        symbols=("BTCUSDT",),
        intervals=("1h",),
        start_time=_utc(2026, 1, 1),
        end_time=_utc(2026, 2, 1),
    )
    cache_path = vision_cache_path(tmp_path, plan.urls[0])
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="empty=1"):
        require_complete_binance_vision_cache(plan, cache_root=tmp_path)


def test_cache_path_is_stable_and_rejects_non_vision_url(tmp_path: Path) -> None:
    url = (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-01.zip"
    )

    path = vision_cache_path(tmp_path, url)

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert path == tmp_path / digest[:2] / f"{digest}.bin"
    with pytest.raises(ValueError, match="Binance Vision"):
        vision_cache_path(tmp_path, "https://example.com/archive.zip")
