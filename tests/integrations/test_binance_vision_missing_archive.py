"""The download step records an HTTP 404 for a Vision archive as cache evidence."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from trade_rl.integrations.binance import (
    BinancePublicTransport,
    BinanceTransportError,
    vision_cache_path,
)
from trade_rl.integrations.binance.cache import (
    missing_archive_evidence_path,
    read_missing_archive_evidence,
    write_missing_archive_evidence,
)

URL = (
    "https://data.binance.vision/data/futures/um/monthly/klines/"
    "DOTUSDT/1d/DOTUSDT-1d-2021-01.zip"
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _http(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, status, "x", {}, None)  # type: ignore[arg-type]


def _transport(root: Path) -> BinancePublicTransport:
    return BinancePublicTransport(
        cache_root=root, max_attempts=2, retry_backoff_seconds=0.0
    )


def test_an_http_404_is_recorded_with_url_status_and_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def urlopen(request: object, *, timeout: float) -> _Response:
        raise _http(404)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(BinanceTransportError, match="404"):
        _transport(tmp_path)._request_bytes(URL)

    path = missing_archive_evidence_path(vision_cache_path(tmp_path, URL))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["url"] == URL and record["http_status"] == 404
    assert record["schema_version"] == "binance_vision_missing_archive_v1"
    assert record["observed_at"].endswith("+00:00") and record["downloader"]
    cache_path = vision_cache_path(tmp_path, URL)
    assert read_missing_archive_evidence(cache_path, URL) == record
    assert not cache_path.exists()  # a 404 never creates an archive


@pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
def test_other_http_failures_are_never_recorded_as_a_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    def urlopen(request: object, *, timeout: float) -> _Response:
        raise _http(status)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(BinanceTransportError):
        _transport(tmp_path)._request_bytes(URL)

    cache_path = vision_cache_path(tmp_path, URL)
    assert read_missing_archive_evidence(cache_path, URL) is None
    assert not tuple(tmp_path.rglob("*.missing.json"))


def test_a_timeout_and_a_cache_miss_without_network_record_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def urlopen(request: object, *, timeout: float) -> _Response:
        raise TimeoutError("slow")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(BinanceTransportError):
        _transport(tmp_path)._request_bytes(URL)
    offline = BinancePublicTransport(cache_root=tmp_path, allow_network=False)
    with pytest.raises(BinanceTransportError, match="network access is disabled"):
        offline._request_bytes(URL)
    assert not tuple(tmp_path.rglob("*.missing.json"))


def test_a_later_successful_download_supersedes_the_404_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = vision_cache_path(tmp_path, URL)
    write_missing_archive_evidence(cache_path, url=URL, downloader="test")
    assert read_missing_archive_evidence(cache_path, URL) is not None

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, *, timeout: _Response(b"zip-bytes")
    )
    assert _transport(tmp_path)._request_bytes(URL) == b"zip-bytes"

    assert read_missing_archive_evidence(cache_path, URL) is None
    assert cache_path.is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        {"http_status": 503},
        {"http_status": True},
        {"url": URL + "x"},
        {"schema_version": "other"},
        {"observed_at": "2026-11-13T00:00:00"},  # no timezone
        {"observed_at": "yesterday"},
        {"extra": 1},
    ],
)
def test_malformed_or_mismatching_evidence_is_never_read_as_a_404(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    cache_path = vision_cache_path(tmp_path, URL)
    write_missing_archive_evidence(cache_path, url=URL, downloader="test")
    path = missing_archive_evidence_path(cache_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, **mutation}), encoding="utf-8")
    with pytest.raises(BinanceTransportError):
        read_missing_archive_evidence(cache_path, URL)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(BinanceTransportError):
        read_missing_archive_evidence(cache_path, URL)
