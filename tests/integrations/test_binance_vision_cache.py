from __future__ import annotations

from pathlib import Path

from trade_rl.integrations.binance import BinancePublicTransport


def test_vision_archive_disk_cache_avoids_second_download(monkeypatch, tmp_path: Path) -> None:
    payload = b"immutable-vision-archive"
    calls: list[str] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return payload

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        calls.append(str(getattr(request, "full_url")))
        assert timeout > 0.0
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2025-01.zip"
    )

    first = BinancePublicTransport(cache_root=tmp_path)._request_bytes(url)
    second = BinancePublicTransport(cache_root=tmp_path)._request_bytes(url)

    assert first == payload
    assert second == payload
    assert calls == [url]
    cache_files = tuple(path for path in tmp_path.rglob("*") if path.is_file())
    assert len(cache_files) == 1
    assert cache_files[0].read_bytes() == payload


def test_rest_requests_are_not_loaded_from_vision_cache(monkeypatch, tmp_path: Path) -> None:
    responses = iter((b'{"version": 1}', b'{"version": 2}'))
    calls = 0

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return next(responses)

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        nonlocal calls
        calls += 1
        assert timeout > 0.0
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = BinancePublicTransport(cache_root=tmp_path)
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    assert transport._request_bytes(url) == b'{"version": 1}'
    assert transport._request_bytes(url) == b'{"version": 2}'
    assert calls == 2
    assert not tuple(path for path in tmp_path.rglob("*") if path.is_file())
