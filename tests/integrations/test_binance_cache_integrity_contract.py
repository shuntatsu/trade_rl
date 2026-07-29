from __future__ import annotations

import json
import urllib.request

import pytest

from trade_rl.integrations.binance import BinancePublicTransport, BinanceTransportError


class _Response:
    headers = {"ETag": '"fixture"', "Last-Modified": "Wed, 29 Jul 2026 00:00:00 GMT"}

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_vision_cache_sidecar_detects_tampering(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"official-archive"
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Response(payload))
    transport = BinancePublicTransport(cache_root=tmp_path, max_attempts=1)
    url = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/file.zip"
    assert transport._request_bytes(url) == payload
    binary = next(tmp_path.rglob("*.bin"))
    evidence = json.loads(binary.with_suffix(".json").read_text())
    assert evidence["schema_version"] == "binance_vision_raw_cache_v1"
    assert evidence["size_bytes"] == len(payload)
    binary.write_bytes(b"tampered")
    with pytest.raises(BinanceTransportError, match="digest|size"):
        transport._request_bytes(url)
