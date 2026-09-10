from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
    FrozenBinanceExchangeInfoTransport,
    binance_interval_milliseconds,
)
from trade_rl.integrations.binance.cache import vision_cache_path


def _publish_cache_entry(path: Path, *, url: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "acquired_at": "2026-09-01T00:00:00+00:00",
                "downloader": "test",
                "etag": None,
                "last_modified": None,
                "schema_version": "binance_vision_raw_cache_v1",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "url": url,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_cache_only_transport_reads_verified_vision_hit_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-01.zip"
    )
    payload = b"frozen-source-bytes"
    cache_path = vision_cache_path(tmp_path, url)
    _publish_cache_entry(cache_path, url=url, payload=payload)
    transport = BinancePublicTransport(cache_root=tmp_path, allow_network=False)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("cache-only transport attempted HTTP"),
    )

    assert transport._request_bytes(url) == payload


def test_cache_only_transport_rejects_uncached_vision_source(tmp_path: Path) -> None:
    transport = BinancePublicTransport(cache_root=tmp_path, allow_network=False)
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-02.zip"
    )

    with pytest.raises(BinanceTransportError, match="network access is disabled"):
        transport._request_bytes(url)


def test_cache_only_transport_rejects_rest_exchange_info_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(cache_root=tmp_path, allow_network=False)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("cache-only transport attempted HTTP"),
    )

    with pytest.raises(BinanceTransportError, match="network access is disabled"):
        transport.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)


def test_allow_network_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="allow_network must be a boolean"):
        BinancePublicTransport(allow_network=1)  # type: ignore[arg-type]


def test_public_interval_duration_query_uses_binance_contract() -> None:
    assert binance_interval_milliseconds("15m") == 15 * 60 * 1_000
    assert binance_interval_milliseconds("1h") == 60 * 60 * 1_000
    assert binance_interval_milliseconds("1d") == 24 * 60 * 60 * 1_000
    with pytest.raises(ValueError, match="unsupported Binance interval"):
        binance_interval_milliseconds("7m")


class _MetadataDelegate:
    def __init__(self, snapshot: BinanceExchangeInfoSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: object = None,
    ) -> BinanceExchangeInfoSnapshot:
        assert BinanceMarket(market) is BinanceMarket.USDS_M
        self.calls += 1
        return self.snapshot


def test_frozen_metadata_transport_matches_dataset_builder_contract(
    tmp_path: Path,
) -> None:
    raw = (
        b'{"symbols":[{"symbol":"BTCUSDT","status":"TRADING",'
        b'"onboardDate":0,"filters":[]}]}'
    )
    snapshot = BinanceExchangeInfoSnapshot(
        payload={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "onboardDate": 0,
                    "filters": [],
                }
            ]
        },
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )
    delegate = _MetadataDelegate(snapshot)
    initial = FrozenBinanceExchangeInfoTransport(tmp_path, delegate=delegate)
    initial.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)
    assert delegate.calls == 1

    frozen = FrozenBinanceExchangeInfoTransport(tmp_path)
    payload, source = frozen.load_exchange_information(market=BinanceMarket.USDS_M)

    assert source == "frozen:exchange-info"
    symbols = payload["symbols"]
    assert isinstance(symbols, list)
    assert symbols[0]["symbol"] == "BTCUSDT"
    symbols[0]["symbol"] = "MUTATED"
    again, _ = frozen.load_exchange_information(market=BinanceMarket.USDS_M)
    assert again["symbols"][0]["symbol"] == "BTCUSDT"
