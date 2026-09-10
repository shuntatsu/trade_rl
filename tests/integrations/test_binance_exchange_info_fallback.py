from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pytest

from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _http_451(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url,
        451,
        "Unavailable For Legal Reasons",
        {},
        None,
    )


def test_usdm_exchange_info_falls_back_to_official_www_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_payload = b'{"serverTime":123,"symbols":[]}'
    requested_urls: list[str] = []

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _Response:
        assert timeout == 30.0
        requested_urls.append(request.full_url)
        if request.full_url == "https://fapi.binance.com/fapi/v1/exchangeInfo":
            raise _http_451(request.full_url)
        if request.full_url == "https://www.binance.com/fapi/v1/exchangeInfo":
            return _Response(raw_payload)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = BinancePublicTransport(max_attempts=1)

    snapshot = transport.load_exchange_information_snapshot(
        market=BinanceMarket.USDS_M,
        clock=lambda: datetime(2026, 9, 10, tzinfo=UTC),
    )

    assert requested_urls == [
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "https://www.binance.com/fapi/v1/exchangeInfo",
    ]
    assert snapshot.source_uri == "https://www.binance.com/fapi/v1/exchangeInfo"
    assert snapshot.raw_payload == raw_payload
    assert snapshot.raw_payload_sha256 == hashlib.sha256(raw_payload).hexdigest()


def test_usdm_exchange_info_fails_closed_when_both_official_hosts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _Response:
        assert timeout == 30.0
        requested_urls.append(request.full_url)
        raise _http_451(request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = BinancePublicTransport(max_attempts=1)

    with pytest.raises(BinanceTransportError, match="Binance request failed"):
        transport.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)

    assert requested_urls == [
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "https://www.binance.com/fapi/v1/exchangeInfo",
    ]
