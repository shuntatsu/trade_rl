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

_PRIMARY_USDM = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_FALLBACK_USDM = "https://www.binance.com/fapi/v1/exchangeInfo"


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


def test_usdm_exchange_info_falls_back_after_primary_transport_failure(
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
        if request.full_url == _PRIMARY_USDM:
            raise _http_451(request.full_url)
        if request.full_url == _FALLBACK_USDM:
            return _Response(raw_payload)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = BinancePublicTransport(max_attempts=1)

    snapshot = transport.load_exchange_information_snapshot(
        market=BinanceMarket.USDS_M,
        clock=lambda: datetime(2026, 9, 10, tzinfo=UTC),
    )

    assert requested_urls == [_PRIMARY_USDM, _FALLBACK_USDM]
    assert snapshot.source_uri == _FALLBACK_USDM
    assert snapshot.raw_payload == raw_payload
    assert snapshot.raw_payload_sha256 == hashlib.sha256(raw_payload).hexdigest()


def test_usdm_exchange_info_fails_closed_when_both_routes_fail(
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

    assert requested_urls == [_PRIMARY_USDM, _FALLBACK_USDM]


@pytest.mark.parametrize(
    "raw_payload, message",
    (
        (b"not-json", "invalid JSON"),
        (b"[]", "must be an object"),
    ),
)
def test_usdm_exchange_info_does_not_fallback_after_successful_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
    raw_payload: bytes,
    message: str,
) -> None:
    requested_urls: list[str] = []

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _Response:
        assert timeout == 30.0
        requested_urls.append(request.full_url)
        if request.full_url != _PRIMARY_USDM:
            raise AssertionError("payload validation failure must not trigger fallback")
        return _Response(raw_payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = BinancePublicTransport(max_attempts=1)

    with pytest.raises(BinanceTransportError, match=message):
        transport.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)

    assert requested_urls == [_PRIMARY_USDM]


@pytest.mark.parametrize(
    "market, primary_url",
    (
        (BinanceMarket.SPOT, "https://api.binance.com/api/v3/exchangeInfo"),
        (BinanceMarket.COIN_M, "https://dapi.binance.com/dapi/v1/exchangeInfo"),
    ),
)
def test_non_usdm_exchange_info_does_not_use_usdm_fallback(
    monkeypatch: pytest.MonkeyPatch,
    market: BinanceMarket,
    primary_url: str,
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
        transport.load_exchange_information_snapshot(market=market)

    assert requested_urls == [primary_url]


def test_network_disabled_usdm_exchange_info_does_not_enter_fallback_route() -> None:
    transport = BinancePublicTransport(allow_network=False)

    with pytest.raises(BinanceTransportError) as error:
        transport.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)

    message = str(error.value)
    assert _PRIMARY_USDM in message
    assert _FALLBACK_USDM not in message
