from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceInstrumentMetadata,
    BinanceMarket,
    BinanceMarketDataSource,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
    BinanceUnsupportedContractError,
    build_binance_market_dataset,
    vision_funding_url,
    vision_kline_url,
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _kline(open_time: datetime, *, quote_volume: float, close: float) -> list[Any]:
    open_ms = _ms(open_time)
    return [
        open_ms,
        str(close - 1.0),
        str(close + 1.0),
        str(close - 2.0),
        str(close),
        "12.0",
        open_ms + 3_599_999,
        str(quote_volume),
        10,
        "5.0",
        str(quote_volume / 2.0),
        "0",
    ]


class FakeTransport:
    def __init__(self, *, duplicate: bool = False) -> None:
        self.start = datetime(2026, 6, 1, tzinfo=UTC)
        rows = [
            _kline(self.start, quote_volume=6_250_000.0, close=100.0),
            _kline(
                self.start + timedelta(hours=1),
                quote_volume=6_500_000.0,
                close=101.0,
            ),
            _kline(
                self.start + timedelta(hours=2),
                quote_volume=6_750_000.0,
                close=102.0,
            ),
        ]
        if duplicate:
            rows[1] = list(rows[0])
        self.rows = rows

    def load_klines(self, **_: object) -> tuple[list[list[Any]], str]:
        return self.rows, "fixture:klines"

    def load_funding_rates(self, **_: object) -> tuple[list[tuple[int, float]], str]:
        return [
            (_ms(self.start + timedelta(hours=1)), 0.0001),
            (_ms(self.start + timedelta(hours=3)), -0.0002),
        ], "fixture:funding"

    def load_exchange_information(self, **_: object) -> tuple[dict[str, object], str]:
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "onboardDate": _ms(self.start - timedelta(days=30)),
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }, "fixture:exchange-info"


class _ExactBytesResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _ExactBytesResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_exchange_information_snapshot_preserves_exact_rest_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_payload = b'{"serverTime": 123, "symbols": []}\n'
    requested_urls: list[str] = []

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _ExactBytesResponse:
        assert timeout == 30.0
        requested_urls.append(request.full_url)
        return _ExactBytesResponse(raw_payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    captured_at = datetime(2026, 7, 17, 15, 30, tzinfo=timezone(timedelta(hours=9)))
    transport = BinancePublicTransport(max_attempts=1)

    snapshot = transport.load_exchange_information_snapshot(
        market=BinanceMarket.USDS_M,
        clock=lambda: captured_at,
    )

    assert requested_urls == ["https://fapi.binance.com/fapi/v1/exchangeInfo"]
    assert snapshot == BinanceExchangeInfoSnapshot(
        payload={"serverTime": 123, "symbols": []},
        raw_payload=raw_payload,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 7, 17, 6, 30, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
    )
    assert snapshot.retrieved_at.tzinfo is UTC
    with pytest.raises(FrozenInstanceError):
        snapshot.source_uri = "changed"  # type: ignore[misc]


def test_exchange_information_snapshot_rejects_vision_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1)
    monkeypatch.setattr(
        transport,
        "_request_bytes",
        lambda _: pytest.fail("Vision mode must not issue a REST request"),
    )

    with pytest.raises(BinanceTransportError, match="Vision"):
        transport.load_exchange_information_snapshot(
            market=BinanceMarket.SPOT,
            mode=BinanceTransportMode.VISION,
        )


@pytest.mark.parametrize("raw_payload", [b"[]", b"not-json"])
def test_exchange_information_snapshot_rejects_invalid_object_json(
    monkeypatch: pytest.MonkeyPatch,
    raw_payload: bytes,
) -> None:
    transport = BinancePublicTransport(max_attempts=1)
    monkeypatch.setattr(transport, "_request_bytes", lambda _: raw_payload)

    with pytest.raises(BinanceTransportError, match="invalid JSON|must be an object"):
        transport.load_exchange_information_snapshot(market=BinanceMarket.SPOT)


def test_load_exchange_information_delegates_to_snapshot_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport()
    snapshot = BinanceExchangeInfoSnapshot(
        payload={"symbols": []},
        raw_payload=b'{"symbols": []}',
        source_uri="https://api.binance.com/api/v3/exchangeInfo",
        retrieved_at=datetime(2026, 7, 17, tzinfo=UTC),
        raw_payload_sha256="digest",
    )
    calls: list[dict[str, object]] = []

    def load_snapshot(**kwargs: object) -> BinanceExchangeInfoSnapshot:
        calls.append(kwargs)
        return snapshot

    monkeypatch.setattr(
        transport,
        "load_exchange_information_snapshot",
        load_snapshot,
    )

    result = transport.load_exchange_information(
        market=BinanceMarket.SPOT,
        mode=BinanceTransportMode.REST,
    )

    assert result == ({"symbols": []}, "rest")
    assert calls == [{"market": BinanceMarket.SPOT, "mode": BinanceTransportMode.REST}]


def test_exchange_information_snapshot_deeply_freezes_nested_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_payload = (
        b'{"symbols":[{"symbol":"BTCUSDT","filters":'
        b'[{"filterType":"PRICE_FILTER","tickSize":"0.10"}]}]}'
    )
    transport = BinancePublicTransport(max_attempts=1)
    monkeypatch.setattr(transport, "_request_bytes", lambda _: raw_payload)

    snapshot = transport.load_exchange_information_snapshot(
        market=BinanceMarket.USDS_M,
    )

    symbols = snapshot.payload["symbols"]
    assert isinstance(symbols, tuple)
    symbol = symbols[0]
    assert isinstance(symbol, Mapping)
    filters = symbol["filters"]
    assert isinstance(filters, tuple)
    price_filter = filters[0]
    assert isinstance(price_filter, Mapping)
    with pytest.raises(TypeError):
        price_filter["tickSize"] = "999"  # type: ignore[index]


def test_load_exchange_information_returns_independent_mutable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_payload = b'{"symbols":[{"symbol":"BTCUSDT"}]}'
    transport = BinancePublicTransport(max_attempts=1)
    monkeypatch.setattr(transport, "_request_bytes", lambda _: raw_payload)
    snapshot = transport.load_exchange_information_snapshot(
        market=BinanceMarket.USDS_M,
    )
    monkeypatch.setattr(
        transport,
        "load_exchange_information_snapshot",
        lambda **_: snapshot,
    )

    payload, source = transport.load_exchange_information(
        market=BinanceMarket.USDS_M,
    )

    assert source == "rest"
    symbols = payload["symbols"]
    assert isinstance(symbols, list)
    symbol = symbols[0]
    assert isinstance(symbol, dict)
    symbol["symbol"] = "MUTATED"
    snapshot_symbols = snapshot.payload["symbols"]
    assert isinstance(snapshot_symbols, tuple)
    snapshot_symbol = snapshot_symbols[0]
    assert isinstance(snapshot_symbol, Mapping)
    assert snapshot_symbol["symbol"] == "BTCUSDT"


def test_usds_m_source_uses_close_boundaries_quote_volume_and_sparse_funding() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    source = BinanceMarketDataSource(
        market=BinanceMarket.USDS_M,
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=3),
        transport_mode=BinanceTransportMode.VISION,
        transport=FakeTransport(),
    )

    series = source.load("BTCUSDT")

    expected = np.array(
        [
            "2026-06-01T01:00:00",
            "2026-06-01T02:00:00",
            "2026-06-01T03:00:00",
        ],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(series.timestamps, expected)
    np.testing.assert_array_equal(series.available_at, expected)
    np.testing.assert_allclose(
        series.volume,
        np.array([6_250_000.0, 6_500_000.0, 6_750_000.0]),
    )
    np.testing.assert_allclose(series.funding_rate, np.array([0.0001, 0.0, -0.0002]))
    np.testing.assert_array_equal(
        series.funding_available,
        np.array([True, False, True]),
    )
    assert source.sources_used == ("fixture:funding", "fixture:klines")


def test_source_rejects_duplicate_kline_open_times() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=3),
        transport=FakeTransport(duplicate=True),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        source.load("BTCUSDT")


def test_build_uses_exchange_metadata_and_quote_notional_volume() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)

    result = build_binance_market_dataset(
        market="usds-m",
        symbols=("BTCUSDT",),
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=3),
        transport=FakeTransport(),
    )

    dataset = result.dataset
    assert dataset.volume_units == (VolumeUnit.QUOTE_NOTIONAL,)
    np.testing.assert_allclose(dataset.resolved_array("tick_size"), 0.1)
    np.testing.assert_allclose(dataset.resolved_array("lot_size"), 0.001)
    np.testing.assert_allclose(dataset.resolved_array("minimum_notional"), 5.0)
    assert result.metadata == (
        BinanceInstrumentMetadata(
            symbol="BTCUSDT",
            listed_at=start - timedelta(days=30),
            tick_size=0.1,
            lot_size=0.001,
            minimum_notional=5.0,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            contract_multiplier=1.0,
        ),
    )


def test_build_binds_metadata_evidence_to_dataset_identity() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    build_kwargs = {
        "market": "usds-m",
        "symbols": ("BTCUSDT",),
        "interval": "1h",
        "start_time": start,
        "end_time": start + timedelta(hours=3),
        "transport": FakeTransport(),
    }

    rest = build_binance_market_dataset(
        **build_kwargs,
        metadata_evidence={"mode": "rest", "digest": "sha256:rest"},
    )
    vision = build_binance_market_dataset(
        **build_kwargs,
        metadata_evidence={"mode": "vision", "digest": "sha256:vision"},
    )
    reordered = build_binance_market_dataset(
        **build_kwargs,
        metadata_evidence={"digest": "sha256:rest", "mode": "rest"},
    )

    assert rest.dataset.dataset_id != vision.dataset.dataset_id
    assert rest.dataset.dataset_id == reordered.dataset.dataset_id


def test_coin_m_fails_closed_before_linear_dataset_publication() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)

    with pytest.raises(BinanceUnsupportedContractError, match="inverse"):
        build_binance_market_dataset(
            market="coin-m",
            symbols=("BTCUSD_PERP",),
            interval="1h",
            start_time=start,
            end_time=start + timedelta(hours=3),
            transport=FakeTransport(),
        )


def test_vision_urls_are_official_and_market_specific() -> None:
    day = datetime(2026, 6, 1, tzinfo=UTC)
    assert vision_kline_url(
        BinanceMarket.USDS_M,
        "BTCUSDT",
        "1h",
        day,
    ) == (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-06-01.zip"
    )
    assert vision_funding_url(BinanceMarket.USDS_M, "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        "BTCUSDT/BTCUSDT-fundingRate-2026-06.zip"
    )


def test_auto_transport_falls_back_from_rest_to_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    rows = [[1, "1", "1", "1", "1", "1", 2, "1", 1, "1", "1", "0"]]

    def fail_rest(**_: object) -> list[list[object]]:
        raise BinanceTransportError("HTTP 451")

    monkeypatch.setattr(transport, "_load_rest_klines", fail_rest)
    monkeypatch.setattr(transport, "_load_vision_klines", lambda **_: rows)

    observed, source = transport.load_klines(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        interval="1h",
        start_ms=0,
        end_ms=3_600_000,
        mode=BinanceTransportMode.AUTO,
    )

    assert observed == rows
    assert source == "vision"


def test_vision_funding_uses_rest_for_partial_trailing_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    start = datetime(2026, 6, 1, tzinfo=UTC)
    july = datetime(2026, 7, 1, tzinfo=UTC)
    end = datetime(2026, 7, 15, tzinfo=UTC)
    calls: list[tuple[str, int, int]] = []

    def vision(**kwargs: object) -> list[tuple[int, float]]:
        calls.append(("vision", int(kwargs["start_ms"]), int(kwargs["end_ms"])))
        return [(_ms(start), 0.0001)]

    def rest(**kwargs: object) -> list[tuple[int, float]]:
        calls.append(("rest", int(kwargs["start_ms"]), int(kwargs["end_ms"])))
        return [(_ms(july), 0.0002)]

    monkeypatch.setattr(transport, "_load_vision_funding", vision)
    monkeypatch.setattr(transport, "_load_rest_funding", rest)

    observed, source = transport.load_funding_rates(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        start_ms=_ms(start),
        end_ms=_ms(end),
        mode=BinanceTransportMode.VISION,
    )

    assert observed == [(_ms(start), 0.0001), (_ms(july), 0.0002)]
    assert source == "vision+rest"
    assert calls == [
        ("vision", _ms(start), _ms(july)),
        ("rest", _ms(july), _ms(end)),
    ]
