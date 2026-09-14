from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.integrations.binance import (
    BinanceMarket,
    BinanceMarketDataSource,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
    vision_monthly_index_price_kline_url,
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _row(open_time: datetime, *, close: float = 100.0) -> list[Any]:
    open_ms = _ms(open_time)
    return [
        open_ms,
        str(close),
        str(close),
        str(close),
        str(close),
        "0",
        open_ms + 3_599_999,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def _csv_bytes(rows: list[list[Any]], *, header: bool = False) -> bytes:
    lines: list[str] = []
    if header:
        lines.append(
            ",".join(
                (
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                )
            )
        )
    lines.extend(",".join(str(value) for value in row) for row in rows)
    return ("\n".join(lines) + "\n").encode()


def _zip_bytes(member_name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)
    return buffer.getvalue()


def _fixture(
    *,
    header: bool = False,
    member_name: str = "BTCUSDT-1h-2022-01.csv",
    malformed_field_count: bool = False,
) -> tuple[str, bytes, bytes, str]:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    rows = [_row(start, close=100.0), _row(start + timedelta(hours=1), close=101.0)]
    if malformed_field_count:
        rows[1] = rows[1][:-1]
    payload = _zip_bytes(member_name, _csv_bytes(rows, header=header))
    digest = hashlib.sha256(payload).hexdigest()
    url = vision_monthly_index_price_kline_url(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        interval="1h",
        month="2022-01",
    )
    checksum = f"{digest}  BTCUSDT-1h-2022-01.zip\n".encode()
    return url, payload, checksum, digest


def test_index_price_vision_url_is_exact_frozen_usdm_monthly_path() -> None:
    assert vision_monthly_index_price_kline_url(
        market="usds-m",
        symbol="BTCUSDT",
        interval="1h",
        month="2022-01",
    ) == (
        "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines/"
        "BTCUSDT/1h/BTCUSDT-1h-2022-01.zip"
    )
    with pytest.raises(ValueError, match="USD|USDS|index"):
        vision_monthly_index_price_kline_url(
            market=BinanceMarket.SPOT,
            symbol="BTCUSDT",
            interval="1h",
            month="2022-01",
        )


@pytest.mark.parametrize("header", [False, True])
def test_transport_normalizes_headerless_and_headered_index_archives(
    monkeypatch: pytest.MonkeyPatch,
    header: bool,
) -> None:
    url, payload, checksum, digest = _fixture(header=header)
    transport = BinancePublicTransport(max_attempts=1)

    def request_bytes(request_url: str) -> bytes:
        if request_url == url:
            return payload
        if request_url == url + ".CHECKSUM":
            return checksum
        raise AssertionError(request_url)

    monkeypatch.setattr(transport, "_request_bytes", request_bytes)
    rows, sources = transport.load_index_price_klines(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_ms(datetime(2022, 1, 1, tzinfo=UTC)),
        end_ms=_ms(datetime(2022, 2, 1, tzinfo=UTC)),
        mode=BinanceTransportMode.VISION,
        expected_archive_sha256={url: digest},
    )

    assert sources == (url,)
    assert len(rows) == 2
    assert int(rows[0][0]) == _ms(datetime(2022, 1, 1, tzinfo=UTC))
    assert float(rows[0][4]) == 100.0
    assert float(rows[1][4]) == 101.0


@pytest.mark.parametrize("mode", [BinanceTransportMode.REST, BinanceTransportMode.AUTO])
def test_transport_rejects_rest_or_auto_fallback_for_index_history(
    mode: BinanceTransportMode,
) -> None:
    transport = BinancePublicTransport(max_attempts=1)
    with pytest.raises(BinanceTransportError, match="Vision|VISION|index"):
        transport.load_index_price_klines(
            market=BinanceMarket.USDS_M,
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_ms(datetime(2022, 1, 1, tzinfo=UTC)),
            end_ms=_ms(datetime(2022, 2, 1, tzinfo=UTC)),
            mode=mode,
            expected_archive_sha256={},
        )


def test_transport_rejects_manifest_and_provider_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, payload, checksum, digest = _fixture()
    transport = BinancePublicTransport(max_attempts=1)

    def request_bytes(request_url: str) -> bytes:
        if request_url == url:
            return payload
        if request_url == url + ".CHECKSUM":
            return checksum
        raise AssertionError(request_url)

    monkeypatch.setattr(transport, "_request_bytes", request_bytes)
    kwargs = {
        "market": BinanceMarket.USDS_M,
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_ms": _ms(datetime(2022, 1, 1, tzinfo=UTC)),
        "end_ms": _ms(datetime(2022, 2, 1, tzinfo=UTC)),
        "mode": BinanceTransportMode.VISION,
    }
    with pytest.raises(BinanceTransportError, match="manifest|digest|sha"):
        transport.load_index_price_klines(
            **kwargs,
            expected_archive_sha256={url: "0" * 64},
        )

    bad_checksum = f"{'0' * 64}  BTCUSDT-1h-2022-01.zip\n".encode()

    def bad_checksum_request(request_url: str) -> bytes:
        if request_url == url:
            return payload
        if request_url == url + ".CHECKSUM":
            return bad_checksum
        raise AssertionError(request_url)

    monkeypatch.setattr(transport, "_request_bytes", bad_checksum_request)
    with pytest.raises(BinanceTransportError, match="checksum|digest|sha"):
        transport.load_index_price_klines(
            **kwargs,
            expected_archive_sha256={url: digest},
        )


def test_transport_rejects_wrong_member_and_wrong_field_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1)
    kwargs = {
        "market": BinanceMarket.USDS_M,
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_ms": _ms(datetime(2022, 1, 1, tzinfo=UTC)),
        "end_ms": _ms(datetime(2022, 2, 1, tzinfo=UTC)),
        "mode": BinanceTransportMode.VISION,
    }

    for fixture_kwargs, pattern in (
        ({"member_name": "wrong.csv"}, "member|archive|CSV|csv"),
        ({"malformed_field_count": True}, "field|column|12"),
    ):
        url, payload, checksum, digest = _fixture(**fixture_kwargs)

        def request_bytes(request_url: str) -> bytes:
            if request_url == url:
                return payload
            if request_url == url + ".CHECKSUM":
                return checksum
            raise AssertionError(request_url)

        monkeypatch.setattr(transport, "_request_bytes", request_bytes)
        with pytest.raises(BinanceTransportError, match=pattern):
            transport.load_index_price_klines(
                **kwargs,
                expected_archive_sha256={url: digest},
            )


class _IndexFixtureTransport:
    def __init__(self) -> None:
        self.start = datetime(2022, 1, 1, tzinfo=UTC)
        self.url = vision_monthly_index_price_kline_url(
            market=BinanceMarket.USDS_M,
            symbol="BTCUSDT",
            interval="1h",
            month="2022-01",
        )
        self.calls: list[dict[str, object]] = []

    def load_index_price_klines(
        self, **kwargs: object
    ) -> tuple[list[list[Any]], tuple[str, ...]]:
        self.calls.append(dict(kwargs))
        return (
            [
                _row(self.start, close=100.0),
                _row(self.start + timedelta(hours=1), close=101.0),
            ],
            (self.url,),
        )


def test_binance_source_exposes_sparse_index_series_and_content_provenance() -> None:
    transport = _IndexFixtureTransport()
    source = BinanceMarketDataSource(
        market=BinanceMarket.USDS_M,
        interval="1h",
        start_time=transport.start,
        end_time=transport.start + timedelta(hours=2),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        index_archive_sha256={transport.url: "a" * 64},
        index_manifest_digest="b" * 64,
    )

    series = source.load_index_price("BTCUSDT", "1h")

    expected_timestamps = np.array(
        ["2022-01-01T01:00:00", "2022-01-01T02:00:00"],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(series.timestamps, expected_timestamps)
    np.testing.assert_array_equal(series.available_at, expected_timestamps)
    np.testing.assert_array_equal(series.close, np.array([100.0, 101.0]))
    assert transport.calls and transport.calls[0]["mode"] is BinanceTransportMode.VISION
    assert transport.calls[0]["expected_archive_sha256"] == {transport.url: "a" * 64}
    provenance = source.index_price_provenance
    assert provenance["source_family"] == "indexPriceKlines"
    assert provenance["manifest_digest"] == "b" * 64
    assert provenance["archive_sha256_digest"]
    assert provenance["sources"] == [transport.url]


def test_binance_source_requires_vision_and_manifest_for_index_history() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    transport = _IndexFixtureTransport()

    for mode in (BinanceTransportMode.AUTO, BinanceTransportMode.REST):
        source = BinanceMarketDataSource(
            market=BinanceMarket.USDS_M,
            interval="1h",
            start_time=start,
            end_time=start + timedelta(hours=2),
            transport_mode=mode,
            transport=transport,
            index_archive_sha256={transport.url: "a" * 64},
            index_manifest_digest="b" * 64,
        )
        with pytest.raises(ValueError, match="Vision|VISION|index"):
            source.load_index_price("BTCUSDT", "1h")

    source = BinanceMarketDataSource(
        market=BinanceMarket.USDS_M,
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=2),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
    )
    with pytest.raises(ValueError, match="manifest|sha|digest|index"):
        source.load_index_price("BTCUSDT", "1h")
