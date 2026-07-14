from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.integrations.binance import (
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

    def load_exchange_information(
        self, **_: object
    ) -> tuple[dict[str, object], str]:
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


def test_auto_transport_falls_back_from_rest_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
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
