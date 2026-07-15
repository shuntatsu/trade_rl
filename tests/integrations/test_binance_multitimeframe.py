from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from trade_rl.integrations.binance import (
    BinanceMarket,
    BinanceMarketDataSource,
    BinanceTransportMode,
    binance_multitimeframe_feature_specs,
    plan_vision_kline_urls,
    vision_monthly_kline_url,
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _interval_delta(interval: str) -> timedelta:
    return {
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }[interval]


def _kline(open_time: datetime, interval: str, close: float) -> list[Any]:
    delta = _interval_delta(interval)
    return [
        _ms(open_time),
        str(close - 1.0),
        str(close + 1.0),
        str(close - 2.0),
        str(close),
        "12.0",
        _ms(open_time + delta) - 1,
        "100000.0",
        10,
        "5.0",
        "50000.0",
        "0",
    ]


class CountingTransport:
    def __init__(self) -> None:
        self.kline_calls: list[tuple[str, str]] = []
        self.funding_calls = 0
        self.start = datetime(2026, 6, 1, tzinfo=UTC)

    def load_klines(
        self,
        *,
        symbol: str,
        interval: str,
        **_: object,
    ) -> tuple[list[list[Any]], tuple[str, ...]]:
        self.kline_calls.append((symbol, interval))
        delta = _interval_delta(interval)
        rows = [
            _kline(self.start + index * delta, interval, 100.0 + index)
            for index in range(2)
        ]
        return rows, (f"fixture:{symbol}:{interval}",)

    def load_funding_rates(
        self, **_: object
    ) -> tuple[list[tuple[int, float]], tuple[str, ...]]:
        self.funding_calls += 1
        return [], ("fixture:funding",)

    def load_exchange_information(self, **_: object) -> tuple[dict[str, object], str]:
        return {"symbols": []}, "fixture:exchange-info"


def test_official_monthly_vision_url_is_market_specific() -> None:
    month = datetime(2025, 12, 1, tzinfo=UTC)

    assert vision_monthly_kline_url(
        BinanceMarket.USDS_M,
        "BTCUSDT",
        "15m",
        month,
    ) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/15m/BTCUSDT-15m-2025-12.zip"
    )
    assert vision_monthly_kline_url(
        BinanceMarket.SPOT,
        "ETHUSDT",
        "1d",
        month,
    ) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "ETHUSDT/1d/ETHUSDT-1d-2025-12.zip"
    )


def test_complete_months_use_monthly_archives_and_partial_months_use_daily() -> None:
    complete = plan_vision_kline_urls(
        BinanceMarket.USDS_M,
        "BTCUSDT",
        "1h",
        datetime(2025, 12, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
    )
    partial = plan_vision_kline_urls(
        BinanceMarket.USDS_M,
        "BTCUSDT",
        "1h",
        datetime(2026, 1, 30, tzinfo=UTC),
        datetime(2026, 2, 2, tzinfo=UTC),
    )

    assert complete == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2025-12.zip",
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-01.zip",
    )
    assert all("/daily/klines/" in url for url in partial)
    assert len(partial) == 3


def test_source_caches_each_symbol_timeframe_and_shared_funding() -> None:
    transport = CountingTransport()
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=transport.start,
        end_time=transport.start + timedelta(hours=8),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
    )

    first = source.load_timeframe("BTCUSDT", "4h")
    second = source.load_timeframe("BTCUSDT", "4h")
    source.load_timeframe("BTCUSDT", "1h")

    assert first is second
    assert transport.kline_calls == [("BTCUSDT", "4h"), ("BTCUSDT", "1h")]
    assert transport.funding_calls == 1
    assert source.sources_used == (
        "fixture:BTCUSDT:1h",
        "fixture:BTCUSDT:4h",
        "fixture:funding",
    )


def test_maintained_feature_preset_uses_role_specific_extended_features() -> None:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="1h",
        feature_timeframes=("15m", "4h", "1d"),
    )

    names = tuple(spec.name for spec in specs)
    counts = {
        timeframe: sum(name.startswith(f"{timeframe}__") for name in names)
        for timeframe in ("15m", "1h", "4h", "1d")
    }
    assert len(specs) == 226
    assert counts == {"15m": 59, "1h": 59, "4h": 55, "1d": 53}
    assert "15m__bollinger_percent_b_centered_20_2" in names
    assert "1h__garman_klass_volatility_24bar" in names
    assert "4h__trend_r2_18bar" in names
    assert "1d__funding_zscore_12events" in names
    assert "15m__upper_wick_ratio" in names
    assert "1d__upper_wick_ratio" not in names
