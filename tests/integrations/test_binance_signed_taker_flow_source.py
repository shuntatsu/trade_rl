from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.integrations.binance import BinanceMarketDataSource


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _row(
    open_time: datetime,
    *,
    quote_volume: float,
    taker_buy_quote_volume: float,
    include_taker: bool = True,
) -> list[Any]:
    open_ms = _ms(open_time)
    row: list[Any] = [
        open_ms,
        "100",
        "102",
        "99",
        "101",
        "12",
        open_ms + 3_599_999,
        str(quote_volume),
        10,
        "5",
        str(taker_buy_quote_volume),
        "0",
    ]
    return row if include_taker else row[:8]


class _Transport:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows

    def load_klines(self, **_: object) -> tuple[list[list[Any]], str]:
        return self.rows, "fixture:klines"

    def load_funding_rates(self, **_: object) -> tuple[list[tuple[int, float]], str]:
        return [], "fixture:funding"


def _source(rows: list[list[Any]]) -> BinanceMarketDataSource:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    return BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=len(rows)),
        transport=_Transport(rows),
    )


def test_binance_standard_kline_preserves_taker_buy_quote_volume() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    source = _source(
        [
            _row(start, quote_volume=10.0, taker_buy_quote_volume=6.0),
            _row(
                start + timedelta(hours=1),
                quote_volume=20.0,
                taker_buy_quote_volume=7.0,
            ),
        ]
    )

    series = source.load("BTCUSDT")

    assert series.taker_buy_quote_volume is not None
    np.testing.assert_allclose(series.taker_buy_quote_volume, [6.0, 7.0])
    assert not series.taker_buy_quote_volume.flags.writeable


def test_binance_legacy_short_kline_may_omit_taker_field() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    source = _source(
        [
            _row(
                start,
                quote_volume=10.0,
                taker_buy_quote_volume=6.0,
                include_taker=False,
            ),
            _row(
                start + timedelta(hours=1),
                quote_volume=20.0,
                taker_buy_quote_volume=7.0,
                include_taker=False,
            ),
        ]
    )

    series = source.load("BTCUSDT")

    assert series.taker_buy_quote_volume is None
    np.testing.assert_allclose(series.volume, [10.0, 20.0])


def test_binance_mixed_kline_taker_layout_fails_closed() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    source = _source(
        [
            _row(start, quote_volume=10.0, taker_buy_quote_volume=6.0),
            _row(
                start + timedelta(hours=1),
                quote_volume=20.0,
                taker_buy_quote_volume=7.0,
                include_taker=False,
            ),
        ]
    )

    with pytest.raises(ValueError, match="taker|layout"):
        source.load("BTCUSDT")
