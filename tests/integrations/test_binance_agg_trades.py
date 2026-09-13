from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.integrations.binance import (
    BinanceAggTradesSeries,
    BinanceMarket,
    BinanceTransportError,
    parse_vision_agg_trades_archive,
    plan_vision_agg_trades_urls,
    vision_agg_trades_url,
)

_HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)


def _zip_csv(
    rows: list[str], *, name: str = "BTCUSDT-aggTrades-2021-01-01.csv"
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, "\n".join(rows) + "\n")
    return buffer.getvalue()


def _rows() -> list[str]:
    return [
        "100,30000.5,0.25,200,202,1609459200123,true",
        "101,30001.0,0.50,203,203,1609459201123,false",
        "102,30002.0,0.75,204,207,1609459201123,true",
    ]


def _payload(*, header: bool = True, rows: list[str] | None = None) -> bytes:
    body = list(_rows() if rows is None else rows)
    if header:
        body.insert(0, ",".join(_HEADER))
    return _zip_csv(body)


def test_agg_trades_url_and_plan_are_usds_m_daily_only() -> None:
    day = datetime(2021, 1, 1, tzinfo=UTC)

    assert vision_agg_trades_url("usds-m", "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-01-01.zip"
    )
    assert plan_vision_agg_trades_urls(
        "usds-m",
        "BTCUSDT",
        day + timedelta(hours=8),
        day + timedelta(days=2, hours=1),
    ) == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-01-01.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-01-02.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-01-03.zip",
    )

    for market in (BinanceMarket.SPOT, BinanceMarket.COIN_M):
        with pytest.raises(ValueError, match="USD-M"):
            vision_agg_trades_url(market, "BTCUSDT", day)


def test_agg_trades_parser_accepts_exact_header_or_headerless_identically() -> None:
    source = (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-01-01.zip"
    )
    with_header_payload = _payload(header=True)
    without_header_payload = _payload(header=False)

    with_header = parse_vision_agg_trades_archive(with_header_payload, source=source)
    without_header = parse_vision_agg_trades_archive(
        without_header_payload,
        source=source,
    )

    assert isinstance(with_header, BinanceAggTradesSeries)
    assert with_header.header_present is True
    assert without_header.header_present is False
    np.testing.assert_array_equal(
        with_header.aggregate_trade_ids,
        np.array([100, 101, 102], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        with_header.first_trade_ids,
        np.array([200, 203, 204], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        with_header.last_trade_ids,
        np.array([202, 203, 207], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        with_header.timestamps,
        np.array(
            [
                "2021-01-01T00:00:00.123",
                "2021-01-01T00:00:01.123",
                "2021-01-01T00:00:01.123",
            ],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(
        with_header.buyer_is_maker,
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(with_header.prices, [30000.5, 30001.0, 30002.0])
    np.testing.assert_array_equal(with_header.quantities, [0.25, 0.5, 0.75])

    for field in (
        "aggregate_trade_ids",
        "prices",
        "quantities",
        "first_trade_ids",
        "last_trade_ids",
        "timestamps",
        "buyer_is_maker",
    ):
        np.testing.assert_array_equal(
            getattr(with_header, field), getattr(without_header, field)
        )
        assert getattr(with_header, field).flags.writeable is False

    assert with_header.source_uri == source
    assert (
        with_header.raw_payload_sha256
        == hashlib.sha256(with_header_payload).hexdigest()
    )
    assert with_header.raw_payload_size_bytes == len(with_header_payload)


def test_agg_trades_parser_rejects_unsupported_header() -> None:
    rows = [
        "id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
        *_rows(),
    ]
    with pytest.raises(BinanceTransportError, match="header"):
        parse_vision_agg_trades_archive(_zip_csv(rows), source="fixture")


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (["100,1,1,200,200,1609459200000"], "seven|fields"),
        (["x,1,1,200,200,1609459200000,true"], "aggregate|integer|ID"),
        (["100,0,1,200,200,1609459200000,true"], "price.*positive"),
        (["100,nan,1,200,200,1609459200000,true"], "price.*finite"),
        (["100,1,0,200,200,1609459200000,true"], "quantity.*positive"),
        (["100,1,inf,200,200,1609459200000,true"], "quantity.*finite"),
        (["100,1,1,201,200,1609459200000,true"], "first.*last"),
        (["100,1,1,200,200,-1,true"], "timestamp.*non-negative"),
        (["100,1,1,200,200,1609459200000,1"], "buyer.*maker|boolean"),
    ],
)
def test_agg_trades_parser_rejects_invalid_rows(rows: list[str], match: str) -> None:
    with pytest.raises(BinanceTransportError, match=match):
        parse_vision_agg_trades_archive(_payload(rows=rows), source="fixture")


def test_agg_trades_parser_rejects_decreasing_aggregate_ids() -> None:
    rows = [
        "101,1,1,200,200,1609459200000,true",
        "100,1,1,201,201,1609459201000,false",
    ]
    with pytest.raises(BinanceTransportError, match="aggregate.*strictly increasing"):
        parse_vision_agg_trades_archive(_payload(rows=rows), source="fixture")


def test_agg_trades_parser_rejects_decreasing_timestamps() -> None:
    rows = [
        "100,1,1,200,200,1609459201000,true",
        "101,1,1,201,201,1609459200000,false",
    ]
    with pytest.raises(BinanceTransportError, match="timestamp.*nondecreasing"):
        parse_vision_agg_trades_archive(_payload(rows=rows), source="fixture")


def test_agg_trades_parser_rejects_multiple_archive_members() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.csv", "\n".join(_rows()) + "\n")
        archive.writestr("b.csv", "\n".join(_rows()) + "\n")

    with pytest.raises(BinanceTransportError, match="exactly one"):
        parse_vision_agg_trades_archive(buffer.getvalue(), source="fixture")


def test_agg_trades_series_rejects_invalid_public_time_and_shape_evidence() -> None:
    series = parse_vision_agg_trades_archive(_payload(), source="fixture")

    with pytest.raises(ValueError, match="timestamps.*NaT"):
        replace(
            series,
            timestamps=np.array(
                ["NaT", *series.timestamps[1:]], dtype="datetime64[ns]"
            ),
        )
    with pytest.raises(ValueError, match="prices.*shape|same length"):
        replace(series, prices=np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="aggregate.*non-negative"):
        changed = series.aggregate_trade_ids.copy()
        changed[0] = -1
        replace(series, aggregate_trade_ids=changed)
