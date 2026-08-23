from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from trade_rl.integrations.binance_cache import vision_cache_path
from trade_rl.integrations.binance_v4_cached_sources import (
    build_cached_v4_cross_market_inputs,
    load_cached_v4_funding_events,
    load_cached_v4_kline_series,
    load_cached_v4_metrics_series,
)


def _zip_csv(rows: list[list[object]], *, name: str) -> bytes:
    buffer = io.BytesIO()
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows(rows)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text.getvalue())
    return buffer.getvalue()


def _publish(cache_root: Path, url: str, payload: bytes) -> None:
    path = vision_cache_path(cache_root, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    evidence = {
        "acquired_at": "2026-08-23T00:00:00+00:00",
        "downloader": "test",
        "etag": None,
        "last_modified": None,
        "schema_version": "binance_vision_raw_cache_v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "url": url,
    }
    path.with_suffix(".json").write_text(
        json.dumps(evidence, sort_keys=True),
        encoding="utf-8",
    )


def _kline_row(open_ms: int, close: float) -> list[object]:
    return [
        open_ms,
        close - 0.5,
        close + 0.5,
        close - 1.0,
        close,
        10.0,
        open_ms + 899_999,
        1000.0,
        20,
        5.0,
        500.0,
        0,
    ]


def _kline_url(day: int, *, kind: str = "klines") -> str:
    date = f"2026-01-{day:02d}"
    if kind == "spot":
        return (
            "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m/"
            f"BTCUSDT-15m-{date}.zip"
        )
    if kind == "mark":
        return (
            "https://data.binance.vision/data/futures/um/daily/markPriceKlines/"
            f"BTCUSDT/15m/BTCUSDT-15m-{date}.zip"
        )
    return (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/"
        f"BTCUSDT-15m-{date}.zip"
    )


def test_cached_kline_loader_merges_archives_in_authored_order(tmp_path: Path) -> None:
    first = _kline_url(1, kind="spot")
    second = _kline_url(2, kind="spot")
    _publish(tmp_path, first, _zip_csv([_kline_row(1767225600000, 100.0)], name="a.csv"))
    _publish(tmp_path, second, _zip_csv([_kline_row(1767312000000, 110.0)], name="b.csv"))

    series = load_cached_v4_kline_series((first, second), cache_root=tmp_path)
    np.testing.assert_array_equal(
        series.open_time_ms,
        np.asarray([1767225600000, 1767312000000], dtype=np.int64),
    )
    np.testing.assert_allclose(series.close, [100.0, 110.0])
    assert len(series.source_digest) == 64


def test_cached_kline_loader_rejects_duplicate_or_reversed_time(tmp_path: Path) -> None:
    first = _kline_url(1, kind="spot")
    second = _kline_url(2, kind="spot")
    duplicate = _zip_csv([_kline_row(1767225600000, 101.0)], name="dup.csv")
    _publish(tmp_path, first, _zip_csv([_kline_row(1767225600000, 100.0)], name="a.csv"))
    _publish(tmp_path, second, duplicate)
    with pytest.raises(ValueError, match="strictly increasing|duplicate|order"):
        load_cached_v4_kline_series((first, second), cache_root=tmp_path)


def test_cached_loader_rejects_tampered_cache_before_parsing(tmp_path: Path) -> None:
    url = _kline_url(1, kind="spot")
    payload = _zip_csv([_kline_row(1767225600000, 100.0)], name="a.csv")
    _publish(tmp_path, url, payload)
    vision_cache_path(tmp_path, url).write_bytes(payload + b"tamper")
    with pytest.raises(Exception, match="cache|digest|size|evidence"):
        load_cached_v4_kline_series((url,), cache_root=tmp_path)


def test_cached_funding_loader_merges_event_archives(tmp_path: Path) -> None:
    first = (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-01.zip"
    )
    second = (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-02.zip"
    )
    _publish(
        tmp_path,
        first,
        _zip_csv(
            [["calc_time", "last_funding_rate"], [1767225600000, 0.001]],
            name="jan.csv",
        ),
    )
    _publish(
        tmp_path,
        second,
        _zip_csv(
            [["calc_time", "last_funding_rate"], [1769904000000, -0.001]],
            name="feb.csv",
        ),
    )
    events = load_cached_v4_funding_events((first, second), cache_root=tmp_path)
    np.testing.assert_allclose(events.rate, [0.001, -0.001])


def _metrics_url(day: int) -> str:
    return (
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
        f"BTCUSDT-metrics-2026-01-{day:02d}.zip"
    )


def _metrics_payload(timestamp: str, oi: float) -> bytes:
    return _zip_csv(
        [
            [
                "create_time",
                "symbol",
                "sum_open_interest",
                "sum_open_interest_value",
                "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ],
            [timestamp, "BTCUSDT", 1000.0, oi, 0.9, 1.1, 1.0, 1.2],
        ],
        name="metrics.csv",
    )


def test_cached_metrics_loader_merges_daily_archives(tmp_path: Path) -> None:
    first = _metrics_url(1)
    second = _metrics_url(2)
    _publish(tmp_path, first, _metrics_payload("2026-01-01 00:05:00", 100.0))
    _publish(tmp_path, second, _metrics_payload("2026-01-02 00:05:00", 110.0))
    series = load_cached_v4_metrics_series(
        (first, second),
        cache_root=tmp_path,
        expected_symbol="BTCUSDT",
    )
    np.testing.assert_allclose(series.open_interest_value, [100.0, 110.0])


def test_cached_cross_market_builder_uses_verified_archives(tmp_path: Path) -> None:
    spot_url = _kline_url(1, kind="spot")
    perp_url = _kline_url(1, kind="perp")
    mark_url = _kline_url(1, kind="mark")
    funding_url = (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-01.zip"
    )
    metrics_url = _metrics_url(1)
    for url, close in ((spot_url, 100.0), (perp_url, 101.0), (mark_url, 100.5)):
        _publish(tmp_path, url, _zip_csv([_kline_row(1767225600000, close)], name="k.csv"))
    _publish(
        tmp_path,
        funding_url,
        _zip_csv(
            [["calc_time", "last_funding_rate"], [1767226200000, 0.001]],
            name="funding.csv",
        ),
    )
    _publish(tmp_path, metrics_url, _metrics_payload("2026-01-01 00:10:00", 123.0))

    result = build_cached_v4_cross_market_inputs(
        decision_indices=np.asarray([10], dtype=np.int64),
        decision_timestamps=np.asarray(
            [np.datetime64("2026-01-01T00:15")], dtype="datetime64[ns]"
        ),
        cache_root=tmp_path,
        spot_kline_urls=(spot_url,),
        perp_kline_urls=(perp_url,),
        mark_price_kline_urls=(mark_url,),
        funding_urls=(funding_url,),
        metrics_urls=(metrics_url,),
        expected_symbol="BTCUSDT",
    )
    assert result.spot_row_available[0]
    assert result.perp_row_available[0]
    assert result.funding_event_available[0]
    assert result.derivatives_available is not None
    assert result.derivatives_available[0]
    assert result.open_interest_value is not None
    assert result.open_interest_value[0] == pytest.approx(123.0)
