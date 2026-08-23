from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.integrations.binance_v4_context import (
    BINANCE_FUTURES_METRICS_COLUMNS,
    BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS,
    align_futures_metrics_to_decisions,
    parse_binance_funding_archive,
    parse_binance_futures_metrics_archive,
    parse_binance_v4_kline_archive,
    vision_futures_metrics_url,
)


def _zip_csv(rows: list[list[object]], *, name: str = "data.csv") -> bytes:
    buffer = io.BytesIO()
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows(rows)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text.getvalue())
    return buffer.getvalue()


def _metrics_rows(*rows: list[object]) -> list[list[object]]:
    return [list(BINANCE_FUTURES_METRICS_COLUMNS), *rows]


def _make_metrics(create_times: np.ndarray):
    rows = []
    for index, timestamp in enumerate(create_times.astype("datetime64[s]").astype(str)):
        rows.append(
            [
                timestamp.replace("T", " "),
                "BTCUSDT",
                str(1000.0 + index),
                str(100_000_000.0 + index),
                str(0.9 + index * 0.01),
                str(1.1 + index * 0.01),
                str(1.0 + index * 0.01),
                str(1.2 + index * 0.01),
            ]
        )
    return parse_binance_futures_metrics_archive(
        _zip_csv(_metrics_rows(*rows), name="BTCUSDT-metrics.csv"),
        source_uri="https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip",
        expected_symbol="BTCUSDT",
    )


def test_vision_futures_metrics_url_is_frozen() -> None:
    value = vision_futures_metrics_url(
        "BTCUSDT",
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert value == (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-01-02.zip"
    )


def test_parse_v4_kline_archive_extracts_quote_and_taker_quote_volume() -> None:
    header = [
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
    ]
    # Spot public-data timestamps may be microseconds from 2025 onward.
    row = [
        "1735689600000000",
        "100.0",
        "102.0",
        "99.0",
        "101.0",
        "12.0",
        "1735690499999999",
        "1212.0",
        "42",
        "7.0",
        "707.0",
        "0",
    ]
    series = parse_binance_v4_kline_archive(
        _zip_csv([header, row], name="BTCUSDT-15m.csv"),
        source_uri="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2025-01-01.zip",
    )
    np.testing.assert_array_equal(series.open_time_ms, np.asarray([1735689600000]))
    np.testing.assert_array_equal(series.close_time_ms, np.asarray([1735690499999]))
    np.testing.assert_allclose(series.close, [101.0])
    np.testing.assert_allclose(series.quote_volume, [1212.0])
    np.testing.assert_allclose(series.taker_buy_quote_volume, [707.0])
    assert len(series.source_digest) == 64
    assert not series.close.flags.writeable


def test_parse_v4_kline_archive_accepts_headerless_exact_twelve_columns() -> None:
    row = [
        "1735689600000",
        "100.0",
        "102.0",
        "99.0",
        "101.0",
        "12.0",
        "1735690499999",
        "1212.0",
        "42",
        "7.0",
        "707.0",
        "0",
    ]
    series = parse_binance_v4_kline_archive(
        _zip_csv([row]),
        source_uri="https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2026-01-01.zip",
    )
    assert series.close[0] == pytest.approx(101.0)


def test_parse_v4_kline_archive_rejects_short_or_inconsistent_rows() -> None:
    short = ["1735689600000", "100", "102", "99", "101"]
    with pytest.raises(ValueError, match="12|field|row"):
        parse_binance_v4_kline_archive(
            _zip_csv([short]),
            source_uri="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2026-01-01.zip",
        )

    invalid_taker = [
        "1735689600000",
        "100.0",
        "102.0",
        "99.0",
        "101.0",
        "12.0",
        "1735690499999",
        "100.0",
        "42",
        "7.0",
        "101.0",
        "0",
    ]
    with pytest.raises(ValueError, match="taker|quote"):
        parse_binance_v4_kline_archive(
            _zip_csv([invalid_taker]),
            source_uri="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2026-01-01.zip",
        )


def test_parse_funding_archive_preserves_actual_events_only() -> None:
    payload = _zip_csv(
        [
            ["calc_time", "last_funding_rate"],
            ["1767225600000", "0.00010"],
            ["1767254400000", "-0.00020"],
        ],
        name="BTCUSDT-fundingRate.csv",
    )
    series = parse_binance_funding_archive(
        payload,
        source_uri="https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-01.zip",
    )
    np.testing.assert_array_equal(
        series.event_time_ms,
        np.asarray([1767225600000, 1767254400000], dtype=np.int64),
    )
    np.testing.assert_allclose(series.rate, [0.00010, -0.00020])


def test_parse_metrics_archive_maps_header_names_not_positions() -> None:
    reordered_header = [
        "symbol",
        "create_time",
        "sum_open_interest_value",
        "sum_open_interest",
        "sum_toptrader_long_short_ratio",
        "count_toptrader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "count_long_short_ratio",
    ]
    row = [
        "BTCUSDT",
        "2026-01-01 00:10:00",
        "250000000.0",
        "1500.0",
        "1.25",
        "0.85",
        "1.15",
        "0.95",
    ]
    series = parse_binance_futures_metrics_archive(
        _zip_csv([reordered_header, row], name="BTCUSDT-metrics.csv"),
        source_uri="https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip",
        expected_symbol="BTCUSDT",
    )
    assert series.open_interest_value[0] == pytest.approx(250000000.0)
    assert series.global_long_short_ratio[0] == pytest.approx(0.95)
    assert series.top_position_long_short_ratio[0] == pytest.approx(1.25)


def test_parse_metrics_archive_rejects_missing_column_or_wrong_symbol() -> None:
    header = list(BINANCE_FUTURES_METRICS_COLUMNS[:-1])
    with pytest.raises(ValueError, match="column|header"):
        parse_binance_futures_metrics_archive(
            _zip_csv([header]),
            source_uri="https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip",
            expected_symbol="BTCUSDT",
        )

    row = [
        "2026-01-01 00:10:00",
        "ETHUSDT",
        "1500.0",
        "250000000.0",
        "0.85",
        "1.25",
        "0.95",
        "1.15",
    ]
    with pytest.raises(ValueError, match="symbol"):
        parse_binance_futures_metrics_archive(
            _zip_csv(_metrics_rows(row)),
            source_uri="https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip",
            expected_symbol="BTCUSDT",
        )


def test_future_metrics_event_is_not_visible_to_earlier_decision() -> None:
    decisions = np.asarray(
        [np.datetime64("2026-01-01T00:00"), np.datetime64("2026-01-01T00:15")],
        dtype="datetime64[ns]",
    )
    metrics = _make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:10")], dtype="datetime64[ns]")
    )
    aligned = align_futures_metrics_to_decisions(decisions, metrics)
    assert not aligned.available[0]
    assert aligned.available[1]
    assert aligned.staleness_hours[1] == pytest.approx(5.0 / 60.0)


def test_metrics_older_than_one_decision_are_unavailable() -> None:
    decisions = np.asarray([np.datetime64("2026-01-01T00:30")], dtype="datetime64[ns]")
    metrics = _make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:10")], dtype="datetime64[ns]")
    )
    aligned = align_futures_metrics_to_decisions(decisions, metrics)
    assert BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS == pytest.approx(0.25)
    assert not aligned.available[0]
    assert aligned.staleness_hours[0] == pytest.approx(20.0 / 60.0)
    assert aligned.open_interest_value[0] == pytest.approx(
        metrics.open_interest_value[0]
    )


def test_metrics_exactly_fifteen_minutes_old_remain_available() -> None:
    decisions = np.asarray([np.datetime64("2026-01-01T00:30")], dtype="datetime64[ns]")
    metrics = _make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:15")], dtype="datetime64[ns]")
    )
    aligned = align_futures_metrics_to_decisions(decisions, metrics)
    assert aligned.available[0]
    assert aligned.staleness_hours[0] == pytest.approx(0.25)


def test_future_metrics_mutation_cannot_change_existing_prefix() -> None:
    decisions = np.asarray([np.datetime64("2026-01-01T00:15")], dtype="datetime64[ns]")
    before = _make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:10")], dtype="datetime64[ns]")
    )
    after = _make_metrics(
        np.asarray(
            [
                np.datetime64("2026-01-01T00:10"),
                np.datetime64("2026-01-01T00:20"),
            ],
            dtype="datetime64[ns]",
        )
    )
    left = align_futures_metrics_to_decisions(decisions, before)
    right = align_futures_metrics_to_decisions(decisions, after)
    np.testing.assert_array_equal(left.open_interest_value, right.open_interest_value)
    np.testing.assert_array_equal(left.available, right.available)
    np.testing.assert_array_equal(left.staleness_hours, right.staleness_hours)
