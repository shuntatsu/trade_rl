from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import (
    CsvMarketDataSource,
    InMemoryMarketDataSource,
    RawMarketSeries,
)


def _raw_series(
    *,
    timestamps: np.ndarray,
    available_at: np.ndarray | None = None,
) -> RawMarketSeries:
    size = np.asarray(timestamps).size
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=np.full(size, 100.0),
        high=np.full(size, 101.0),
        low=np.full(size, 99.0),
        close=np.full(size, 100.0),
        volume=np.ones(size),
        funding_rate=np.zeros(size),
        tradable=np.ones(size, dtype=np.bool_),
    )


def _csv_source(
    tmp_path: Path,
    rows: list[tuple[str, str]],
) -> CsvMarketDataSource:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ("timestamp", "available_at", "open", "high", "low", "close", "volume")
    )
    for timestamp, available_at in rows:
        writer.writerow((timestamp, available_at, 100, 101, 99, 100, 1))
    (tmp_path / "BTCUSDT.csv").write_text(output.getvalue(), encoding="utf-8")
    return CsvMarketDataSource(tmp_path)


def test_raw_market_series_rejects_duplicates_and_is_read_only() -> None:
    timestamps = np.array(
        ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    series = RawMarketSeries(
        timestamps=timestamps,
        open=np.array([100.0, 101.0]),
        high=np.array([101.0, 102.0]),
        low=np.array([99.0, 100.0]),
        close=np.array([100.5, 101.5]),
        volume=np.array([10.0, 11.0]),
        funding_rate=np.array([0.0, 0.0001]),
        tradable=np.array([True, False]),
    )

    for values in (
        series.timestamps,
        series.available_at,
        series.open,
        series.high,
        series.low,
        series.close,
        series.volume,
        series.funding_rate,
        series.tradable,
        series.funding_available,
        series.funding_event_count,
    ):
        assert values is not None
        assert values.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE|writeable|writable"):
            values.setflags(write=True)

    with pytest.raises(ValueError, match="strictly increasing"):
        RawMarketSeries(
            timestamps=np.array(
                ["2026-01-01T00:00:00", "2026-01-01T00:00:00"],
                dtype="datetime64[ns]",
            ),
            open=np.array([100.0, 100.0]),
            high=np.array([101.0, 101.0]),
            low=np.array([99.0, 99.0]),
            close=np.array([100.0, 100.0]),
            volume=np.array([1.0, 1.0]),
            funding_rate=np.zeros(2),
            tradable=np.ones(2, dtype=np.bool_),
        )


def test_csv_source_reads_optional_columns_and_unix_milliseconds(
    tmp_path: Path,
) -> None:
    (tmp_path / "BTCUSDT.csv").write_text(
        "timestamp,available_at,open,high,low,close,volume,funding_rate,tradable\n"
        "1767225600000,1767225600000,100,102,99,101,12,0.0001,true\n"
        "1767229200000,1767229200000,101,103,100,102,13,,false\n",
        encoding="utf-8",
    )
    source = CsvMarketDataSource(tmp_path)

    series = source.load("BTCUSDT")

    assert series.timestamps.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_allclose(series.funding_rate, [0.0001, 0.0])
    np.testing.assert_array_equal(series.tradable, [True, False])


def test_csv_source_requires_explicit_available_at(tmp_path: Path) -> None:
    (tmp_path / "BTCUSDT.csv").write_text(
        "timestamp,open,high,low,close,volume\n1767225600000,100,102,99,101,12\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="available_at"):
        CsvMarketDataSource(tmp_path).load("BTCUSDT")


def test_csv_source_requires_one_file_per_symbol(tmp_path: Path) -> None:
    source = CsvMarketDataSource(tmp_path)
    with pytest.raises(FileNotFoundError, match="BTCUSDT"):
        source.load("BTCUSDT")


def test_csv_source_preserves_late_nanosecond_for_causal_builder(
    tmp_path: Path,
) -> None:
    timestamps = [f"2024-01-01T0{hour}:00:00.000000000Z" for hour in range(4)]
    available_at = timestamps.copy()
    available_at[1] = "2024-01-01T01:00:00.000000001Z"
    source = _csv_source(tmp_path, list(zip(timestamps, available_at, strict=True)))

    raw = source.load("BTCUSDT")

    assert raw.available_at is not None
    assert raw.available_at[1] == np.datetime64("2024-01-01T01:00:00.000000001", "ns")
    assert raw.available_at[1] > raw.timestamps[1]

    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT", listed_at=datetime(2020, 1, 1, tzinfo=timezone.utc)
            ),
        ),
    )

    assert dataset.available_at[1, 0] > dataset.timestamps[1]
    assert not dataset.information_available[1, 0]
    assert not dataset.feature_available[1, 0, 0]


@pytest.mark.parametrize(
    ("raw_timestamp", "expected"),
    (
        ("2024-01-01", "2024-01-01T00:00:00.000000000"),
        ("2024-01-01T00:00:00", "2024-01-01T00:00:00.000000000"),
        ("20240101T010203Z", "2024-01-01T01:02:03.000000000"),
        ("2024-W01-1T01:02:03+00:00", "2024-01-01T01:02:03.000000000"),
        ("2024W01T01.123456Z", "2024-01-01T01:00:00.123456000"),
        ("2024W01T01.123456789Z", "2024-01-01T01:00:00.123456789"),
        ("2024W01T0102.123456Z", "2024-01-01T01:02:00.123456000"),
        ("2024W01T01:02.123456Z", "2024-01-01T01:02:00.123456000"),
        ("2024W01T010203.123456Z", "2024-01-01T01:02:03.123456000"),
        ("2024W01T01:02:03.123456Z", "2024-01-01T01:02:03.123456000"),
        ("2024-01-01T01:00:00+01:00", "2024-01-01T00:00:00.000000000"),
        (
            "2024-01-01T01:00:00.123456789+01:00",
            "2024-01-01T00:00:00.123456789",
        ),
        (
            "2024-01-01T00:00:00,123456789Z",
            "2024-01-01T00:00:00.123456789",
        ),
        (
            "2024-01-01T01:02.123456Z",
            "2024-01-01T01:02:00.123456000",
        ),
        (
            "2024-01-01T01.123456Z",
            "2024-01-01T01:00:00.123456000",
        ),
        (
            "2024-01-01T01:02.123456789Z",
            "2024-01-01T01:02:00.123456789",
        ),
        (
            "2024-01-01T01.123456789Z",
            "2024-01-01T01:00:00.123456789",
        ),
        (
            "2024-01-01T01:02:03+01:02:03.5",
            "2023-12-31T23:59:59.500000000",
        ),
        (
            "2024-01-01T01:02:03.123456+01:02:03.5",
            "2023-12-31T23:59:59.623456000",
        ),
        (
            "20240101T010203.123456789Z",
            "2024-01-01T01:02:03.123456789",
        ),
        (
            "2024-01-01T00:00:00.1234567890Z",
            "2024-01-01T00:00:00.123456789",
        ),
        (
            "2262-04-11T23:47:16.854775807Z",
            "2262-04-11T23:47:16.854775807",
        ),
        (
            "1677-09-21T00:12:43.145224193Z",
            "1677-09-21T00:12:43.145224193",
        ),
    ),
)
def test_csv_source_preserves_lossless_iso_timestamp_semantics(
    tmp_path: Path,
    raw_timestamp: str,
    expected: str,
) -> None:
    series = _csv_source(tmp_path, [(raw_timestamp, raw_timestamp)]).load("BTCUSDT")

    assert series.timestamps[0] == np.datetime64(expected, "ns")
    assert series.available_at is not None
    assert series.available_at[0] == np.datetime64(expected, "ns")


@pytest.mark.parametrize(
    "raw_timestamp",
    (
        "1677-01-01T00:00:00Z",
        "2024-01-01T00:00:00.0000000001Z",
        "2262-04-11T23:47:16.854775808Z",
        "1677-09-21T00:12:43.145224192Z",
        "2024-01-01T01:02:03+01:02:03.0000001",
        "9223372036855",
    ),
)
def test_csv_source_rejects_lossy_timestamp_inputs(
    tmp_path: Path,
    raw_timestamp: str,
) -> None:
    source = _csv_source(tmp_path, [(raw_timestamp, raw_timestamp)])

    with pytest.raises(ValueError, match="losslessly representable"):
        source.load("BTCUSDT")


@pytest.mark.parametrize(
    ("field", "raw_timestamp", "unit"),
    (
        ("timestamps", "1677-01-01T00:00:00", "s"),
        ("available_at", "1677-01-01T00:00:00", "s"),
        ("timestamps", "1970-01-01T00:00:00.000000000001", "ps"),
        ("available_at", "1970-01-01T00:00:00.000000000001", "ps"),
    ),
)
def test_raw_market_series_rejects_lossy_typed_timestamp_arrays(
    field: str,
    raw_timestamp: str,
    unit: str,
) -> None:
    invalid = np.array([raw_timestamp], dtype=f"datetime64[{unit}]")
    normalized_invalid = invalid.astype("datetime64[ns]")
    timestamps = invalid if field == "timestamps" else normalized_invalid
    available_at = invalid if field == "available_at" else None

    with pytest.raises(ValueError, match=f"{field}.*exactly representable"):
        _raw_series(timestamps=timestamps, available_at=available_at)


@pytest.mark.parametrize("field", ("timestamps", "available_at"))
def test_raw_market_series_rejects_typed_nat_timestamps(field: str) -> None:
    valid = np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]")
    invalid = np.array(["NaT"], dtype="datetime64[ns]")
    timestamps = invalid if field == "timestamps" else valid
    available_at = invalid if field == "available_at" else None

    with pytest.raises(ValueError, match=f"{field}.*NaT"):
        _raw_series(timestamps=timestamps, available_at=available_at)


def test_raw_market_series_orders_timestamps_without_integer_diff_overflow() -> None:
    limits = np.iinfo(np.int64)
    descending = np.array([limits.max, limits.min + 1], dtype=np.int64).astype(
        "datetime64[ns]"
    )
    ascending = np.array([limits.min + 1, limits.max], dtype=np.int64).astype(
        "datetime64[ns]"
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        _raw_series(timestamps=descending)

    series = _raw_series(timestamps=ascending)
    np.testing.assert_array_equal(
        series.timestamps.astype(np.int64), ascending.astype(np.int64)
    )


@pytest.mark.parametrize(
    ("field", "raw_timestamp", "unit", "expected"),
    (
        (
            "timestamps",
            "2026-01-01T04",
            "h",
            "2026-01-01T04:00:00.000000000",
        ),
        (
            "available_at",
            "2026-01-01T04",
            "2h",
            "2026-01-01T04:00:00.000000000",
        ),
        (
            "timestamps",
            "1970-01-01T00:00:00.123456789000",
            "ps",
            "1970-01-01T00:00:00.123456789",
        ),
    ),
)
def test_raw_market_series_preserves_lossless_typed_timestamp_units(
    field: str,
    raw_timestamp: str,
    unit: str,
    expected: str,
) -> None:
    value = np.array([raw_timestamp], dtype=f"datetime64[{unit}]")
    expected_ns = np.datetime64(expected, "ns")
    timestamps = value if field == "timestamps" else np.array([expected_ns])
    available_at = value if field == "available_at" else None

    series = _raw_series(timestamps=timestamps, available_at=available_at)

    normalized = series.timestamps if field == "timestamps" else series.available_at
    assert normalized is not None
    assert normalized[0] == expected_ns


@pytest.mark.parametrize("field", ("timestamps", "available_at"))
def test_raw_market_series_rejects_large_week_multiplier_overflow(
    field: str,
) -> None:
    raw_datetime = np.asarray((0, -1, -2), dtype=np.int64).view(
        np.dtype("datetime64[1000000W]")
    )
    original_bytes = raw_datetime.tobytes()
    timestamps = raw_datetime
    available_at = None
    if field == "available_at":
        timestamps = np.datetime64("1969-01-01", "ns") + np.arange(3) * np.timedelta64(
            1, "h"
        )
        available_at = raw_datetime

    with pytest.raises(
        ValueError,
        match=rf"{field} values must be exactly representable as datetime64\[ns\]",
    ):
        _raw_series(timestamps=timestamps, available_at=available_at)

    assert raw_datetime.tobytes() == original_bytes


def test_raw_market_series_accepts_big_endian_large_multiplier_hours() -> None:
    ticks = np.asarray((-2, -1, 0), dtype=">i8")
    timestamps = ticks.view(np.dtype(">M8[1000000h]"))

    series = _raw_series(timestamps=timestamps)

    assert series.timestamps.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(
        series.timestamps.astype(np.int64),
        np.asarray(
            (-7_200_000_000_000_000_000, -3_600_000_000_000_000_000, 0),
            dtype=np.int64,
        ),
    )


def test_csv_source_converts_epoch_milliseconds_without_float_rounding(
    tmp_path: Path,
) -> None:
    values = (9_223_372_036_001, 9_223_372_036_002, 9_223_372_036_854)
    rows = [(str(value), str(value)) for value in values]
    series = _csv_source(tmp_path, rows).load("BTCUSDT")
    expected = np.asarray(values, dtype=np.int64) * 1_000_000

    np.testing.assert_array_equal(
        series.timestamps,
        expected.astype("datetime64[ns]"),
    )
    assert int(np.diff(series.timestamps.astype(np.int64))[0]) == 1_000_000


def test_csv_source_keeps_epoch_seconds_semantics(tmp_path: Path) -> None:
    series = _csv_source(tmp_path, [("1767225600", "1767225600")]).load("BTCUSDT")

    assert series.timestamps[0] == np.datetime64("2026-01-01T00:00:00", "ns")
