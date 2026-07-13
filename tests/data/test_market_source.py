from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.source import CsvMarketDataSource, RawMarketSeries


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

    assert not series.close.flags.writeable
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


def test_csv_source_reads_optional_columns_and_unix_milliseconds(tmp_path: Path) -> None:
    (tmp_path / "BTCUSDT.csv").write_text(
        "timestamp,open,high,low,close,volume,funding_rate,tradable\n"
        "1767225600000,100,102,99,101,12,0.0001,true\n"
        "1767229200000,101,103,100,102,13,,false\n",
        encoding="utf-8",
    )
    source = CsvMarketDataSource(tmp_path)

    series = source.load("BTCUSDT")

    assert series.timestamps.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_allclose(series.funding_rate, [0.0001, 0.0])
    np.testing.assert_array_equal(series.tradable, [True, False])


def test_csv_source_requires_one_file_per_symbol(tmp_path: Path) -> None:
    source = CsvMarketDataSource(tmp_path)
    with pytest.raises(FileNotFoundError, match="BTCUSDT"):
        source.load("BTCUSDT")
