from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.integrations.postgres_universal_source import (
    RawSymbolSource,
    UniversalSourceScope,
)
from trade_rl.workflows.native_indicator_materializer import (
    build_native_indicator_cache,
    combine_native_indicator_builds,
)


def one_symbol_minutes(start: str, count: int) -> RawSymbolSource:
    parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
    timestamps = np.datetime64(parsed.replace(tzinfo=None), "ns") + np.arange(
        count
    ) * np.timedelta64(1, "m")
    row = np.arange(count, dtype=np.float64)
    open_ = 100.0 + row * 0.01
    close = open_ + 0.005
    base_volume = 2.0 + row * 0.001
    derivative_indices = np.arange(0, count, 10)
    derivative_values = np.zeros((len(derivative_indices), 4), dtype=np.float64)
    derivative_values[:, 0] = 1_000.0 + derivative_indices
    orderflow_values = np.zeros((count, 5), dtype=np.float64)
    orderflow_values[:, 0] = 3.0 + row
    funding_offset = min(15, count - 1)
    return RawSymbolSource(
        timestamps=timestamps,
        open=open_,
        high=close + 0.01,
        low=open_ - 0.01,
        close=close,
        base_volume=base_volume,
        funding_timestamps=np.asarray(
            [timestamps[funding_offset] + np.timedelta64(30, "s")],
            dtype="datetime64[ns]",
        ),
        funding_rate=np.asarray([0.0001], dtype=np.float64),
        derivative_timestamps=(
            timestamps[derivative_indices] + np.timedelta64(1, "s")
        ),
        derivative_values=derivative_values,
        orderflow_timestamps=timestamps.copy(),
        orderflow_values=orderflow_values,
    )


def scope_for(symbol: str, minutes: int) -> UniversalSourceScope:
    start = datetime(2024, 11, 13, tzinfo=UTC)
    return UniversalSourceScope(
        symbols=(symbol,), start=start, end=start + timedelta(minutes=minutes)
    )


def complete_fixture() -> dict[str, RawSymbolSource]:
    return {"BTCUSDT": one_symbol_minutes("2024-11-13T00:00:00Z", 1_440)}


def fixture_scope() -> UniversalSourceScope:
    return scope_for("BTCUSDT", 1_440)


def test_native_bars_close_on_boundary_without_lookahead() -> None:
    source = one_symbol_minutes("2024-11-13T00:00:00Z", count=61)

    build = build_native_indicator_cache(
        {"BTCUSDT": source}, scope=scope_for("BTCUSDT", minutes=60)
    )

    bars = build.market_bars[("BTCUSDT", "15m")]
    assert bars.open_time_ms.tolist() == [
        1_731_456_000_000,
        1_731_456_900_000,
        1_731_457_800_000,
        1_731_458_700_000,
    ]
    assert bars.close[-1] == source.close[59]
    assert source.close[60] not in bars.close
    expected_quote_volume = np.sum(source.base_volume[:15] * source.close[:15])
    assert bars.quote_volume[0] == expected_quote_volume
    assert bars.funding_available.tolist() == [False, True, False, False]


def test_indicator_payload_is_deterministic_and_has_206_features() -> None:
    first = build_native_indicator_cache(complete_fixture(), scope=fixture_scope())
    second = build_native_indicator_cache(complete_fixture(), scope=fixture_scope())

    assert first.manifest.feature_count == 206
    assert first.manifest.volume_conversion_method == (
        "base_volume_times_minute_close_v1"
    )
    assert first.manifest.digest == second.manifest.digest
    assert [item.payload_sha256 for item in first.artifacts] == [
        item.payload_sha256 for item in second.artifacts
    ]
    assert sum(item.feature_count for item in first.artifacts) == 206
    assert all(item.values.dtype == np.dtype(np.float32) for item in first.artifacts)
    assert all(item.available.dtype == np.dtype(np.bool_) for item in first.artifacts)
    assert all(item.event_time_ms.dtype == np.dtype(np.int64) for item in first.artifacts)
    assert sum(len(item.payload) for item in first.artifacts) < sum(
        item.event_time_ms.nbytes + item.values.nbytes + item.available.nbytes
        for item in first.artifacts
    )


def test_report_exposes_missing_nonfinite_ohlcv_and_feature_counts() -> None:
    report = build_native_indicator_cache(
        complete_fixture(), scope=fixture_scope()
    ).report

    item = report.members[0]
    assert report.volume_conversion_method == "base_volume_times_minute_close_v1"
    assert item.duplicate_timestamps == 0
    assert item.missing_timestamps == 0
    assert item.nonfinite_available_features == 0
    assert item.ohlcv_violations == 0
    assert len(item.feature_statistics) > 0
    assert all(stat.available_count >= 0 for stat in item.feature_statistics)
    assert len(report.digest) == 64


def test_combines_single_symbol_builds_in_declared_scope_order() -> None:
    start = datetime(2024, 11, 13, tzinfo=UTC)
    end = start + timedelta(minutes=60)
    scope = UniversalSourceScope(
        symbols=("ETHUSDT", "BTCUSDT"), start=start, end=end
    )
    builds = tuple(
        build_native_indicator_cache(
            {symbol: one_symbol_minutes("2024-11-13T00:00:00Z", 60)},
            scope=UniversalSourceScope(symbols=(symbol,), start=start, end=end),
        )
        for symbol in scope.symbols
    )

    combined = combine_native_indicator_builds(builds, scope=scope)

    assert combined.manifest.symbols == scope.symbols
    assert len(combined.artifacts) == 8
    assert all(item.values.size == 0 for item in combined.artifacts)
    assert all(item.available_value_count >= 0 for item in combined.artifacts)
    assert tuple(combined.market_bars) == tuple(
        (symbol, timeframe)
        for symbol in scope.symbols
        for timeframe in ("15m", "1h", "4h", "1d")
    )
    assert combined.manifest.digest == combine_native_indicator_builds(
        builds, scope=scope
    ).manifest.digest
