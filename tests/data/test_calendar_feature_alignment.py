"""Independent literal as-of oracle on explicitly complete native bars."""

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    NormalizationMode,
)
from trade_rl.data.features.multitimeframe import (
    align_clocked_native_feature,
    align_native_feature,
)
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars
from trade_rl.data.source import RawMarketSeries


def raw(closes: np.ndarray) -> RawMarketSeries:
    price = 100 + np.arange(len(closes), dtype=float)
    return RawMarketSeries(
        timestamps=closes,
        open=price,
        high=price + 2,
        low=price - 1,
        close=price,
        volume=np.ones(len(price)),
        funding_rate=np.zeros(len(price)),
        funding_available=np.zeros(len(price), dtype=bool),
        tradable=np.ones(len(price), dtype=bool),
    )


def align(
    series: RawMarketSeries,
    clock: NativeBarClock,
    base: np.ndarray,
    *,
    spec: FeatureSpec | None = None,
    contract: InstrumentContract | None = None,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return align_clocked_native_feature(
        spec
        or FeatureSpec(
            "cloud", FeatureKind.ICHIMOKU_CLOUD_POSITION, max_staleness_hours=800
        ),
        series,
        contract or InstrumentContract("SYNTH"),
        base,
        np.ones(len(base), dtype=bool) if active is None else active,
        clock=clock,
    )


@pytest.mark.parametrize(
    "kind",
    [
        NativeClockKind.UTC_WEEK,
        NativeClockKind.UTC_MONTH,
        NativeClockKind.FIXED_ELAPSED,
    ],
)
def test_clocked_cloud_literal_oracle(kind: NativeClockKind) -> None:
    if kind == NativeClockKind.UTC_MONTH:
        closes = (np.datetime64("2020-01", "M") + np.arange(80)).astype(
            "datetime64[ns]"
        )
        clock = NativeBarClock(kind)
    else:
        days = 7 if kind == NativeClockKind.UTC_WEEK else 3
        closes = (
            np.datetime64("2020-01-06", "D") + np.arange(80) * np.timedelta64(days, "D")
        ).astype("datetime64[ns]")
        clock = (
            NativeBarClock(kind)
            if days == 7
            else NativeBarClock(kind, 3 * 86400, closes[0])
        )
    series = raw(closes)
    delay = closes.copy()
    delay[51] = closes[53] + np.timedelta64(1, "h")
    series = replace(series, available_at=delay)
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "h"))))
    observed = align(series, clock, base)
    values, ready, age, staleness = observed
    for i, t in enumerate(base):
        candidates = [
            j
            for j in range(51, len(closes))
            if all(delay[k] <= t for k in range(j - 51, j + 1))
        ]
        if not candidates:
            assert not ready[i] and values[i] == 0
            continue
        j = candidates[-1]
        hours = int(t.astype(np.int64)) - int(closes[j].astype(np.int64))
        hours /= 3_600_000_000_000
        assert age[i] == hours
        assert staleness[i] == min(hours / 800, 1)
        assert ready[i] == (hours <= 800)

        def mid(n: int) -> float:
            return (
                max(series.high[j - n + 1 : j + 1]) + min(series.low[j - n + 1 : j + 1])
            ) / 2

        expected = (
            series.close[j] - ((mid(9) + mid(26)) / 2 + mid(52)) / 2
        ) / series.close[j]
        assert values[i] == pytest.approx(expected if ready[i] else 0)
    cut = base < closes[70]
    prefix = raw(closes[:70])
    prefix = replace(prefix, available_at=delay[:70])
    altered = series.close.copy()
    altered[70:] *= 2
    mutated = replace(
        series, open=altered, close=altered, high=altered + 2, low=altered - 1
    )
    for candidate in (prefix, mutated):
        for actual, earlier in zip(
            observed, align(candidate, clock, base[cut]), strict=True
        ):
            np.testing.assert_array_equal(actual[cut], earlier)


def test_monthly_incomplete_grid_and_bad_base_are_rejected() -> None:
    closes = np.arange("2024-01", "2024-05", dtype="datetime64[M]")
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    with pytest.raises(ValueError, match="consecutive"):
        align(raw(closes[[0, 2, 3]]), clock, closes)
    for bad in (closes[::-1], closes[[0, 0]], np.array([np.datetime64("NaT")])):
        with pytest.raises(ValueError):
            align(raw(closes), clock, bad)
    with pytest.raises(ValueError):
        align(raw(closes), clock, closes, active=np.ones(4, dtype=int))
    with pytest.raises(ValueError):
        align(
            raw(closes),
            clock,
            closes,
            active=np.ma.array(np.ones(4, dtype=bool), mask=[0, 0, 1, 0]),
        )
    with pytest.raises(ValueError, match="timeframe"):
        align(
            raw(closes),
            clock,
            closes,
            spec=FeatureSpec("body", FeatureKind.BODY_RETURN, timeframe="1d"),
        )


def test_partial_listing_bar_not_used_for_warmup() -> None:
    closes = np.arange("2024-01", "2024-05", dtype="datetime64[M]")
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    contract = InstrumentContract(
        "SYNTH", listed_at=datetime(2024, 1, 15, tzinfo=timezone.utc)
    )
    spec = FeatureSpec("ret", FeatureKind.LOG_RETURN, lookback=1)
    _, ready, _, _ = align(raw(closes), clock, closes, spec=spec, contract=contract)
    np.testing.assert_array_equal(ready, [False, False, False, True])


@pytest.mark.parametrize(
    "kind",
    [
        FeatureKind.FUNDING_BPS,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.ROLLING_BETA_TO_BTC,
    ],
)
def test_unsupported_semantics_rejected_before_warmup(kind: FeatureKind) -> None:
    closes = np.arange("2024-01", "2024-05", dtype="datetime64[M]")
    with pytest.raises(ValueError, match="single-instrument price/volume"):
        align(
            raw(closes),
            NativeBarClock(NativeClockKind.UTC_MONTH),
            closes,
            spec=FeatureSpec("unsupported", kind),
        )


def test_delisting_masks_stale_features_even_if_caller_active() -> None:
    closes = np.arange("2024-01", "2024-05", dtype="datetime64[M]")
    contract = InstrumentContract(
        "SYNTH", delisted_at=datetime(2024, 3, 15, tzinfo=timezone.utc)
    )
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN, max_staleness_hours=2000)
    _, ready, _, _ = align(
        raw(closes),
        NativeBarClock(NativeClockKind.UTC_MONTH),
        closes,
        contract=contract,
        spec=spec,
    )
    np.testing.assert_array_equal(ready, [True, True, True, False])


@pytest.mark.parametrize(
    "kind",
    [
        FeatureKind.LOG_RETURN,
        FeatureKind.RSI,
        FeatureKind.ICHIMOKU_CLOUD_POSITION,
        FeatureKind.VOLUME_ZSCORE,
    ],
)
def test_legacy_daily_exact_parity(kind: FeatureKind) -> None:
    closes = np.arange("2024-01-01", "2024-04-01", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    series = raw(closes)
    spec = FeatureSpec(
        "test",
        kind,
        lookback=9,
        normalization=NormalizationMode.ROLLING_ZSCORE,
        normalization_window=5,
        min_periods=3,
    )
    base = closes + np.timedelta64(1, "h")
    active = np.ones(len(base), dtype=bool)
    old = align_native_feature(
        spec, series, InstrumentContract("SYNTH"), base, active, timeframe="1d"
    )
    new = align(
        series,
        NativeBarClock(NativeClockKind.FIXED_ELAPSED, 86400, closes[0]),
        base,
        spec=spec,
    )
    for actual, expected in zip(new, old, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_age_across_signed_nanosecond_span_does_not_wrap() -> None:
    closes = np.array([np.datetime64("1701-01-01", "ns")])
    base = np.array([np.datetime64("2200-01-01", "ns")])
    contract = InstrumentContract(
        "SYNTH", listed_at=datetime(1700, 1, 1, tzinfo=timezone.utc)
    )
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN)
    _, ready, age, stale = align(
        raw(closes),
        NativeBarClock(NativeClockKind.UTC_MONTH),
        base,
        spec=spec,
        contract=contract,
    )
    assert not ready[0] and stale[0] == 1
    assert age[0] == (datetime(2200, 1, 1) - datetime(1701, 1, 1)).days * 24


def test_complete_daily_to_calendar_feature_publication_and_staleness() -> None:
    daily = np.arange("2024-01-02", "2024-04-02", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    prices = 100 + np.arange(len(daily), dtype=float)
    published = daily.copy()
    published[daily == np.datetime64("2024-02-01")] = np.datetime64(
        "2024-03-01T01:00", "ns"
    )
    months = np.arange("2024-02", "2024-05", dtype="datetime64[M]")
    month_clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    result = resample_complete_native_bars(
        source_clock=NativeBarClock(NativeClockKind.FIXED_ELAPSED, 86400, daily[0]),
        source_closes=daily,
        source_available_at=published,
        source_ohlcv=np.column_stack(
            (prices, prices + 1, prices - 1, prices, np.ones(len(daily)), prices)
        ),
        target_clock=month_clock,
        target_closes=months,
        as_of=np.datetime64("2024-04-02", "ns"),
    )
    np.testing.assert_array_equal(result.member_counts, [31, 29, 31])
    n = len(months)
    series = RawMarketSeries(
        timestamps=result.closes,
        available_at=result.available_at,
        open=result.ohlcv[:, 0],
        high=result.ohlcv[:, 1],
        low=result.ohlcv[:, 2],
        close=result.ohlcv[:, 3],
        volume=result.ohlcv[:, 4],
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        tradable=np.ones(n, dtype=bool),
    )
    base = np.array(
        [
            "2024-03-01T00:00",
            "2024-03-01T01:00",
            "2024-03-01T02:00",
            "2024-03-01T03:00",
            "2024-04-01T00:00",
        ],
        dtype="datetime64[ns]",
    )
    spec = FeatureSpec("ret", FeatureKind.LOG_RETURN, lookback=1, max_staleness_hours=2)
    values, ready, age, stale = align(series, month_clock, base, spec=spec)
    np.testing.assert_array_equal(ready, [False, True, True, False, True])
    np.testing.assert_array_equal(age, [2, 1, 2, 3, 0])
    np.testing.assert_array_equal(stale, [1, 0.5, 1, 1, 0])
    np.testing.assert_allclose(
        values, [0, np.log(159 / 130), np.log(159 / 130), 0, np.log(190 / 159)]
    )
