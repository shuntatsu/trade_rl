"""Chart-visible native cloud publication and as-of alignment, synthetic only."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.contracts import InstrumentContract
from trade_rl.data.features.indicators import ichimoku_cloud as cloud
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.source import NativeRawMarketSeries, RawMarketSeries


def _raw(closes, *, received=None, tradable=None):
    n = len(closes)
    prices = 100.0 + np.arange(n) / 8 + np.sin(np.arange(n))
    return NativeRawMarketSeries(
        timestamps=closes,
        available_at=closes if received is None else received,
        open=prices,
        high=prices + 1,
        low=prices - 2,
        close=prices,
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        funding_event_count=np.zeros(n, dtype=np.int32),
        tradable=np.ones(n, dtype=bool) if tradable is None else tradable,
    )


def _grid(name, n=130):
    start = np.datetime64("2010-01-04", "ns")
    if name == "month":
        return (
            np.arange("2010-02", "2020-12", dtype="datetime64[M]")[:n].astype(
                "datetime64[ns]"
            ),
            NativeBarClock(NativeClockKind.UTC_MONTH),
        )
    if name == "week":
        return start + np.arange(1, n + 1) * np.timedelta64(7, "D"), NativeBarClock(
            NativeClockKind.UTC_WEEK
        )
    seconds = int(name)
    return (
        start + np.arange(1, n + 1) * np.timedelta64(seconds, "s"),
        NativeBarClock(NativeClockKind.FIXED_ELAPSED, seconds, start),
    )


def _oracle(raw, *, periods, starts, listed=None, delisted=None):
    a, b, c, d = periods
    n = len(raw.close)
    span_a = np.full(n, np.nan)
    span_b = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    publication = np.full(n, np.datetime64("NaT", "ns"))
    for t in range(max(a, b, c) - 1 + d, n):
        source = t - d
        first = source - max(a, b, c) + 1
        if not all(raw.tradable[first : t + 1]):
            continue
        if listed is not None and starts[first] < listed:
            continue
        if delisted is not None and raw.timestamps[t] >= delisted:
            continue

        def mid(period):
            return (
                max(raw.high[source - period + 1 : source + 1])
                + min(raw.low[source - period + 1 : source + 1])
            ) * 0.5

        span_a[t] = (mid(a) + mid(b)) * 0.5
        span_b[t] = mid(c)
        valid[t] = True
        publication[t] = max(raw.available_at[first : t + 1])
    return span_a, span_b, valid, publication


def _options(periods):
    return dict(
        zip(
            ("tenkan_period", "kijun_period", "span_b_period", "displacement"),
            periods,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "cadence",
    [
        "900",
        "1800",
        "3600",
        "7200",
        "14400",
        "21600",
        "28800",
        "43200",
        "86400",
        "259200",
        "week",
        "month",
    ],
)
@pytest.mark.parametrize(
    "periods", [(9, 26, 52, 26), (12, 3, 5, 4), (2, 11, 3, 1), (84, 42, 60, 30)]
)
def test_visible_events_literal_and_publication(cadence, periods):
    closes, clock = _grid(cadence)
    delay = closes.copy()
    delay[17] = closes[92] + np.timedelta64(1, "ns")
    delay[77] = closes[100]
    raw = _raw(closes, received=delay)
    a, b, valid, at = _oracle(
        raw, periods=periods, starts=clock.interval_bounds(closes)[0]
    )
    events = cloud.visible_ichimoku_events(
        raw, clock=clock, contract=InstrumentContract("SYNTH"), **_options(periods)
    )
    for actual, expected in (
        (events.span_a, a),
        (events.span_b, b),
        (events.top, np.maximum(a, b)),
        (events.bottom, np.minimum(a, b)),
        (events.valid, valid),
        (events.available_at, at),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    assert np.flatnonzero(events.valid)[0] == max(periods[:3]) - 1 + periods[3]


def test_align_visible_cloud_uses_latest_published_close_once(monkeypatch):
    closes, clock = _grid("month")
    receipts = closes.copy()
    receipts[77] = closes[82] + np.timedelta64(1, "ns")
    receipts[80] = closes[85]
    tradable = np.ones(len(closes), dtype=bool)
    tradable[4] = False
    raw = _raw(closes, received=receipts, tradable=tradable)
    periods = (9, 26, 52, 26)
    a, b, valid, at = _oracle(
        raw, periods=periods, starts=clock.interval_bounds(closes)[0]
    )
    base = np.sort(
        np.concatenate(
            (closes, closes + np.timedelta64(1, "ns"), closes + np.timedelta64(25, "D"))
        )
    )
    active = np.arange(len(base)) % 13 != 0
    calls = []
    original = cloud._rolling_midpoints

    def counted(high, low, period):
        calls.append(period)
        return original(high, low, period)

    monkeypatch.setattr(cloud, "_rolling_midpoints", counted)
    result = cloud.align_visible_ichimoku_cloud(
        raw,
        clock=clock,
        contract=InstrumentContract("SYNTH"),
        base_timestamps=base,
        base_active=active,
        max_staleness_hours=480,
        **_options(periods),
    )
    assert sorted(calls) == [9, 26, 52]
    expected_a = np.full(len(base), np.nan)
    expected_b = np.full(len(base), np.nan)
    expected_available = np.zeros(len(base), dtype=bool)
    expected_age = np.full(len(base), 480.0)
    expected_stale = np.ones(len(base))
    for i, t in enumerate(base):
        candidates = np.flatnonzero(valid & (at <= t))
        if not active[i] or not len(candidates):
            continue
        j = candidates[-1]
        age = (
            float(int(t.astype(np.int64)) - int(closes[j].astype(np.int64)))
            / 3_600_000_000_000
        )
        expected_age[i] = age
        expected_stale[i] = min(age / 480, 1)
        if age <= 480 + 1e-12:
            expected_a[i], expected_b[i], expected_available[i] = a[j], b[j], True
    for actual, expected in (
        (result.span_a, expected_a),
        (result.span_b, expected_b),
        (result.top, np.maximum(expected_a, expected_b)),
        (result.bottom, np.minimum(expected_a, expected_b)),
        (result.available, expected_available),
        (result.age_hours, expected_age),
        (result.staleness, expected_stale),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable


def test_prefix_and_future_receipt_mutation_are_invariant():
    closes, clock = _grid("week")
    raw = _raw(closes)
    contract = InstrumentContract("SYNTH")
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "h"))))
    kwargs = dict(clock=clock, contract=contract, max_staleness_hours=200)
    expected = cloud.align_visible_ichimoku_cloud(
        raw, base_timestamps=base, base_active=np.ones(len(base), bool), **kwargs
    )
    cut = 100
    short = _raw(closes[:cut])
    changed = raw.close.copy()
    changed[cut:] *= 2
    delay = closes.copy()
    delay[cut:] += np.timedelta64(50, "D")
    mutated = replace(
        raw,
        open=changed,
        close=changed,
        high=changed + 1,
        low=changed - 2,
        available_at=delay,
    )
    before = base < closes[cut]
    for candidate in (short, mutated):
        actual = cloud.align_visible_ichimoku_cloud(
            candidate,
            base_timestamps=base[before],
            base_active=np.ones(before.sum(), bool),
            **kwargs,
        )
        for field in (
            "span_a",
            "span_b",
            "top",
            "bottom",
            "available",
            "age_hours",
            "staleness",
        ):
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(expected, field)[before]
            )


def test_full_lifecycle_and_longest_gap_window():
    closes, clock = _grid("month")
    tradable = np.ones(len(closes), bool)
    tradable[5] = False
    receipts = closes.copy()
    receipts[18] = closes[30] + np.timedelta64(1, "ns")
    raw = _raw(closes, received=receipts, tradable=tradable)
    contract = InstrumentContract(
        "SYNTH",
        listed_at=datetime(2010, 3, 15, tzinfo=UTC),
        delisted_at=datetime(2012, 9, 1, tzinfo=UTC),
    )
    periods = (13, 3, 4, 5)
    kwargs = dict(clock=clock, contract=contract, **_options(periods))
    events = cloud.visible_ichimoku_events(raw, **kwargs)
    a, b, valid, at = _oracle(
        raw,
        periods=periods,
        starts=clock.interval_bounds(closes)[0],
        listed=np.datetime64("2010-03-15", "ns"),
        delisted=np.datetime64("2012-09-01", "ns"),
    )
    np.testing.assert_array_equal(events.valid, valid)
    np.testing.assert_array_equal(events.available_at, at)
    np.testing.assert_array_equal(events.span_a, a)
    np.testing.assert_array_equal(events.span_b, b)
    base = np.sort(np.append(closes, receipts[18]))
    result = cloud.align_visible_ichimoku_cloud(
        raw,
        base_timestamps=base,
        base_active=np.ones(len(base), bool),
        max_staleness_hours=20000,
        **kwargs,
    )
    assert not result.available[base >= np.datetime64("2012-09-01", "ns")].any()
    assert not result.available[base <= closes[30]].any()
    assert result.available[base == receipts[18]].item()


def test_strict_boundary_missing_grid_and_insufficient_warmup():
    closes, clock = _grid("week")
    raw = _raw(closes)
    contract = InstrumentContract("SYNTH")
    kwargs = dict(clock=clock, contract=contract)
    with pytest.raises(ValueError, match="consecutive"):
        cloud.visible_ichimoku_events(_raw(np.delete(closes, 10)), **kwargs)
    with pytest.raises(ValueError, match="NativeRawMarketSeries"):
        cloud.visible_ichimoku_events(
            RawMarketSeries(
                raw.timestamps,
                raw.open,
                raw.high,
                raw.low,
                raw.close,
                raw.volume,
                raw.funding_rate,
                raw.tradable,
            ),
            **kwargs,
        )
    events = cloud.visible_ichimoku_events(raw, displacement=10**80, **kwargs)
    assert not events.valid.any()
    assert np.isnan(events.span_a).all()
    assert np.isnat(events.available_at).all()
    for bad in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integers"):
            cloud.visible_ichimoku_events(raw, tenkan_period=bad, **kwargs)
    for bad in (True, 0, float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="max_staleness_hours"):
            cloud.align_visible_ichimoku_cloud(
                raw,
                base_timestamps=closes,
                base_active=np.ones(len(closes), bool),
                max_staleness_hours=bad,
                **kwargs,
            )
    for bad in (closes[::-1], closes[[0, 0]], closes[:0]):
        with pytest.raises(ValueError, match="base timestamps"):
            cloud.align_visible_ichimoku_cloud(
                raw,
                base_timestamps=bad,
                base_active=np.ones(len(bad), bool),
                max_staleness_hours=100,
                **kwargs,
            )
    with pytest.raises(ValueError, match="base active"):
        cloud.align_visible_ichimoku_cloud(
            raw,
            base_timestamps=closes,
            base_active=np.ones(len(closes)),
            max_staleness_hours=100,
            **kwargs,
        )


def test_extreme_finite_prices_keep_existing_cloud_arithmetic():
    closes, clock = _grid("week")
    raw = _raw(closes)
    p = np.full(len(closes), 1e308)
    raw = replace(raw, open=p, high=p, low=p, close=p)
    old = cloud.visible_ichimoku_cloud(raw)
    events = cloud.visible_ichimoku_events(
        raw, clock=clock, contract=InstrumentContract("SYNTH")
    )
    np.testing.assert_array_equal(events.span_a, old.span_a)
    np.testing.assert_array_equal(events.span_b, old.span_b)
    assert np.all(events.span_a[77:] == 1e308)


@pytest.mark.parametrize("minutes", [15, 60])
def test_three_year_aligned_visible_cloud_crosses_batches(minutes):
    closes, clock = _grid("month")
    receipts = closes.copy()
    receipts[85] = closes[87] + np.timedelta64(1, "ns")
    raw = _raw(closes, received=receipts)
    base = np.arange(
        np.datetime64("2017-01-01", "ns"),
        np.datetime64("2020-01-01", "ns"),
        np.timedelta64(minutes, "m"),
    )
    active = np.arange(len(base)) % 127 != 0
    contract = InstrumentContract("SYNTH")
    result = cloud.align_visible_ichimoku_cloud(
        raw,
        clock=clock,
        contract=contract,
        base_timestamps=base,
        base_active=active,
        max_staleness_hours=720,
    )
    a, b, valid, at = _oracle(
        raw, periods=(9, 26, 52, 26), starts=clock.interval_bounds(closes)[0]
    )
    # Independent scalar oracle at every row, including both sides of each
    # batch boundary, future/late receipts and exact elapsed-hour staleness.
    for i, t in enumerate(base):
        eligible = np.flatnonzero(valid & (at <= t))
        if not active[i] or not len(eligible):
            assert not result.available[i]
            assert result.age_hours[i] == 720
            assert result.staleness[i] == 1
            continue
        j = eligible[-1]
        age = (
            float(int(t.astype(np.int64)) - int(closes[j].astype(np.int64)))
            / 3_600_000_000_000
        )
        assert result.age_hours[i] == age
        assert result.staleness[i] == min(age / 720, 1)
        fresh = age <= 720 + 1e-12
        assert result.available[i] == fresh
        if fresh:
            assert result.span_a[i] == a[j]
            assert result.span_b[i] == b[j]
            assert result.top[i] == max(a[j], b[j])
            assert result.bottom[i] == min(a[j], b[j])
    assert np.isnan(result.top[~result.available]).all()
    assert np.isnan(result.bottom[~result.available]).all()
    assert np.isnan(result.span_a[~result.available]).all()
    assert np.isnan(result.span_b[~result.available]).all()


def test_insufficient_warmup_aligned_nan_and_explicit_clock_refusals():
    closes, clock = _grid("week", n=20)
    raw = _raw(closes)
    contract = InstrumentContract("SYNTH")
    kwargs = dict(
        clock=clock,
        contract=contract,
        base_timestamps=closes,
        base_active=np.ones(len(closes), bool),
    )
    result = cloud.align_visible_ichimoku_cloud(raw, **kwargs, max_staleness_hours=720)
    assert not result.available.any()
    assert np.isnan(result.top).all() and np.isnan(result.bottom).all()
    assert np.isnan(result.span_a).all() and np.isnan(result.span_b).all()
    assert np.all(result.age_hours == 720) and np.all(result.staleness == 1)
    with pytest.raises(ValueError, match="native clock"):
        cloud.visible_ichimoku_events(raw, clock="1w", contract=contract)
    with pytest.raises(ValueError, match="instrument contract"):
        cloud.visible_ichimoku_events(raw, clock=clock, contract="SYNTH")
    with pytest.raises(ValueError, match="masked"):
        cloud.align_visible_ichimoku_cloud(
            raw,
            **(
                kwargs
                | {"base_active": np.ma.array(np.ones(len(closes), bool), mask=True)}
            ),
            max_staleness_hours=720,
        )
