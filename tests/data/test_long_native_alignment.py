"""Long-history as-of alignment: literal causal values and bounded bulk work."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features import multitimeframe as mtf
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.source import NativeRawMarketSeries

_HOUR_NS = 3_600_000_000_000


def _source(closes, receipts):
    n = len(closes)
    prices = 100.0 + np.arange(n) / 10 + np.sin(np.arange(n))
    return NativeRawMarketSeries(
        timestamps=closes,
        available_at=receipts,
        open=prices,
        high=prices + 2,
        low=prices - 1,
        close=prices + 0.25,
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        funding_event_count=np.zeros(n, dtype=np.int32),
        tradable=np.ones(n, dtype=bool),
    )


def _literal_asof(spec, raw, events, base, active):
    values, valid, receipts = events
    result = np.zeros(len(base), dtype=float)
    ready = np.zeros(len(base), dtype=bool)
    ages = np.full(len(base), spec.max_staleness_hours, dtype=float)
    stale = np.ones(len(base), dtype=float)
    # Independent scalar publication queue with latest event by native close.
    # Never derive the expected result from the bulk search implementation.
    waiting = sorted(
        (int(receipts[j].astype(np.int64)), int(j)) for j in np.flatnonzero(valid)
    )
    cursor = 0
    latest = -1
    for i, t in enumerate(base.astype(np.int64)):
        while cursor < len(waiting) and waiting[cursor][0] <= int(t):
            latest = max(latest, waiting[cursor][1])
            cursor += 1
        if latest < 0 or not active[i]:
            continue
        age = float(int(t) - int(raw.timestamps[latest].astype(np.int64))) / _HOUR_NS
        if age < -1e-12:
            raise ValueError("future event")
        ages[i] = age
        stale[i] = min(age / spec.max_staleness_hours, 1.0)
        if age <= spec.max_staleness_hours + 1e-12:
            result[i], ready[i] = values[latest], True
    return result, ready, ages, stale


@pytest.mark.parametrize("minutes", [15, 60])
def test_multiyear_monthly_cloud_bulk_alignment_matches_literal(minutes, monkeypatch):
    # 96 actual consecutive UTC monthly bars including a full 78-bar warm-up.
    closes = np.arange("2017-02", "2025-02", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    receipts = closes.copy()
    receipts[65] = closes[67] + np.timedelta64(1, "ns")
    receipts[77] = closes[78]
    raw = _source(closes, receipts)
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    spec = FeatureSpec(
        "cloud", FeatureKind.ICHIMOKU_CLOUD_POSITION, max_staleness_hours=720.0
    )
    contract = InstrumentContract("SYNTH", listed_at=datetime(2017, 1, 1, tzinfo=UTC))
    base = np.arange(
        np.datetime64("2022-01-01", "ns"),
        np.datetime64("2025-01-01", "ns"),
        np.timedelta64(minutes, "m"),
    )
    active = np.arange(len(base)) % 113 != 0
    events = mtf._native_events(spec, raw, contract)
    expected = _literal_asof(spec, raw, events, base, active)

    # Structural resource regression, not a host-dependent wall-clock limit:
    # the alignment must search by bounded base chunks, never row by row.
    original = np.searchsorted
    searches = []

    def counted(a, v, *args, **kwargs):
        searches.append(np.asarray(v).size)
        return original(a, v, *args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", counted)
    actual = mtf.align_clocked_native_feature(
        spec, raw, contract, base, active, clock=clock
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
    assert len(searches) > 1
    assert sum(searches) == len(base)
    assert max(searches) <= 8192
    assert len(searches) <= (len(base) + 8191) // 8192


def test_alignment_spans_full_finite_ns_range_without_signed_overflow():
    lo = np.datetime64(np.iinfo(np.int64).min + 1, "ns")
    hi = np.datetime64(np.iinfo(np.int64).max, "ns")
    raw = _source(np.array([lo]), np.array([lo]))
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN, max_staleness_hours=1e8)
    events = (np.array([0.125]), np.array([True]), np.array([lo]))
    base = np.array([lo, np.datetime64("1970-01-01", "ns"), hi])
    active = np.ones(len(base), dtype=bool)
    actual = mtf._align_events(spec, raw, events, base, active)
    expected = _literal_asof(spec, raw, events, base, active)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
    assert actual[2][-1] == float(2**64 - 2) / _HOUR_NS


def test_late_old_publication_equal_ties_and_inactive_base_rows():
    closes = np.datetime64("2024-01-01", "ns") + np.arange(6) * np.timedelta64(1, "h")
    receipts = closes.copy()
    receipts[0] = closes[4]
    receipts[1] = closes[3]
    receipts[2] = closes[3]
    raw = _source(closes, receipts)
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN, max_staleness_hours=1.0)
    events = (
        np.arange(6, dtype=float),
        np.array([True, True, True, False, True, True]),
        receipts,
    )
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "ns"))))
    active = np.ones(len(base), dtype=bool)
    active[6] = False
    expected = _literal_asof(spec, raw, events, base, active)
    actual = mtf._align_events(spec, raw, events, base, active)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
    # Publication at an inactive base row becomes visible at the next active row.
    assert not actual[1][6] and actual[0][7] == 2.0
    assert actual[0][8] == 4.0


def test_legacy_cursor_behavior_and_future_event_refusal_across_batches(monkeypatch):
    monkeypatch.setattr(mtf, "_ALIGNMENT_BATCH_ROWS", 1)
    # Legacy helper advanced the publication cursor even at inactive rows.
    # The explicit clock path independently rejects reversed base clocks.
    closes = np.array([np.datetime64("2024-01-01", "ns")])
    raw = _source(closes, closes)
    events = (np.array([0.5]), np.array([True]), closes)
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN)
    base = np.array(
        [closes[0] + np.timedelta64(5, "ns"), closes[0] - np.timedelta64(1, "ns")]
    )
    actual = mtf._align_events(spec, raw, events, base, np.array([False, True]))
    expected = _literal_asof(spec, raw, events, base, np.array([False, True]))
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
    with pytest.raises(ValueError, match="event timestamp"):
        mtf._align_events(
            spec,
            raw,
            events,
            np.array([base[0], closes[0] - np.timedelta64(4, "ns")]),
            np.array([False, True]),
        )


@pytest.mark.parametrize("minutes,timeframe", [(15, "15m"), (60, "1h")])
def test_dense_three_year_native_source_matches_scalar(minutes, timeframe):
    base = np.arange(
        np.datetime64("2022-01-01", "ns"),
        np.datetime64("2025-01-01", "ns"),
        np.timedelta64(minutes, "m"),
    )
    i = np.arange(len(base))
    receipts = base + np.where(i % 173 == 5, 300, i % 7) * np.timedelta64(minutes, "m")
    raw = _source(base, receipts)
    active = i % 113 != 0
    spec = FeatureSpec("body", FeatureKind.BODY_RETURN, max_staleness_hours=240.0)
    contract = InstrumentContract("SYNTH")
    events = mtf._native_events(spec, raw, contract)
    expected = _literal_asof(spec, raw, events, base, active)
    actual = mtf.align_native_feature(
        spec, raw, contract, base, active, timeframe=timeframe
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)


def test_alignment_no_events_empty_base_and_randomized_receipts(monkeypatch):
    monkeypatch.setattr(mtf, "_ALIGNMENT_BATCH_ROWS", 7)
    rng = np.random.default_rng(1357)
    closes = np.datetime64("1970-01-01", "ns") + (np.arange(40) - 20) * np.timedelta64(
        1, "h"
    )
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "ns"))))
    for _ in range(20):
        receipts = closes + rng.integers(0, 10, len(closes)) * np.timedelta64(1, "h")
        raw = _source(closes, receipts)
        active = rng.integers(0, 2, len(base)).astype(bool)
        for valid in (
            np.zeros(len(closes), dtype=bool),
            rng.integers(0, 2, len(closes)).astype(bool),
        ):
            events = (rng.standard_normal(len(closes)), valid, receipts)
            spec = FeatureSpec(
                "body",
                FeatureKind.BODY_RETURN,
                max_staleness_hours=float(rng.integers(1, 10)),
            )
            for times, mask in ((base, active), (base[:0], active[:0])):
                actual = mtf._align_events(spec, raw, events, times, mask)
                expected = _literal_asof(spec, raw, events, times, mask)
                for observed, reference in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(observed, reference)
