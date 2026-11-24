"""Native feature dependency publication must scale without changing causality."""

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    NormalizationMode,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.features.multitimeframe import _native_events, align_native_feature
from trade_rl.data.source import RawMarketSeries


def _raw(n: int) -> RawMarketSeries:
    i = np.arange(n)
    times = np.datetime64("2020-01-01", "ns") + i * np.timedelta64(15, "m")
    # Late old bars deliberately make publication times non-monotone.
    delay = np.where(i % 173 == 5, 300, i % 7)
    close = 100.0 + np.sin(i / 13) + i / 10_000
    return RawMarketSeries(
        timestamps=times,
        available_at=times + delay * np.timedelta64(15, "m"),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        funding_available=i % 32 == 0,
        tradable=np.ones(n, dtype=bool),
    )


@pytest.mark.parametrize(
    "kind",
    [
        FeatureKind.RSI,
        FeatureKind.LOG_RETURN,
        FeatureKind.ICHIMOKU_CLOUD_POSITION,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.MACD_HISTOGRAM,
    ],
)
@pytest.mark.parametrize("normalized", [False, True])
def test_native_publication_matches_literal_dependency_max(kind, normalized):
    raw = _raw(800)
    contract = InstrumentContract(
        "SYNTH",
        listed_at=datetime(2020, 1, 1, 2, tzinfo=UTC),
        delisted_at=datetime(2020, 1, 8, tzinfo=UTC),
    )
    spec = FeatureSpec(
        "test",
        kind,
        lookback=14,
        normalization=NormalizationMode.ROLLING_ZSCORE
        if normalized
        else NormalizationMode.NONE,
        normalization_window=8 if normalized else 1,
        min_periods=3 if normalized else 1,
        max_staleness_hours=400,
    )
    active = (raw.timestamps >= np.datetime64("2020-01-01T02:00")) & (
        raw.timestamps < np.datetime64("2020-01-08")
    )
    expected_values, expected_valid, starts = calculate_feature_events(
        spec,
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(len(active), dtype=bool),
        active=active,
    )
    expected_at = np.full(len(active), np.datetime64("NaT", "ns"))
    for i in np.flatnonzero(expected_valid):
        expected_at[i] = np.max(raw.available_at[starts[i] : i + 1])
    values, valid, at = _native_events(spec, raw, contract)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(at, expected_at)
    # Independent as-of oracle, including delayed earlier events and active mask.
    aligned, available, ages, stale = align_native_feature(
        spec,
        raw,
        contract,
        raw.timestamps,
        active,
        timeframe="15m",
    )
    for i, t in enumerate(raw.timestamps):
        candidates = np.flatnonzero(expected_valid & (expected_at <= t))
        if not active[i] or not len(candidates):
            assert not available[i]
            assert ages[i] == spec.max_staleness_hours
            assert stale[i] == 1
            continue
        j = candidates[-1]
        age = float((t - raw.timestamps[j]) / np.timedelta64(1, "h"))
        assert ages[i] == age
        assert stale[i] == min(age / spec.max_staleness_hours, 1)
        assert available[i] == (age <= spec.max_staleness_hours)
        assert aligned[i] == (expected_values[j] if available[i] else 0)


def test_recursive_native_dependency_does_not_rescan_quadratic_history(monkeypatch):
    raw = _raw(4096)
    original = np.max
    scanned = 0

    def counted_max(values, *args, **kwargs):
        nonlocal scanned
        if np.asarray(values).dtype.kind == "M":
            scanned += np.asarray(values).size
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "max", counted_max)
    _, valid, _ = _native_events(
        FeatureSpec("rsi", FeatureKind.RSI, lookback=14),
        raw,
        InstrumentContract("SYNTH"),
    )
    assert valid.sum() == 4096 - 14
    assert scanned <= 2 * len(raw.timestamps)


def test_range_max_preserves_backward_starts_and_invalid_rows():
    from trade_rl.data.features.multitimeframe import _dependency_available_at

    rng = np.random.default_rng(1357)
    for n in (1, 16, 200):
        raw = np.datetime64("2020-01-01", "ns") + rng.integers(
            0, 100, n
        ) * np.timedelta64(1, "h")
        valid = rng.integers(0, 2, n).astype(bool)
        starts = np.array([rng.integers(0, i + 1) for i in range(n)])
        starts[~valid] = -1
        expected = np.full(n, np.datetime64("NaT", "ns"))
        for i in np.flatnonzero(valid):
            expected[i] = np.max(raw[starts[i] : i + 1])
        np.testing.assert_array_equal(
            _dependency_available_at(raw, valid, starts), expected
        )
    for bad in (-1, 2):
        with pytest.raises(ValueError, match="feature source range is invalid"):
            _dependency_available_at(
                raw[:2], np.array([False, True]), np.array([-1, bad])
            )
