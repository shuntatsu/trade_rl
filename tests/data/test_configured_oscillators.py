"""Explicit native-bar MACD and stochastic smoothing profiles."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_configured_ichimoku import _events
from tests.data.test_long_native_alignment import _source
from tests.data.test_native_spot_builder import _fixture
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.build.config import load_market_build_request
from trade_rl.data.build.native_spot import build_native_fixed_spot_dataset
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features import multitimeframe as mtf
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind

_MACD = (
    FeatureKind.MACD_LINE,
    FeatureKind.MACD_SIGNAL,
    FeatureKind.MACD_HISTOGRAM,
)


def _macd_periods(*args):
    from trade_rl.data.contracts import MACDPeriods

    return MACDPeriods(*args)


def _macd_spec(kind, periods, **kwargs):
    return FeatureSpec("macd", kind, macd_periods=_macd_periods(*periods), **kwargs)


def _stochastic_spec(k, smooth, **kwargs):
    return FeatureSpec(
        "stoch",
        FeatureKind.STOCHASTIC_D,
        lookback=k,
        stochastic_smoothing_period=smooth,
        **kwargs,
    )


def _raw(n=150):
    times = np.datetime64("2024-01-01", "ns") + np.arange(n) * np.timedelta64(1, "h")
    return _source(times, times)


def _literal_macd(raw, kind, periods, present):
    fast_period, slow_period, signal_period = periods
    values = np.zeros(len(present))
    valid = np.zeros(len(present), dtype=bool)
    starts = np.full(len(present), -1)
    fast_alpha, slow_alpha, signal_alpha = (2.0 / (period + 1.0) for period in periods)
    segment = 0
    signal = 0.0
    fast = slow = 0.0
    for i, usable in enumerate(present):
        if not usable:
            segment = i + 1
            continue
        close = float(raw.close[i])
        if i == segment:
            fast = slow = close
        else:
            fast = fast_alpha * close + (1 - fast_alpha) * fast
            slow = slow_alpha * close + (1 - slow_alpha) * slow
        age = i - segment + 1
        line = fast - slow
        if age < slow_period:
            continue
        if age == slow_period:
            signal = line
        else:
            signal = signal_alpha * line + (1 - signal_alpha) * signal
        if kind is FeatureKind.MACD_LINE:
            value = line
        elif age < slow_period + signal_period - 1:
            continue
        else:
            value = signal if kind is FeatureKind.MACD_SIGNAL else line - signal
        values[i], valid[i], starts[i] = value / close, True, segment
    return values, valid, starts


@pytest.mark.parametrize("kind", _MACD)
@pytest.mark.parametrize("periods", [(12, 26, 9), (2, 7, 4), (1, 2, 1)])
def test_macd_matches_literal_segment_recurrence(kind, periods):
    raw = _raw()
    present = np.ones(len(raw.close), dtype=bool)
    present[64] = False
    actual = _events(_macd_spec(kind, periods), raw, present)
    for a, b in zip(actual, _literal_macd(raw, kind, periods, present), strict=True):
        np.testing.assert_array_equal(a, b)
    if periods == (12, 26, 9):
        legacy = FeatureSpec("macd", kind, lookback=99)
        assert "macd_periods" not in legacy.canonical_payload()
        for a, b in zip(actual, _events(legacy, raw, present), strict=True):
            np.testing.assert_array_equal(a, b)


def _literal_stochastic(raw, k, smooth, present):
    values = np.zeros(len(present))
    valid = np.zeros(len(present), dtype=bool)
    starts = np.full(len(present), -1)
    for i in range(k + smooth - 2, len(present)):
        start = i - k - smooth + 2
        if not all(present[start : i + 1]):
            continue
        ks = []
        for t in range(i - smooth + 1, i + 1):
            hi = float(max(raw.high[t - k + 1 : t + 1]))
            lo = float(min(raw.low[t - k + 1 : t + 1]))
            ratio = 0.5 if hi - lo <= 1e-12 else (raw.close[t] - lo) / (hi - lo)
            ks.append(2.0 * ratio - 1.0)
        values[i] = math.fsum(float(value) for value in ks) / len(ks)
        valid[i], starts[i] = True, start
    return values, valid, starts


@pytest.mark.parametrize("k,smooth", [(14, 3), (4, 5), (1, 1), (20, 17)])
def test_stochastic_matches_literal_trailing_windows(k, smooth):
    raw = _raw()
    present = np.ones(len(raw.close), dtype=bool)
    present[66] = False
    actual = _events(_stochastic_spec(k, smooth), raw, present)
    for a, b in zip(actual, _literal_stochastic(raw, k, smooth, present), strict=True):
        np.testing.assert_array_equal(a, b)
    if smooth == 3:
        legacy = FeatureSpec("stoch", FeatureKind.STOCHASTIC_D, lookback=k)
        assert "stochastic_smoothing_period" not in legacy.canonical_payload()
        for a, b in zip(actual, _events(legacy, raw, present), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", [*_MACD, FeatureKind.STOCHASTIC_D])
@pytest.mark.parametrize(
    "clockkind",
    [
        NativeClockKind.FIXED_ELAPSED,
        NativeClockKind.UTC_WEEK,
        NativeClockKind.UTC_MONTH,
    ],
)
def test_explicit_periods_bind_gap_and_publication(kind, clockkind):
    if clockkind is NativeClockKind.FIXED_ELAPSED:
        anchor = np.datetime64("2024-01-01", "ns")
        closes = anchor + np.arange(1, 21) * np.timedelta64(3, "D")
        clock = NativeBarClock(clockkind, 3 * 86400, anchor)
    elif clockkind is NativeClockKind.UTC_WEEK:
        closes = np.datetime64("2024-01-08", "ns") + np.arange(20) * np.timedelta64(
            7, "D"
        )
        clock = NativeBarClock(clockkind)
    else:
        closes = np.arange("2024-02", "2025-10", dtype="datetime64[M]").astype(
            "datetime64[ns]"
        )
        clock = NativeBarClock(clockkind)
    receipts = closes.copy()
    receipts[1] = closes[5] + np.timedelta64(1, "ns")
    raw = _source(closes, receipts)
    present = raw.tradable.copy()
    present[9] = False
    raw = replace(raw, tradable=present)
    spec = (
        _stochastic_spec(3, 4, max_staleness_hours=10000)
        if kind is FeatureKind.STOCHASTIC_D
        else _macd_spec(kind, (2, 4, 2), max_staleness_hours=10000)
    )
    values, valid, available_at = mtf._native_events(
        spec, raw, InstrumentContract("SYNTH")
    )
    expected = _events(spec, raw, present)
    np.testing.assert_array_equal(values, expected[0])
    np.testing.assert_array_equal(valid, expected[1])
    for t in np.flatnonzero(valid):
        assert available_at[t] == max(receipts[expected[2][t] : t + 1])
    assert not valid[9:12].any()
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "ns"))))
    aligned = mtf.align_clocked_native_feature(
        spec,
        raw,
        InstrumentContract("SYNTH"),
        base,
        np.ones(len(base), dtype=bool),
        clock=clock,
    )
    assert not aligned[1][10] and aligned[1][11]
    prices = {
        name: getattr(raw, name).copy() for name in ("open", "high", "low", "close")
    }
    for array in prices.values():
        array[7:] += 100
    future = replace(raw, **prices)
    again = mtf.align_clocked_native_feature(
        spec,
        future,
        InstrumentContract("SYNTH"),
        base,
        np.ones(len(base), dtype=bool),
        clock=clock,
    )
    for a, b in zip(aligned, again, strict=True):
        np.testing.assert_array_equal(a[:14], b[:14])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "9", np.int64(9)])
def test_period_profiles_reject_non_positive_or_non_native_integers(bad):
    with pytest.raises(ValueError, match="period"):
        _macd_periods(bad, 26, 9)
    with pytest.raises(ValueError, match="period"):
        _stochastic_spec(14, bad)


def test_profiles_reject_unused_or_conflicting_settings():
    with pytest.raises(ValueError, match="fast"):
        _macd_periods(26, 12, 9)
    with pytest.raises(ValueError, match="fast"):
        _macd_periods(12, 12, 9)
    with pytest.raises(ValueError, match="lookback"):
        _macd_spec(FeatureKind.MACD_LINE, (12, 26, 9), lookback=12)
    with pytest.raises(ValueError, match="MACD"):
        _macd_spec(FeatureKind.RSI, (12, 26, 9))
    with pytest.raises(ValueError, match="stochastic"):
        FeatureSpec("k", FeatureKind.STOCHASTIC_K, stochastic_smoothing_period=5)
    with pytest.raises(ValueError, match="MACD"):
        FeatureSpec("macd", FeatureKind.MACD_LINE, macd_periods={"fast_period": 2})


def test_unreachable_huge_periods_do_not_allocate_or_convert_to_float():
    raw = _raw(8)
    for spec in (
        _macd_spec(FeatureKind.MACD_LINE, (2, 10**1000, 3)),
        _macd_spec(FeatureKind.MACD_SIGNAL, (2, 3, 10**1000)),
        _stochastic_spec(3, 10**1000),
    ):
        values, valid, starts = _events(spec, raw)
        assert not valid.any() and not values.any() and np.all(starts == -1)


def test_configured_periods_roundtrip_json_and_dataset_identity(tmp_path):
    macd = _macd_spec(FeatureKind.MACD_HISTOGRAM, (2, 3, 2), max_staleness_hours=200)
    stochastic = _stochastic_spec(3, 2, max_staleness_hours=200)
    specs = (macd, stochastic)
    payload = [s.canonical_payload() for s in specs]
    assert payload[0]["macd_periods"] == {
        "schema_version": "macd_periods_v1",
        "fast_period": 2,
        "slow_period": 3,
        "signal_period": 2,
    }
    assert payload[1]["stochastic_smoothing_period"] == 2
    request = tmp_path / "build.json"
    body = {
        "source_root": ".",
        "base_timeframe": "1h",
        "features": payload,
        "instruments": [{"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}],
    }
    request.write_text(json.dumps(body))
    assert load_market_build_request(request).config.features == specs
    args, _ = _fixture(7)
    group = replace(args["feature_groups"][0], specs=specs)
    a = build_native_fixed_spot_dataset(**(args | {"feature_groups": (group,)}))
    for changed in (
        (replace(macd, macd_periods=_macd_periods(2, 4, 2)), stochastic),
        (macd, replace(stochastic, stochastic_smoothing_period=3)),
    ):
        b = build_native_fixed_spot_dataset(
            **(args | {"feature_groups": (replace(group, specs=changed),)})
        )
        assert a.feature_config_digest != b.feature_config_digest
        assert a.dataset_id != b.dataset_id
    assert not a.feature_available[0].any()
    assert a.feature_available[1:].all()
    write_market_dataset_files(tmp_path / "artifact", a)
    restored = load_dataset_files(tmp_path / "artifact")
    assert restored.identity_verified and restored.dataset_id == a.dataset_id
    np.testing.assert_array_equal(restored.features, a.features)
    for field, invalid in (
        ("macd_periods", None),
        ("macd_periods", {"schema_version": "macd_periods_v1", "fast_period": 2}),
    ):
        invalid_features = [dict(payload[0], **{field: invalid}), payload[1]]
        request.write_text(json.dumps(body | {"features": invalid_features}))
        with pytest.raises(ValueError):
            load_market_build_request(request)
    for invalid in (None, True, 2.0, "2"):
        invalid_features = [
            payload[0],
            dict(payload[1], stochastic_smoothing_period=invalid),
        ]
        request.write_text(json.dumps(body | {"features": invalid_features}))
        with pytest.raises(ValueError):
            load_market_build_request(request)


def test_explicit_periods_base_builder_respects_tradability_gap():
    from tests.data.test_market_builder import _stable_identity_series
    from trade_rl.data.build.builder import MarketDatasetBuilder
    from trade_rl.data.contracts import MarketBuildConfig
    from trade_rl.data.source import InMemoryMarketDataSource

    raw = _stable_identity_series(40)
    present = raw.tradable.copy()
    present[20] = False
    raw = replace(raw, tradable=present)
    specs = (
        _macd_spec(FeatureKind.MACD_HISTOGRAM, (2, 4, 2), max_staleness_hours=0.5),
        _stochastic_spec(3, 3, max_staleness_hours=0.5),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(base_timeframe="1h", features=specs)
    ).build(InMemoryMarketDataSource({"SYNTH": raw}), (InstrumentContract("SYNTH"),))
    for index, spec in enumerate(specs):
        expected, valid, _ = _events(spec, raw, present)
        np.testing.assert_array_equal(dataset.feature_available[:, 0, index], valid)
        np.testing.assert_array_equal(
            dataset.features[valid, 0, index],
            expected[valid].astype(dataset.features.dtype),
        )
    assert not dataset.feature_available[20:25].any()


def test_fragmented_macd_allocates_only_segment_scratch(monkeypatch):
    from trade_rl.data.features import core

    raw = _raw(4000)
    present = np.arange(len(raw.close)) % 5 != 4
    original = core._ema
    sizes = []

    def counted(values, *args):
        sizes.append(len(values))
        return original(values, *args)

    monkeypatch.setattr(core, "_ema", counted)
    _events(_macd_spec(FeatureKind.MACD_LINE, (1, 2, 1)), raw, present)
    assert max(sizes) == 4 and sum(sizes) == 2 * int(np.sum(present))


def test_stochastic_computes_each_k_once(monkeypatch):
    from trade_rl.data.features import core

    raw = _raw(200)
    present = np.ones(200, dtype=bool)
    original = core.np.max
    samples = []

    def counted(array, *args, **kwargs):
        samples.append(len(array))
        return original(array, *args, **kwargs)

    monkeypatch.setattr(core.np, "max", counted)
    _events(_stochastic_spec(17, 31), raw, present)
    assert samples == [17] * (200 - 17 + 1)


@pytest.mark.parametrize("kind", _MACD)
def test_explicit_macd_refuses_nonfinite_valid_price(kind):
    raw = _raw(8)
    close = raw.close.copy()
    close[0] = np.inf
    from trade_rl.data.features import calculate_feature_events

    with pytest.raises(ValueError, match="positive finite closes"):
        calculate_feature_events(
            _macd_spec(kind, (1, 2, 2)),
            open_price=raw.open,
            high=raw.high,
            low=raw.low,
            close=close,
            volume=raw.volume,
            funding_rate=raw.funding_rate,
            funding_available=raw.funding_available,
            row_present=raw.tradable,
            active=raw.tradable,
        )
