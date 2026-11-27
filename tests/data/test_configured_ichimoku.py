"""Opt-in Ichimoku periods, placement and causal native publication."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_long_native_alignment import _source
from tests.data.test_native_spot_builder import _fixture
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.build.config import load_market_build_request
from trade_rl.data.build.native_spot import build_native_fixed_spot_dataset
from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.features import multitimeframe as mtf
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind


def _periods(*args, **kwargs):
    from trade_rl.data.contracts import IchimokuPeriods

    return IchimokuPeriods(*args, **kwargs)


def _spec(kind, periods, *, displaced=False, **kwargs):
    return FeatureSpec(
        "cloud",
        kind,
        ichimoku_periods=periods,
        alignment=FeatureAlignment("visible_displaced_cloud") if displaced else None,
        **kwargs,
    )


def _events(spec, raw, present=None):
    return calculate_feature_events(
        spec,
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=raw.tradable if present is None else present,
        active=np.ones(len(raw.close), dtype=bool),
    )


def _literal(raw, kind, params, present):
    v = np.zeros(len(raw.close))
    valid = np.zeros(len(v), dtype=bool)
    starts = np.full(len(v), -1)
    tenkan, kijun, span, displacement = params
    if kind is FeatureKind.ICHIMOKU_TENKAN_DISTANCE:
        required = tenkan
    elif kind is FeatureKind.ICHIMOKU_KIJUN_DISTANCE:
        required = kijun
    else:
        required = max(tenkan, kijun, span)
    for t in range(required - 1 + displacement, len(v)):
        s = t - displacement
        start = s - required + 1
        if not all(present[start : t + 1]):
            continue

        def mid(k):
            return (
                float(max(raw.high[s - k + 1 : s + 1]))
                + float(min(raw.low[s - k + 1 : s + 1]))
            ) * 0.5

        c = float(raw.close[t])
        if kind is FeatureKind.ICHIMOKU_TENKAN_DISTANCE:
            v[t] = (c - mid(tenkan)) / c
        elif kind is FeatureKind.ICHIMOKU_KIJUN_DISTANCE:
            v[t] = (c - mid(kijun)) / c
        else:
            a, b = (mid(tenkan) + mid(kijun)) * 0.5, mid(span)
            v[t] = (
                (c - (a + b) * 0.5) / c
                if kind is FeatureKind.ICHIMOKU_CLOUD_POSITION
                else abs(a - b) / c
            )
        starts[t], valid[t] = start, True
    return v, valid, starts


_KINDS = (
    FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
    FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
    FeatureKind.ICHIMOKU_CLOUD_POSITION,
    FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("params", [(9, 26, 52, 0), (7, 2, 3, 0), (1, 1, 1, 0)])
def test_configured_unshifted_matches_literal_and_legacy_default(kind, params):
    times = np.datetime64("2024-01-01", "ns") + np.arange(130) * np.timedelta64(1, "h")
    raw = _source(times, times)
    present = np.ones(len(times), dtype=bool)
    present[66] = False
    spec = _spec(kind, _periods(*params))
    actual = _events(spec, raw, present)
    for observed, expected in zip(
        actual, _literal(raw, kind, params, present), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    if params == (9, 26, 52, 0):
        legacy = FeatureSpec("cloud", kind)
        assert "ichimoku_periods" not in legacy.canonical_payload()
        for observed, expected in zip(
            actual, _events(legacy, raw, present), strict=True
        ):
            np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("kind", _KINDS[2:])
@pytest.mark.parametrize("params", [(9, 26, 52, 26), (7, 2, 3, 2), (1, 1, 1, 1)])
def test_displaced_cloud_requires_full_window_through_visible_row(kind, params):
    times = np.datetime64("2024-01-01", "ns") + np.arange(175) * np.timedelta64(1, "h")
    raw = _source(times, times)
    present = np.ones(len(times), dtype=bool)
    present[84] = False
    spec = _spec(kind, _periods(*params), displaced=True)
    actual = _events(spec, raw, present)
    for observed, expected in zip(
        actual, _literal(raw, kind, params, present), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert np.flatnonzero(actual[1])[0] == max(params[:3]) - 1 + params[3]
    assert not actual[1][84 : 84 + max(params[:3]) + params[3]].any()


@pytest.mark.parametrize(
    "clockkind", [NativeClockKind.UTC_WEEK, NativeClockKind.UTC_MONTH]
)
def test_displaced_publication_dependency_and_gap(clockkind):
    if clockkind is NativeClockKind.UTC_WEEK:
        closes = np.datetime64("2024-01-08", "ns") + np.arange(14) * np.timedelta64(
            7, "D"
        )
    else:
        closes = np.arange("2024-02", "2025-04", dtype="datetime64[M]").astype(
            "datetime64[ns]"
        )
    receipts = closes.copy()
    receipts[1] = closes[5] + np.timedelta64(1, "ns")
    raw = _source(closes, receipts)
    tradable = raw.tradable.copy()
    tradable[8] = False
    raw = replace(raw, tradable=tradable)
    clock = NativeBarClock(clockkind)
    spec = _spec(
        _KINDS[2], _periods(2, 3, 4, 2), displaced=True, max_staleness_hours=10000.0
    )
    values, valid, at = mtf._native_events(spec, raw, InstrumentContract("SYNTH"))
    literal = _literal(raw, _KINDS[2], (2, 3, 4, 2), tradable)
    np.testing.assert_array_equal(values, literal[0])
    np.testing.assert_array_equal(valid, literal[1])
    for t in np.flatnonzero(valid):
        assert at[t] == max(receipts[literal[2][t] : t + 1])
    assert not valid[:5].any() and valid[5]
    base = np.sort(np.concatenate((closes, closes + np.timedelta64(1, "ns"))))
    v, ok, _, _ = mtf.align_clocked_native_feature(
        spec,
        raw,
        InstrumentContract("SYNTH"),
        base,
        np.ones(len(base), dtype=bool),
        clock=clock,
    )
    assert not ok[10] and ok[11] and v[11] == values[5]
    assert not valid[8:14].any()  # no early post-gap warm-up
    # A mutation after the visible row cannot change earlier as-of outputs.
    prices = {
        name: getattr(raw, name).copy() for name in ("open", "high", "low", "close")
    }
    for array in prices.values():
        array[7:] += 100
    future = replace(raw, **prices)
    result = mtf.align_clocked_native_feature(
        spec,
        future,
        InstrumentContract("SYNTH"),
        base,
        np.ones(len(base), dtype=bool),
        clock=clock,
    )
    np.testing.assert_array_equal(result[0][:14], v[:14])
    np.testing.assert_array_equal(result[1][:14], ok[:14])


def test_config_and_native_dataset_roundtrip(tmp_path):
    params = _periods(2, 3, 4, 1)
    spec = _spec(_KINDS[2], params, displaced=True, max_staleness_hours=200)
    payload = spec.canonical_payload()
    assert payload["ichimoku_periods"] == {
        "schema_version": "ichimoku_periods_v1",
        "tenkan_period": 2,
        "kijun_period": 3,
        "span_b_period": 4,
        "displacement": 1,
    }
    request = tmp_path / "build.json"
    request.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "features": [payload],
                "instruments": [
                    {"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}
                ],
            }
        )
    )
    assert load_market_build_request(request).config.features == (spec,)
    args, _ = _fixture(7)
    group = args["feature_groups"][0]
    first = replace(group, specs=(spec,))
    a = build_native_fixed_spot_dataset(**(args | {"feature_groups": (first,)}))
    changed = replace(spec, ichimoku_periods=_periods(2, 3, 5, 1))
    b = build_native_fixed_spot_dataset(
        **(args | {"feature_groups": (replace(first, specs=(changed,)),)})
    )
    assert (
        a.feature_config_digest != b.feature_config_digest
        and a.dataset_id != b.dataset_id
    )
    assert (
        not a.feature_available[:2, 0, 0].any() and a.feature_available[2:, 0, 0].all()
    )
    write_market_dataset_files(tmp_path / "artifact", a)
    restored = load_dataset_files(tmp_path / "artifact")
    assert restored.dataset_id == a.dataset_id and restored.identity_verified
    np.testing.assert_array_equal(restored.features, a.features)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "9", np.int64(9)])
def test_periods_require_positive_native_ints(bad):
    with pytest.raises(ValueError, match="period"):
        _periods(bad, 26, 52, 0)


def test_invalid_placement_and_unused_parameters_refused():
    with pytest.raises(ValueError, match="Ichimoku"):
        _spec(FeatureKind.LOG_RETURN, _periods())
    with pytest.raises(ValueError, match="displacement"):
        _periods(9, 26, 52, -1)
    with pytest.raises(ValueError, match="displacement"):
        _spec(_KINDS[2], _periods(9, 26, 52, 26))
    with pytest.raises(ValueError, match="displacement"):
        _spec(_KINDS[2], _periods(), displaced=True)
    with pytest.raises(ValueError, match="cloud"):
        _spec(_KINDS[0], _periods(9, 26, 52, 2), displaced=True)
    with pytest.raises(ValueError, match="period"):
        FeatureSpec(
            "cloud", _KINDS[2], alignment=FeatureAlignment("visible_displaced_cloud")
        )
    with pytest.raises(ValueError, match="period"):
        FeatureSpec("cloud", _KINDS[2], ichimoku_periods={"tenkan_period": 9})


def test_large_period_skips_without_window_allocation(monkeypatch):
    times = np.datetime64("2024-01-01", "ns") + np.arange(8) * np.timedelta64(1, "h")
    raw = _source(times, times)
    from trade_rl.data.features.indicators import _ichimoku_windows

    def forbidden(*args):
        pytest.fail("unreachable warm-up allocated rolling windows")

    monkeypatch.setattr(_ichimoku_windows, "_rolling_midpoints", forbidden)
    spec = _spec(_KINDS[2], _periods(9, 26, 10**80, 0))
    values, valid, starts = _events(spec, raw)
    assert not valid.any() and not values.any() and np.all(starts == -1)


def test_explicit_periods_do_not_silently_ignore_legacy_lookback():
    with pytest.raises(ValueError, match="lookback"):
        _spec(_KINDS[2], _periods(), lookback=52)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "schema", "bool", "null", "float"]
)
def test_json_profile_is_versioned_and_strict(tmp_path, mutation):
    spec = _spec(_KINDS[2], _periods(2, 3, 4, 1), displaced=True)
    payload = spec.canonical_payload()
    profile = payload["ichimoku_periods"]
    if mutation == "missing":
        profile.pop("span_b_period")
    elif mutation == "extra":
        profile["ignored"] = 1
    elif mutation == "schema":
        profile["schema_version"] = "ichimoku_periods_v2"
    elif mutation == "bool":
        profile["displacement"] = True
    elif mutation == "null":
        payload["ichimoku_periods"] = None
    else:
        profile["tenkan_period"] = 2.0
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "features": [payload],
                "instruments": [
                    {"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="ichimoku_periods"):
        load_market_build_request(path)


def test_base_builder_respects_explicit_untradable_gap():
    from trade_rl.data.build.builder import MarketDatasetBuilder
    from trade_rl.data.contracts import MarketBuildConfig
    from trade_rl.data.source import InMemoryMarketDataSource

    times = np.datetime64("2024-01-01", "ns") + np.arange(28) * np.timedelta64(1, "h")
    raw = _source(times, times)
    present = raw.tradable.copy()
    present[10] = False
    raw = replace(raw, tradable=present)
    spec = _spec(
        _KINDS[2], _periods(2, 3, 4, 2), displaced=True, max_staleness_hours=0.5
    )
    dataset = MarketDatasetBuilder(MarketBuildConfig("1h", (spec,))).build(
        InMemoryMarketDataSource({"SYNTH": raw}), (InstrumentContract("SYNTH"),)
    )
    v, ok, _ = _literal(raw, _KINDS[2], (2, 3, 4, 2), present)
    np.testing.assert_array_equal(dataset.feature_available[:, 0, 0], ok)
    np.testing.assert_array_equal(dataset.features[:, 0, 0], v.astype(np.float32))


def test_legacy_multitimeframe_builder_delayed_profile():
    from tests.data.test_multitimeframe_builder import MemoryMultiTimeframeSource
    from trade_rl.data.build.builder import MarketDatasetBuilder
    from trade_rl.data.contracts import MarketBuildConfig

    times = np.datetime64("2024-01-01", "ns") + np.arange(49) * np.timedelta64(1, "h")
    native_times = times[::4]
    receipts = native_times.copy()
    receipts[1] = native_times[5] + np.timedelta64(1, "h")
    raw = _source(native_times, receipts)
    base = _source(times, times)
    spec = _spec(
        _KINDS[2],
        _periods(2, 3, 4, 2),
        displaced=True,
        timeframe="4h",
        max_staleness_hours=8.0,
    )
    dataset = MarketDatasetBuilder(MarketBuildConfig("1h", (spec,))).build(
        MemoryMultiTimeframeSource({("SYNTH", "1h"): base, ("SYNTH", "4h"): raw}),
        (InstrumentContract("SYNTH"),),
    )
    v, ok, starts = _literal(raw, _KINDS[2], (2, 3, 4, 2), raw.tradable)
    expected = np.zeros(len(times), dtype=np.float32)
    valid = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        candidates = [
            k for k in np.flatnonzero(ok) if max(receipts[starts[k] : k + 1]) <= t
        ]
        if candidates and t - native_times[candidates[-1]] <= np.timedelta64(8, "h"):
            expected[i] = v[candidates[-1]]
            valid[i] = True
    np.testing.assert_array_equal(dataset.feature_available[:, 0, 0], valid)
    np.testing.assert_array_equal(dataset.features[:, 0, 0], expected)
    assert not valid[20] and valid[21]


def test_explicit_month_overlay_binds_displacement_and_preserves_execution():
    from tests.data.test_clocked_feature_overlay import _append
    from tests.data.test_clocked_feature_overlay import _fixture as overlay_fixture

    base, clock, sources, raw = overlay_fixture("month")
    spec = _spec(
        _KINDS[2], _periods(1, 2, 2, 1), displaced=True, max_staleness_hours=744.0
    )
    result = _append(base, clock, sources, specs=(spec,))
    other = _append(
        base,
        clock,
        sources,
        specs=(replace(spec, ichimoku_periods=_periods(1, 2, 2, 2)),),
    )
    assert result.identity_verified and other.identity_verified
    assert result.dataset_id != other.dataset_id
    assert result.periods_per_year == base.periods_per_year == 8760
    for j, series in enumerate(raw):
        row = np.flatnonzero(base.timestamps == series.timestamps[2])[0]
        assert not result.feature_available[:row, j, 1].any()
        assert result.feature_available[row, j, 1]
        expected = _literal(series, _KINDS[2], (1, 2, 2, 1), series.tradable)[0][2]
        assert result.features[row, j, 1] == np.float32(expected)
    for name, values in base.identity_arrays().items():
        np.testing.assert_array_equal(
            result.identity_arrays()[name][:, :, :1]
            if name.startswith("feature")
            else result.identity_arrays()[name],
            values,
        )


def test_configured_window_passes_scale_with_rows_not_period_length(monkeypatch):
    from trade_rl.data.features.indicators import _ichimoku_windows

    times = np.datetime64("2024-01-01", "ns") + np.arange(10000) * np.timedelta64(
        1, "h"
    )
    raw = _source(times, times)
    original = _ichimoku_windows._rolling_midpoints
    calls = []

    def counted(high, low, period):
        calls.append((len(high), period))
        return original(high, low, period)

    monkeypatch.setattr(_ichimoku_windows, "_rolling_midpoints", counted)
    spec = _spec(_KINDS[2], _periods(3000, 3000, 6000, 10), displaced=True)
    v, valid, starts = _events(spec, raw)
    assert sorted(calls) == [(9990, 3000), (9990, 6000)]
    assert np.flatnonzero(valid)[0] == 6009 and starts[6009] == 0
    assert np.isfinite(v[valid]).all()
