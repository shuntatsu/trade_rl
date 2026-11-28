"""Cloud region, twist and chikou features: literal oracles, validation, publication."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.data.test_long_native_alignment import _source
from tests.data.test_visible_cloud_scaling import _bars
from trade_rl.data.build.config import load_market_build_request
from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
    FeatureSpec,
    IchimokuPeriods,
    InstrumentContract,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.features import multitimeframe as mtf
from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud

EDGE = FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE
TWIST = FeatureKind.ICHIMOKU_CLOUD_TWIST
CHIKOU = FeatureKind.ICHIMOKU_CHIKOU_DISTANCE
VISIBLE = FeatureAlignment.VISIBLE_DISPLACED_CLOUD


def _events(spec, high, low, close, present=None):
    n = len(close)
    flat = np.zeros(n)
    return calculate_feature_events(
        spec,
        open_price=np.asarray(close, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=np.asarray(close, dtype=float),
        volume=np.ones(n),
        funding_rate=flat,
        funding_available=np.zeros(n, dtype=bool),
        row_present=np.ones(n, dtype=bool) if present is None else present,
        active=np.ones(n, dtype=bool),
    )


def _explicit(kind, periods):
    return FeatureSpec(
        "f",
        kind,
        ichimoku_periods=periods,
        alignment=VISIBLE if periods.displacement > 0 else None,
    )


# Hand-computed with periods tenkan=1, kijun=2, span_b=3. For source row s the
# lines are tenkan=(h[s]+l[s])/2, kijun=(max h[s-1:s+1]+min l[s-1:s+1])/2 and
# span_b over three rows, so span A/B at s = 2..11 are
#   (11.75, 11.0) (12.75, 12.0) (12.5, 12.5) (11.0, 11.5) (9.5, 11.0)
#   (8.0, 9.5) (6.75, 8.0) (9.0, 8.5) (11.75, 10.0) (14.25, 12.0).
HIGH = [12, 13, 14, 15, 16, 14, 12, 10, 9, 13, 16, 18]
LOW = [8, 9, 10, 11, 9, 7, 6, 5, 4, 6, 9, 12]
# Closes chosen to sit below, on the bottom, inside the zero-width cloud, on the
# top edge, inside, above, below and above again for the visible rows 4..11.
VISIBLE_CLOSE = [10, 12, 13, 14, 10, 12, 12.5, 11.5, 10, 12, 5, 16]
VISIBLE_EDGE = [-1 / 10, 0.0, 0.0, 0.0, 0.0, 2.5 / 12, -1.75 / 5, 7 / 16]
VISIBLE_TWIST = [
    0.75 / 10,
    0.75 / 12,
    0.0,
    -0.5 / 11.5,
    -1.5 / 10,
    -1.5 / 12,
    -1.25 / 5,
    0.5 / 16,
]


def test_visible_edge_distance_and_twist_literals():
    for kind, expected in ((EDGE, VISIBLE_EDGE), (TWIST, VISIBLE_TWIST)):
        spec = _explicit(kind, IchimokuPeriods(1, 2, 3, 2))
        values, valid, starts = _events(spec, HIGH, LOW, VISIBLE_CLOSE)
        assert valid.tolist() == [False] * 4 + [True] * 8
        np.testing.assert_array_equal(values[4:], expected)
        assert not values[:4].any()
        assert starts.tolist() == [-1] * 4 + list(range(0, 8))
    # Inside values, including both edges and a zero-width cloud, are +0.0.
    edge = _events(
        _explicit(EDGE, IchimokuPeriods(1, 2, 3, 2)), HIGH, LOW, VISIBLE_CLOSE
    )[0]
    assert all(not np.signbit(edge[t]) and edge[t] == 0.0 for t in (5, 6, 7, 8))


def test_visible_gap_resets_full_window_for_new_kinds():
    present = np.ones(12, dtype=bool)
    present[6] = False
    for kind, expected in ((EDGE, VISIBLE_EDGE), (TWIST, VISIBLE_TWIST)):
        spec = _explicit(kind, IchimokuPeriods(1, 2, 3, 2))
        values, valid, _ = _events(spec, HIGH, LOW, VISIBLE_CLOSE, present)
        # Window [t - 4, t] contains row 6 for t = 6..10.
        assert np.flatnonzero(valid).tolist() == [4, 5, 11]
        np.testing.assert_array_equal(
            values[[4, 5, 11]], [expected[i] for i in (0, 1, 7)]
        )
        assert not values[6:11].any()


def test_unshifted_explicit_edge_and_twist_literals():
    # The unshifted spans at source row s equal the visible spans of row s + 2.
    close = [1, 1, 10, 12, 12.5, 11.5, 10, 12, 5, 16, 11.75, 10]
    edge_tail = [0.0, -0.2]
    twist_tail = [1.75 / 11.75, 2.25 / 10]
    for kind, head, tail in (
        (EDGE, VISIBLE_EDGE, edge_tail),
        (TWIST, VISIBLE_TWIST, twist_tail),
    ):
        spec = _explicit(kind, IchimokuPeriods(1, 2, 3, 0))
        values, valid, starts = _events(spec, HIGH, LOW, close)
        assert valid.tolist() == [False, False] + [True] * 10
        np.testing.assert_array_equal(values[2:], [*head, *tail])
        assert starts.tolist() == [-1, -1, *range(0, 10)]


def _ramp(n, up):
    i = np.arange(n, dtype=float)
    centre = 100.0 + i if up else 200.0 - i
    return centre + 1.0, centre - 1.0, centre


@pytest.mark.parametrize("up", [True, False])
def test_legacy_implicit_edge_and_twist_on_ramps(up):
    high, low, close = _ramp(100, up)
    t = np.arange(51, 100)
    # Hand-derived: rising ramps have span A = c - 8.25, span B = c - 25.5 and
    # close above both; falling ramps have A = c + 8.25, B = c + 25.5.
    sign = 1.0 if up else -1.0
    edge = sign * 8.25 / close[t]
    twist = sign * 17.25 / close[t]
    for kind, expected in ((EDGE, edge), (TWIST, twist)):
        values, valid, starts = _events(
            FeatureSpec("f", kind, lookback=52), high, low, close
        )
        assert np.flatnonzero(valid).tolist() == t.tolist()
        np.testing.assert_array_equal(values[t], expected)
        assert starts[51] == 0 and starts[99] == 48


def test_legacy_implicit_edge_is_exactly_zero_inside_and_on_edges():
    high, low, close = _ramp(100, True)
    # At row 60 the rising cloud spans [100+60-25.5, 100+60-8.25].
    close = close.copy()
    close[60], close[61], close[62] = 151.75, 100.0 + 61 - 25.5, 140.0
    values, valid, _ = _events(FeatureSpec("f", EDGE, lookback=52), high, low, close)
    for row in (60, 61, 62):
        assert valid[row] and values[row] == 0.0 and not np.signbit(values[row])
    assert values[63] == 8.25 / close[63]
    twist, _, _ = _events(FeatureSpec("f", TWIST, lookback=52), high, low, close)
    assert twist[60] == 17.25 / 151.75


@pytest.mark.parametrize("kind", [EDGE, TWIST])
def test_explicit_zero_displacement_equals_legacy(kind):
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    high = close * (1 + np.abs(rng.normal(0, 0.003, 200)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, 200)))
    present = np.ones(200, dtype=bool)
    present[90] = False
    legacy = _events(FeatureSpec("f", kind, lookback=52), high, low, close, present)
    explicit = _events(_explicit(kind, IchimokuPeriods()), high, low, close, present)
    for observed, expected in zip(explicit, legacy, strict=True):
        np.testing.assert_array_equal(observed, expected)
    assert not explicit[1][90:142].any() and explicit[1][142]


# Chikou: lag = displacement = 3 with explicit periods.
CHIKOU_CLOSE = [10, 11, 13, 12, 9, 8, 10, 16, 15, 12, 20, 25]


def test_explicit_chikou_literals_gap_and_warmup():
    spec = _explicit(CHIKOU, IchimokuPeriods(displacement=3))
    n = len(CHIKOU_CLOSE)
    high = low = CHIKOU_CLOSE
    values, valid, starts = _events(spec, high, low, CHIKOU_CLOSE)
    assert valid.tolist() == [False] * 3 + [True] * 9
    np.testing.assert_array_equal(
        values[3:],
        [
            (12 - 10) / 12,
            (9 - 11) / 9,
            (8 - 13) / 8,
            (10 - 12) / 10,
            (16 - 9) / 16,
            (15 - 8) / 15,
            (12 - 10) / 12,
            (20 - 16) / 20,
            (25 - 15) / 25,
        ],
    )
    assert starts.tolist() == [-1] * 3 + list(range(0, n - 3))
    present = np.ones(n, dtype=bool)
    present[5] = False
    values, valid, starts = _events(spec, high, low, CHIKOU_CLOSE, present)
    # Window [t - 3, t] contains the gap for t = 5..8: no bridging, no warm-up
    # shortcut for rows whose lag row is usable but the interior is not.
    assert np.flatnonzero(valid).tolist() == [3, 4, 9, 10, 11]
    np.testing.assert_array_equal(
        values[[3, 4, 9, 10, 11]],
        [(12 - 10) / 12, (9 - 11) / 9, (12 - 10) / 12, (20 - 16) / 20, (25 - 15) / 25],
    )
    assert starts[[3, 4, 9, 10, 11]].tolist() == [0, 1, 6, 7, 8]
    assert not values[~valid].any()


def test_chikou_does_not_read_after_t_and_ignores_cloud_periods():
    base = _explicit(CHIKOU, IchimokuPeriods(displacement=3))
    odd = _explicit(CHIKOU, IchimokuPeriods(40, 41, 42, 3))
    reference = _events(base, CHIKOU_CLOSE, CHIKOU_CLOSE, CHIKOU_CLOSE)
    other = _events(odd, CHIKOU_CLOSE, CHIKOU_CLOSE, CHIKOU_CLOSE)
    for a, b in zip(reference, other, strict=True):
        np.testing.assert_array_equal(a, b)
    mutated = list(CHIKOU_CLOSE)
    mutated[8:] = [500, 600, 700, 800]
    changed = _events(base, mutated, mutated, mutated)
    np.testing.assert_array_equal(changed[0][:8], reference[0][:8])
    assert changed[0][8] != reference[0][8]


def test_legacy_implicit_chikou_lag_is_26_on_ramps_and_gaps():
    high, low, close = _ramp(100, True)
    spec = FeatureSpec("f", CHIKOU, lookback=26)
    values, valid, starts = _events(spec, high, low, close)
    t = np.arange(26, 100)
    assert np.flatnonzero(valid).tolist() == t.tolist()
    np.testing.assert_array_equal(values[t], 26.0 / close[t])
    assert starts[26] == 0 and starts[99] == 73
    high, low, close = _ramp(100, False)
    values, valid, _ = _events(spec, high, low, close)
    np.testing.assert_array_equal(values[t], -26.0 / close[t])
    present = np.ones(100, dtype=bool)
    present[40] = False
    _, valid, _ = _events(spec, high, low, close, present)
    assert np.flatnonzero(~valid[26:]).tolist() == [i - 26 for i in range(40, 67)]


def test_tradingview_convention_is_displacement_25():
    periods = IchimokuPeriods(displacement=25)
    assert (periods.tenkan_period, periods.kijun_period, periods.span_b_period) == (
        9,
        26,
        52,
    )
    raw = _bars(160, 60)
    high, low, close = raw.high, raw.low, raw.close

    def mid(s, k):
        return (max(high[s - k + 1 : s + 1]) + min(low[s - k + 1 : s + 1])) * 0.5

    cloud = visible_ichimoku_cloud(raw, displacement=25)
    assert np.flatnonzero(cloud.ready)[0] == 51 + 25
    twist, valid, starts = _events(_explicit(TWIST, periods), high, low, close)
    edge, _, _ = _events(_explicit(EDGE, periods), high, low, close)
    chikou, chikou_valid, chikou_starts = _events(
        _explicit(CHIKOU, periods), high, low, close
    )
    assert np.flatnonzero(valid)[0] == 76 and starts[76] == 0
    for t in range(76, 160):
        s = t - 25  # not t - 24 and not t - 26
        a, b = (mid(s, 9) + mid(s, 26)) * 0.5, mid(s, 52)
        assert cloud.span_a[t] == a and cloud.span_b[t] == b
        assert twist[t] == (a - b) / close[t]
        top, bottom = max(a, b), min(a, b)
        expected = (
            (close[t] - top) / close[t]
            if close[t] > top
            else (close[t] - bottom) / close[t]
            if close[t] < bottom
            else 0.0
        )
        assert edge[t] == expected
    assert np.flatnonzero(chikou_valid)[0] == 25 and chikou_starts[25] == 0
    for t in range(25, 160):
        assert chikou[t] == (close[t] - close[t - 25]) / close[t]
    shifted = visible_ichimoku_cloud(raw, displacement=26)
    assert not np.array_equal(
        cloud.span_a[77:], shifted.span_a[77:]
    )  # one bar of displacement matters


@pytest.mark.parametrize("kind", [EDGE, TWIST, CHIKOU])
def test_publication_binds_the_full_dependency_window(kind):
    times = np.datetime64("2024-01-01", "ns") + np.arange(16) * np.timedelta64(1, "h")
    receipts = times.copy()
    receipts[1] = times[10] + np.timedelta64(1, "ns")
    raw = _source(times, receipts)
    if kind is CHIKOU:
        periods, span = IchimokuPeriods(2, 3, 4, 2), 2  # window [t - 2, t]
    else:
        periods, span = IchimokuPeriods(2, 3, 4, 2), 5  # [t - 2 - 4 + 1, t]
    spec = FeatureSpec(
        "f",
        kind,
        ichimoku_periods=periods,
        alignment=VISIBLE,
        max_staleness_hours=100.0,
    )
    values, valid, at = mtf._native_events(spec, raw, InstrumentContract("SYNTH"))
    assert valid.any()
    for t in np.flatnonzero(valid):
        assert at[t] == max(receipts[t - span : t + 1])
    delayed = [t for t in np.flatnonzero(valid) if at[t] == receipts[1]]
    # The delayed row is bound exactly while it lies in the dependency window.
    assert delayed == [t for t in np.flatnonzero(valid) if t - span <= 1]
    assert delayed and not valid[:span].any()


@pytest.mark.parametrize("kind", [EDGE, TWIST, CHIKOU])
def test_nonfinite_or_nonpositive_inputs_are_refused(kind):
    prices = np.array([1e-308, 1e308, 1e-308, 1e-308])
    spec = _explicit(kind, IchimokuPeriods(1, 1, 2, 1))
    with pytest.raises(ValueError, match="Ichimoku feature must be finite"):
        _events(spec, prices, prices, prices)
    bad = np.array([5.0, 0.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="positive finite close"):
        _events(_explicit(kind, IchimokuPeriods(1, 1, 1, 1)), bad + 1, bad, bad)


def test_huge_displacement_skips_without_allocating():
    spec = _explicit(CHIKOU, IchimokuPeriods(displacement=10**80))
    values, valid, starts = _events(spec, [1.0] * 8, [1.0] * 8, [1.0] * 8)
    assert not valid.any() and not values.any() and np.all(starts == -1)


def test_validation_refusals_for_new_kinds():
    # Chikou needs a positive explicit lag with the visible alignment.
    with pytest.raises(ValueError, match="positive displacement"):
        FeatureSpec("f", CHIKOU, ichimoku_periods=IchimokuPeriods())
    with pytest.raises(ValueError, match="positive displacement"):
        FeatureSpec("f", CHIKOU, ichimoku_periods=IchimokuPeriods(), alignment=VISIBLE)
    with pytest.raises(ValueError, match="displacement"):
        FeatureSpec("f", CHIKOU, ichimoku_periods=IchimokuPeriods(displacement=3))
    with pytest.raises(ValueError, match="displacement"):
        FeatureSpec(
            "f",
            CHIKOU,
            ichimoku_periods=IchimokuPeriods(displacement=3),
            alignment=FeatureAlignment.UNSHIFTED_DECISION_TIME,
        )
    with pytest.raises(ValueError, match="explicit Ichimoku periods"):
        FeatureSpec("f", CHIKOU, lookback=26, alignment=VISIBLE)
    # The implicit chikou lag is fixed at 26 and must be declared, not guessed.
    for lookback in (1, 25, 27, 52):
        with pytest.raises(ValueError, match="26-bar lag"):
            FeatureSpec("f", CHIKOU, lookback=lookback)
    for alignment in (None, FeatureAlignment.UNSHIFTED_DECISION_TIME):
        assert FeatureSpec("f", CHIKOU, lookback=26, alignment=alignment)
    # Cloud kinds follow the existing displacement/alignment pairing.
    for kind in (EDGE, TWIST):
        with pytest.raises(ValueError, match="displacement"):
            FeatureSpec("f", kind, ichimoku_periods=IchimokuPeriods(displacement=2))
        with pytest.raises(ValueError, match="displacement"):
            FeatureSpec(
                "f", kind, ichimoku_periods=IchimokuPeriods(), alignment=VISIBLE
            )
        with pytest.raises(ValueError, match="lookback"):
            FeatureSpec("f", kind, ichimoku_periods=IchimokuPeriods(), lookback=52)
        with pytest.raises(ValueError, match="period"):
            FeatureSpec("f", kind, alignment=VISIBLE)
    # Line-distance kinds still cannot be displaced.
    for kind in (
        FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
        FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
    ):
        with pytest.raises(ValueError, match="cloud and chikou"):
            _explicit(kind, IchimokuPeriods(displacement=2))
    with pytest.raises(ValueError, match="Ichimoku"):
        _explicit(FeatureKind.LOG_RETURN, IchimokuPeriods(displacement=2))


def test_existing_kind_values_are_unchanged_and_new_values_are_stable():
    assert [kind.value for kind in FeatureKind][:21] == [
        "log_return",
        "realized_volatility",
        "volume_zscore",
        "funding_bps",
        "rsi",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "bollinger_position",
        "bollinger_bandwidth",
        "atr_pct",
        "adx",
        "stochastic_k",
        "stochastic_d",
        "cci",
        "williams_r",
        "obv_slope",
        "ichimoku_tenkan_distance",
        "ichimoku_kijun_distance",
        "ichimoku_cloud_position",
        "ichimoku_cloud_thickness",
    ]
    assert (EDGE.value, TWIST.value, CHIKOU.value) == (
        "ichimoku_cloud_edge_distance",
        "ichimoku_cloud_twist",
        "ichimoku_chikou_distance",
    )


def test_config_json_roundtrip_and_distinct_identity(tmp_path):
    specs = (
        FeatureSpec(
            "edge",
            EDGE,
            ichimoku_periods=IchimokuPeriods(7, 3, 5, 2),
            alignment=VISIBLE,
            max_staleness_hours=72.0,
        ),
        FeatureSpec("twist", TWIST, ichimoku_periods=IchimokuPeriods(7, 3, 5, 0)),
        FeatureSpec(
            "chikou",
            CHIKOU,
            ichimoku_periods=IchimokuPeriods(displacement=25),
            alignment=VISIBLE,
        ),
        FeatureSpec("legacy_chikou", CHIKOU, lookback=26),
        FeatureSpec("legacy_edge", EDGE, lookback=52),
        FeatureSpec("legacy_twist", TWIST, lookback=52),
    )
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "features": [spec.canonical_payload() for spec in specs],
                "instruments": [
                    {"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_market_build_request(path).config.features == specs
    payloads = [json.dumps(spec.canonical_payload(), sort_keys=True) for spec in specs]
    assert len(set(payloads)) == len(specs)
    assert specs[2].canonical_payload()["kind"] == "ichimoku_chikou_distance"
    # Chikou is declared by its lag; a different displacement is a different feature.
    other = FeatureSpec(
        "chikou",
        CHIKOU,
        ichimoku_periods=IchimokuPeriods(displacement=26),
        alignment=VISIBLE,
    )
    assert other.canonical_payload() != specs[2].canonical_payload()
    bad = specs[2].canonical_payload() | {"kind": "ichimoku_chikou_span"}
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "features": [bad],
                "instruments": [
                    {"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_market_build_request(path)


def test_native_dataset_roundtrip_with_new_kinds(tmp_path):
    from dataclasses import replace

    from tests.data.test_native_spot_builder import _fixture
    from trade_rl.data.artifacts.codec import (
        load_dataset_files,
        write_market_dataset_files,
    )
    from trade_rl.data.build.native_spot import build_native_fixed_spot_dataset

    args, _ = _fixture(7)
    group = args["feature_groups"][0]
    specs = tuple(
        FeatureSpec(
            kind.value,
            kind,
            ichimoku_periods=IchimokuPeriods(2, 3, 4, 1),
            alignment=VISIBLE,
            max_staleness_hours=200,
        )
        for kind in (EDGE, TWIST, CHIKOU)
    )
    dataset = build_native_fixed_spot_dataset(
        **(args | {"feature_groups": (replace(group, specs=specs),)})
    )
    thickness = replace(specs[0], kind=FeatureKind.ICHIMOKU_CLOUD_THICKNESS)
    other = build_native_fixed_spot_dataset(
        **(args | {"feature_groups": (replace(group, specs=(thickness,)),)})
    )
    assert dataset.feature_config_digest != other.feature_config_digest
    assert dataset.feature_available[:, 0, :].any()
    write_market_dataset_files(tmp_path / "artifact", dataset)
    restored = load_dataset_files(tmp_path / "artifact")
    assert restored.dataset_id == dataset.dataset_id and restored.identity_verified
    np.testing.assert_array_equal(restored.features, dataset.features)
