"""Contract tests for causal multi-indicator divergence and visible Ichimoku cloud."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.features.divergence import DivergenceConfig, detect_divergences
from trade_rl.data.features.indicators.divergence_bank import build_divergence_bank
from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud
from trade_rl.data.source import RawMarketSeries


def _bars(n: int = 24 * 95, *, delayed: bool = False) -> RawMarketSeries:
    t = np.datetime64("2026-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    phase = np.arange(n, dtype=float)
    c = 100 + 0.025 * phase + 2.3 * np.sin(phase / 8) + 2 * np.sin(phase / 101)
    o = c - 0.2
    h = np.maximum(o, c) + 0.5
    low_price = np.minimum(o, c) - 0.5
    avail = t + np.timedelta64(2, "h") if delayed else t
    return RawMarketSeries(
        timestamps=t,
        available_at=avail,
        open=o,
        high=h,
        low=low_price,
        close=c,
        volume=np.full(n, 10_000.0),
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=bool),
    )


def test_cloud_visible_only_after_displacement_and_no_future_dependency() -> None:
    bars = _bars(24 * 90)
    cloud = visible_ichimoku_cloud(bars)
    assert not cloud.ready[:77].any()
    assert cloud.ready[77]
    source = 77 - 26
    tenkan = (
        np.max(bars.high[source - 8 : source + 1])
        + np.min(bars.low[source - 8 : source + 1])
    ) / 2
    kijun = (
        np.max(bars.high[source - 25 : source + 1])
        + np.min(bars.low[source - 25 : source + 1])
    ) / 2
    span_b = (np.max(bars.high[: source + 1]) + np.min(bars.low[: source + 1])) / 2
    assert cloud.top[77] == pytest.approx(max((tenkan + kijun) / 2, span_b))
    short = _bars(95)
    before = visible_ichimoku_cloud(short)
    assert before.ready[77]
    np.testing.assert_array_equal(before.top, cloud.top[:95], strict=True)


def test_canonical_indicator_bank_has_real_warmup_masks() -> None:
    bank = build_divergence_bank(_bars(125), names=("RSI", "MACD_HIST", "MFI"))
    assert set(bank) == {"RSI", "MACD_HIST", "MFI"}
    assert np.isnan(bank["RSI"][:14]).all()
    assert np.isfinite(bank["RSI"][-1])
    assert np.isfinite(bank["MACD_HIST"][-1])


def test_regular_bull_event_is_observed_not_backdated_and_is_deduplicated() -> None:
    close = np.asarray(
        [
            11,
            10.5,
            10,
            9.0,
            8.0,
            9.0,
            10.0,
            11,
            12,
            11,
            10,
            9.5,
            8.5,
            7.0,
            8,
            9,
            10,
            8.5,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        dtype=float,
    )
    osc = np.asarray(
        [0, -1, -2, -3, -6, -5, -4, -2, 0, -1, -2, -1, -2, -3, -1, 0, 1, 2, 3, 4, 5, 6],
        dtype=float,
    )
    out = detect_divergences(
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        indicators={"RSI": osc},
        config=DivergenceConfig(
            pivot_period=2, max_bars=30, min_indicators=1, validate_lines=False
        ),
    )
    events = [e for e in out.events if e.kind == "bull_regular"]
    assert events
    assert all(e.observed_at > e.pivot_at for e in events)
    assert len({(e.indicator, e.kind, e.pivot_at) for e in events}) == len(events)
    assert np.array_equal(
        out.bull_regular.astype(bool),
        np.isin(np.arange(len(close)), [e.observed_at for e in events]),
    )


def test_truncation_and_future_mutation_do_not_change_earlier_signals() -> None:
    bars = _bars(300)
    bank = build_divergence_bank(bars, names=("RSI", "MACD_HIST"))
    cfg = DivergenceConfig(pivot_period=3)
    full = detect_divergences(
        high=bars.high, low=bars.low, close=bars.close, indicators=bank, config=cfg
    )
    k = 210
    prefix = detect_divergences(
        high=bars.high[:k],
        low=bars.low[:k],
        close=bars.close[:k],
        indicators={name: data[:k] for name, data in bank.items()},
        config=cfg,
    )
    for field in ("bull_regular", "bear_regular", "bull_hidden", "bear_hidden"):
        np.testing.assert_array_equal(getattr(full, field)[:k], getattr(prefix, field))


@pytest.mark.parametrize(
    "bad",
    [
        {"pivot_period": 0},
        {"max_bars": 1},
        {"min_indicators": 0},
    ],
)
def test_reject_invalid_configuration(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DivergenceConfig(**bad)


def test_indicator_nan_breaks_divergence_continuity_even_without_line_filter() -> None:
    close = np.asarray(
        [11, 10, 9, 8, 7, 8, 9, 10, 9, 8, 7, 6, 7, 8, 9, 8, 9], dtype=float
    )
    osc = np.asarray(
        [0, 0, 0, 0, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], dtype=float
    )
    osc[8] = np.nan
    output = detect_divergences(
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        indicators={"RSI": osc},
        config=DivergenceConfig(pivot_period=2, min_indicators=1, validate_lines=False),
    )
    assert all(not (e.pivot_at < 8 < e.anchor_at) for e in output.events)


def test_extended_indicators_do_not_reuse_future_or_bridge_gaps() -> None:
    raw = _bars(180)
    names = (
        "MOM",
        "ROC",
        "OBV",
        "VWMACD",
        "PPO",
        "AWESOME",
        "ICHIMOKU_TENKAN",
        "ICHIMOKU_KIJUN",
    )
    actual = build_divergence_bank(raw, names=names)
    before = _bars(155)
    shorter = build_divergence_bank(before, names=names)
    for name in names:
        np.testing.assert_array_equal(actual[name][:155], shorter[name])
        assert np.isfinite(actual[name][-1])
    assert np.isnan(actual["ICHIMOKU_KIJUN"][:25]).all()

    active = raw.tradable.copy()
    active[100] = False
    gapped = RawMarketSeries(
        timestamps=raw.timestamps,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        tradable=active,
    )
    restarted = build_divergence_bank(gapped, names=names)
    for name in names:
        assert np.isnan(restarted[name][100])
    assert np.isnan(restarted["PPO"][124])
    assert np.isfinite(restarted["PPO"][126])


@pytest.mark.parametrize(
    ("kind", "reverse_price", "hidden"),
    [
        ("bull_regular", False, False),
        ("bear_regular", True, False),
        ("bull_hidden", False, True),
        ("bear_hidden", True, True),
    ],
)
def test_four_divergence_classes_are_distinct_and_observed_causally(
    kind: str, reverse_price: bool, hidden: bool
) -> None:
    regular = np.asarray(
        [11, 10.5, 10, 9, 8, 9, 10, 11, 10, 9.5, 9, 8.5, 7, 8, 9, 10],
        dtype=np.float64,
    )
    hidden_price = np.asarray(
        [11, 10.5, 10, 9, 8, 9, 10, 11, 10, 9.5, 9.8, 9.5, 9, 10, 11, 12],
        dtype=np.float64,
    )
    regular_osc = np.asarray(
        [0, -1, -2, -4, -6, -5, -4, -3, -2, -1, -2, -2, -3, -2, 0, 1],
        dtype=np.float64,
    )
    hidden_osc = np.asarray(
        [0, -1, -2, -3, -3, -2, -1, 0, -1, -3, -4, -5, -6, -5, -4, -3],
        dtype=np.float64,
    )
    close = hidden_price if hidden else regular
    osc = hidden_osc if hidden else regular_osc
    if reverse_price:
        close = 20 - close
        osc = -osc
    found = detect_divergences(
        close=close,
        high=close + 0.1,
        low=close - 0.1,
        indicators={"TEST": osc},
        config=DivergenceConfig(
            pivot_period=2,
            min_indicators=1,
            validate_lines=False,
        ),
    )
    matching = [event for event in found.events if event.kind == kind]
    assert matching, kind
    assert all(
        event.observed_at > event.anchor_at > event.pivot_at for event in matching
    )


def test_constant_prices_have_no_divergence() -> None:
    n = 100
    c = np.full(n, 100.0)
    osc = np.sin(np.arange(n, dtype=float))
    output = detect_divergences(
        close=c,
        high=c + 0.5,
        low=c - 0.5,
        indicators={"TEST": osc},
        config=DivergenceConfig(min_indicators=1),
    )
    assert output.events == ()
    assert not np.any(output.bull_total + output.bear_total)


@pytest.mark.parametrize("bearish", [False, True])
def test_high_low_penetration_uses_actual_pivot_price_line(bearish: bool) -> None:
    """Known review regression: an interim wick pierces the pivot price line.

    Close-only penetration incorrectly leaves the divergence alive. Both
    bullish low-to-low and bearish high-to-high optional modes must reject it.
    The disabled filter must still demonstrate this is a filter rejection.
    """
    original_low = np.asarray(
        [96, 95, 94, 90, 92, 94, 86, 89, 92, 93],
        dtype=float,
    )
    original_close = np.asarray(
        [100, 100, 100, 100, 100.6, 100.8, 101, 100.8, 101.1, 101.1],
        dtype=float,
    )
    original_osc = np.asarray(
        [30, 30, 30, 30, 32, 33, 36, 35, 38, 39],
        dtype=float,
    )
    if bearish:
        close = 200 - original_close
        high = 200 - original_low
        low = close - 1
        osc = -original_osc
        kind = "bear_regular"
    else:
        close = original_close
        high = close + 1
        low = original_low
        osc = original_osc
        kind = "bull_regular"

    def exact_event(validate_lines: bool) -> bool:
        out = detect_divergences(
            high=high,
            low=low,
            close=close,
            indicators={"TEST": osc},
            config=DivergenceConfig(
                pivot_period=1,
                min_indicators=1,
                pivot_source="high_low",
                validate_lines=validate_lines,
            ),
        )
        return any(
            e.kind == kind
            and e.observed_at == 8
            and e.pivot_at == 3
            and e.anchor_at == 7
            for e in out.events
        )

    assert exact_event(False), "the fixture must produce a divergence without filtering"
    assert not exact_event(True), "the wick must invalidate the high/low price line"


def test_high_low_pivot_mode_preserves_prefix_causality() -> None:
    raw = _bars(350)
    indicators = build_divergence_bank(raw, names=("RSI", "MACD_HIST"))
    config = DivergenceConfig(
        pivot_source="high_low",
        pivot_period=2,
        min_indicators=1,
        validate_lines=True,
    )
    full = detect_divergences(
        high=raw.high,
        low=raw.low,
        close=raw.close,
        indicators=indicators,
        config=config,
    )
    cutoff = 240
    part = detect_divergences(
        high=raw.high[:cutoff],
        low=raw.low[:cutoff],
        close=raw.close[:cutoff],
        indicators={k: v[:cutoff] for k, v in indicators.items()},
        config=config,
    )
    for kind in ("bull_regular", "bear_regular", "bull_hidden", "bear_hidden"):
        np.testing.assert_array_equal(
            getattr(full, kind)[:cutoff],
            getattr(part, kind),
        )
    assert tuple(e for e in full.events if e.observed_at < cutoff) == part.events


@pytest.mark.parametrize("cadence", [1, 24, 72, 168])
def test_custom_native_periods_respect_full_dependency_window_and_prefix(cadence):
    raw = _bars(60)
    timestamps = np.datetime64("2024-01-01T00:00", "ns") + np.arange(
        60
    ) * np.timedelta64(cadence, "h")
    raw = replace(raw, timestamps=timestamps, available_at=timestamps)
    params = dict(tenkan_period=19, kijun_period=9, span_b_period=5, displacement=4)
    good = visible_ichimoku_cloud(raw, **params)
    assert not good.ready[:22].any() and good.ready[22]
    # Row 10 belongs only to the longer Tenkan dependency at row 22.
    active = raw.tradable.copy()
    active[10] = False
    with_gap = visible_ichimoku_cloud(replace(raw, tradable=active), **params)
    assert not with_gap.ready[22:33].any()
    assert with_gap.ready[33]
    source = 33 - 4
    tenkan = (
        max(raw.high[source - 18 : source + 1]) + min(raw.low[source - 18 : source + 1])
    ) / 2
    kijun = (
        max(raw.high[source - 8 : source + 1]) + min(raw.low[source - 8 : source + 1])
    ) / 2
    assert with_gap.span_a[33] == pytest.approx((tenkan + kijun) / 2)
    prefix_raw = _bars(45)
    prefix_raw = replace(
        prefix_raw,
        timestamps=timestamps[:45],
        available_at=timestamps[:45],
        tradable=active[:45],
    )
    prefix = visible_ichimoku_cloud(prefix_raw, **params)
    np.testing.assert_array_equal(prefix.top, with_gap.top[:45])
    np.testing.assert_array_equal(prefix.ready, with_gap.ready[:45])


def test_visible_cloud_never_emits_nonfinite_ready_prices():
    raw = _bars(80)
    price = np.full(80, 1.7e308)
    raw = replace(raw, open=price, high=price, low=price, close=price)
    cloud = visible_ichimoku_cloud(raw)
    assert cloud.ready[77:].all()
    assert np.isfinite(cloud.top[77:]).all()
    assert np.array_equal(cloud.top[77:], price[77:])


@pytest.mark.parametrize(
    "bad", [np.array(1.0), np.empty((2, 1)), np.array([1 + 1j, 2 + 0j])]
)
def test_detector_rejects_nonvector_or_nonreal_price_inputs(bad):
    with pytest.raises(ValueError, match="price"):
        detect_divergences(high=bad, low=bad, close=bad, indicators={"RSI": np.ones(2)})


def test_detector_rejects_nonreal_oscillator_and_invalid_config():
    c = np.ones(10)
    with pytest.raises(ValueError, match="indicator"):
        detect_divergences(
            high=c, low=c, close=c, indicators={"RSI": np.ones(10, dtype=complex)}
        )
    with pytest.raises(TypeError, match="DivergenceConfig"):
        detect_divergences(high=c, low=c, close=c, indicators={"RSI": c}, config=False)


@pytest.mark.parametrize("kind", ["masked", "buffer", "opaque"])
@pytest.mark.parametrize("field", ["price", "oscillator"])
def test_public_divergence_refuses_unverifiable_array_owners(kind, field):
    values = np.arange(1.0, 11.0)
    masked = np.ma.array(values, mask=[True] + [False] * 9)
    if kind == "masked":
        damaged = masked
    elif kind == "buffer":
        damaged = np.frombuffer(memoryview(masked), dtype=values.dtype)
    else:
        damaged = np.lib.stride_tricks.as_strided(
            masked, shape=values.shape, strides=values.strides
        )
    price = damaged if field == "price" else values
    oscillator = damaged if field == "oscillator" else values
    with pytest.raises(ValueError, match="native array"):
        detect_divergences(
            high=price, low=price, close=price, indicators={"RSI": oscillator}
        )


_ALL_BANK_NAMES = (
    "RSI",
    "MACD",
    "MACD_HIST",
    "MACD_SIGNAL",
    "STOCH_K",
    "STOCH_D",
    "CCI",
    "MFI",
    "CMF",
    "WILLR",
    "OBV_SLOPE",
    "DI_SPREAD",
    "MOM",
    "ROC",
    "OBV",
    "VWMACD",
    "PPO",
    "AWESOME",
    "ICHIMOKU_TENKAN",
    "ICHIMOKU_KIJUN",
)


def _replace_raw(raw: RawMarketSeries, **changes: np.ndarray) -> RawMarketSeries:
    fields = {
        "timestamps": raw.timestamps,
        "available_at": raw.available_at,
        "open": raw.open,
        "high": raw.high,
        "low": raw.low,
        "close": raw.close,
        "volume": raw.volume,
        "funding_rate": raw.funding_rate,
        "tradable": raw.tradable,
    }
    fields.update(changes)
    return RawMarketSeries(**fields)


def test_stoch_d_is_a_14_period_k_with_3_period_smoothing() -> None:
    raw = _bars(80)
    stoch_d = build_divergence_bank(raw, names=("STOCH_D",))["STOCH_D"]

    # 14-bar %K needs 14 rows and the 3-bar mean two more: first value at 15.
    assert np.isnan(stoch_d[:15]).all()
    assert np.isfinite(stoch_d[15:]).all()

    def k(index: int) -> float:
        highest = float(np.max(raw.high[index - 13 : index + 1]))
        lowest = float(np.min(raw.low[index - 13 : index + 1]))
        return 2.0 * (float(raw.close[index]) - lowest) / (highest - lowest) - 1.0

    for index in (15, 40, 79):
        expected = (k(index - 2) + k(index - 1) + k(index)) / 3.0
        assert stoch_d[index] == pytest.approx(expected, abs=1e-12)


def test_late_published_row_is_a_gap_for_every_bank_indicator() -> None:
    raw = _bars(220)
    late_row = 120
    available_at = np.array(raw.available_at, copy=True)
    available_at[late_row] = raw.timestamps[late_row] + np.timedelta64(1, "ns")
    late = _replace_raw(raw, available_at=available_at)
    tradable = np.array(raw.tradable, copy=True)
    tradable[late_row] = False
    blocked = _replace_raw(raw, tradable=tradable)

    actual = build_divergence_bank(late, names=_ALL_BANK_NAMES)
    expected = build_divergence_bank(blocked, names=_ALL_BANK_NAMES)
    clean = build_divergence_bank(raw, names=_ALL_BANK_NAMES)
    for name in _ALL_BANK_NAMES:
        # A row is never read before it is published, and the calculation
        # restarts after it instead of bridging the unpublished row.
        np.testing.assert_array_equal(actual[name], expected[name], strict=True)
        assert np.isnan(actual[name][late_row]), name
        np.testing.assert_array_equal(actual[name][:late_row], clean[name][:late_row])
    assert np.isnan(actual["RSI"][late_row + 1 : late_row + 15]).all()
    assert np.isfinite(actual["RSI"][late_row + 15])


def test_a_uniformly_delayed_series_yields_no_bank_values() -> None:
    bank = build_divergence_bank(_bars(200, delayed=True), names=_ALL_BANK_NAMES)
    for name, values in bank.items():
        assert np.isnan(values).all(), name
