"""Contract tests for causal multi-indicator divergence and visible Ichimoku cloud."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.features.divergence import DivergenceConfig, detect_divergences
from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
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


def test_mtf_uses_only_closed_4h_and_daily_bars_and_is_causal() -> None:
    bars = _bars()
    pack = build_divergence_cloud_mtf(bars, indicators=("RSI", "MACD_HIST"))
    assert len(pack.timestamps) == len(bars.close)
    assert pack.cloud_ready.any()
    assert not pack.cloud_ready[: 24 * 77].any()
    confirmed_4h = (bars.timestamps.astype("datetime64[h]").astype("int64") % 4) == 0
    assert np.all(pack.bull_4h[~confirmed_4h] == 0)
    assert np.all(pack.bear_4h[~confirmed_4h] == 0)
    k = 24 * 87 + 7
    shorter = _bars(k)
    prefix = build_divergence_cloud_mtf(shorter, indicators=("RSI", "MACD_HIST"))
    for key in (
        "cloud_state",
        "cloud_ready",
        "bull_1h",
        "bear_1h",
        "bull_4h",
        "bear_4h",
    ):
        np.testing.assert_array_equal(getattr(pack, key)[:k], getattr(prefix, key))


def test_delayed_data_cannot_create_early_signal() -> None:
    fast = build_divergence_cloud_mtf(_bars(), indicators=("RSI",))
    delayed = build_divergence_cloud_mtf(_bars(delayed=True), indicators=("RSI",))
    assert not delayed.cloud_ready.any()
    assert np.any(fast.cloud_ready)
    for idx in np.flatnonzero(fast.bull_1h + fast.bear_1h):
        assert delayed.bull_1h[idx] + delayed.bear_1h[idx] == 0


def test_bad_gap_rejected_without_silent_reshaping() -> None:
    x = _bars(120)
    deleted = np.arange(len(x.close)) != 54
    bad = RawMarketSeries(
        timestamps=x.timestamps[deleted],
        open=x.open[deleted],
        high=x.high[deleted],
        low=x.low[deleted],
        close=x.close[deleted],
        volume=x.volume[deleted],
        funding_rate=x.funding_rate[deleted],
        tradable=x.tradable[deleted],
    )
    with pytest.raises(ValueError, match="regular"):
        build_divergence_cloud_mtf(bad)


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


def test_late_earlier_candle_blocks_feature_until_available() -> None:
    raw = _bars(24 * 90)
    available = raw.available_at.copy()
    assert available is not None
    late_idx = 24 * 81
    available[late_idx] += np.timedelta64(48, "h")
    modified = RawMarketSeries(
        timestamps=raw.timestamps,
        available_at=available,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        tradable=raw.tradable,
    )
    base = build_divergence_cloud_mtf(raw, indicators=("RSI",))
    lagged = build_divergence_cloud_mtf(modified, indicators=("RSI",))
    assert base.cloud_ready[late_idx + 24]
    assert not lagged.cloud_ready[late_idx + 24]
    assert lagged.cloud_ready[late_idx + 49]


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
