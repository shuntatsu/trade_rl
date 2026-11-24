"""Literal and metamorphic oracles for as-of 4h ADX/ATR context."""

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.features.cloud_trend_context import build_cloud_trend_context
from trade_rl.data.source import RawMarketSeries


def bars(n: int = 160) -> RawMarketSeries:
    t = np.datetime64("2022-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n)
    return RawMarketSeries(
        timestamps=t,
        open=close - 0.25,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.ones(n) * 10000,
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=bool),
    )


def test_literal_monotonic_series_and_native_units():
    # Four-hour bars: first true range 4, then 4.5 on every next bar.
    # With a constant +DM and zero -DM, every valid DX=100, ADX=100.
    raw = bars()
    ctx = build_cloud_trend_context(raw)
    first = 28 * 4 - 1
    assert not ctx.ready[:first].any()
    assert ctx.ready[first:].all()
    assert ctx.adx_fraction[first] == pytest.approx(1.0)
    atr = (4.0 + 13 * 4.5) / 14
    for _ in range(14, 28):
        atr = (13 * atr + 4.5) / 14
    assert ctx.atr_fraction[first] == pytest.approx(atr / raw.close[first])
    np.testing.assert_array_equal(
        ctx.atr_fraction[first : first + 4], np.full(4, ctx.atr_fraction[first])
    )
    assert not ctx.atr_fraction.flags.writeable
    with pytest.raises(ValueError):
        ctx.atr_fraction.setflags(write=True)


def test_delayed_bar_blocks_until_published_without_using_stale_value():
    raw = bars()
    avail = raw.available_at.copy()
    avail[119] += np.timedelta64(2, "h")
    ctx = build_cloud_trend_context(replace(raw, available_at=avail))
    assert ctx.ready[118]
    assert not ctx.ready[119:121].any()
    assert ctx.ready[121]
    prefix = build_cloud_trend_context(
        replace(
            raw,
            timestamps=raw.timestamps[:121],
            available_at=avail[:121],
            open=raw.open[:121],
            high=raw.high[:121],
            low=raw.low[:121],
            close=raw.close[:121],
            volume=raw.volume[:121],
            funding_rate=raw.funding_rate[:121],
            tradable=raw.tradable[:121],
            funding_available=raw.funding_available[:121],
            funding_event_count=raw.funding_event_count[:121],
        )
    )
    np.testing.assert_array_equal(prefix.ready, ctx.ready[:121])
    np.testing.assert_array_equal(prefix.atr_fraction, ctx.atr_fraction[:121])


def test_nontradable_bar_resets_wilder_warmup_and_gaps_fail_closed():
    raw = bars(280)
    tradable = raw.tradable.copy()
    tradable[120] = False
    ctx = build_cloud_trend_context(replace(raw, tradable=tradable))
    assert not ctx.ready[120]
    assert not ctx.ready[123:235].any()
    assert ctx.ready[235]
    stamps = raw.timestamps.copy()
    stamps[5:] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="regular"):
        build_cloud_trend_context(replace(raw, timestamps=stamps, available_at=stamps))


def test_27_completed_bars_cannot_appear_ready_when_future_bar_is_added():
    short = build_cloud_trend_context(bars(108))
    full = build_cloud_trend_context(bars(112))
    assert not short.ready.any()
    np.testing.assert_array_equal(short.ready, full.ready[:108])
    assert full.ready[111]
