"""Synthetic literal oracle and window-work regression for visible Ichimoku."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.features.indicators import ichimoku_cloud as module
from trade_rl.data.source import RawMarketSeries


def _bars(n: int, minutes: int, *, gap: int | None = None) -> RawMarketSeries:
    times = np.datetime64("2024-01-01", "ns") + np.arange(n) * np.timedelta64(
        minutes, "m"
    )
    i = np.arange(n, dtype=float)
    price = 100.0 + i / 1000 + np.sin(i / 11) * 4
    active = np.ones(n, dtype=bool)
    if gap is not None:
        active[gap] = False
    return RawMarketSeries(
        timestamps=times,
        available_at=times + np.timedelta64(1, "s"),
        open=price,
        high=price + 2,
        low=price - 1,
        close=price,
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        tradable=active,
    )


@pytest.mark.parametrize("minutes", [15, 60, 4320, 10080])
@pytest.mark.parametrize("periods", [(9, 26, 52, 26), (41, 7, 12, 3), (7, 7, 7, 1)])
def test_visible_cloud_matches_literal_window_oracle(minutes, periods):
    raw = _bars(220, minutes, gap=75)
    tenkan, kijun, span_b, displacement = periods
    cloud = module.visible_ichimoku_cloud(
        raw,
        tenkan_period=tenkan,
        kijun_period=kijun,
        span_b_period=span_b,
        displacement=displacement,
    )
    longest = max(tenkan, kijun, span_b)
    expected_a = np.full(len(raw.close), np.nan)
    expected_b = expected_a.copy()
    expected_ready = np.zeros(len(raw.close), dtype=bool)
    for t in range(longest - 1 + displacement, len(raw.close)):
        s = t - displacement
        if not all(raw.tradable[k] for k in range(s - longest + 1, t + 1)):
            continue

        def mid(period):
            return (
                max(float(raw.high[k]) for k in range(s - period + 1, s + 1))
                + min(float(raw.low[k]) for k in range(s - period + 1, s + 1))
            ) * 0.5

        expected_a[t] = (mid(tenkan) + mid(kijun)) * 0.5
        expected_b[t] = mid(span_b)
        expected_ready[t] = True
    np.testing.assert_array_equal(cloud.ready, expected_ready)
    np.testing.assert_array_equal(cloud.span_a, expected_a)
    np.testing.assert_array_equal(cloud.span_b, expected_b)
    np.testing.assert_array_equal(cloud.top, np.maximum(expected_a, expected_b))
    np.testing.assert_array_equal(cloud.bottom, np.minimum(expected_a, expected_b))
    assert not any(
        a.flags.writeable
        for a in (cloud.ready, cloud.span_a, cloud.span_b, cloud.top, cloud.bottom)
    )


def test_visible_cloud_does_not_rescan_long_dependency_windows(monkeypatch):
    raw = _bars(4096, 15, gap=7)

    class CountingNumpy:
        scanned = 0

        def __getattr__(self, name):
            return getattr(np, name)

        def _reduce(self, name, values, *args, **kwargs):
            self.scanned += np.asarray(values).size
            return getattr(np, name)(values, *args, **kwargs)

        def max(self, values, *args, **kwargs):
            return self._reduce("max", values, *args, **kwargs)

        def min(self, values, *args, **kwargs):
            return self._reduce("min", values, *args, **kwargs)

        def all(self, values, *args, **kwargs):
            return self._reduce("all", values, *args, **kwargs)

    counter = CountingNumpy()
    monkeypatch.setattr(module, "np", counter)
    cloud = module.visible_ichimoku_cloud(
        raw,
        tenkan_period=512,
        kijun_period=1024,
        span_b_period=2048,
        displacement=256,
    )
    assert cloud.ready[-1]
    # Bound the total number of elements supplied to full-array reductions.
    # The former per-decision window scans exceed this bound by orders of magnitude.
    assert counter.scanned <= 16 * len(raw.close)


@pytest.mark.parametrize("minutes", [15, 60])
def test_two_year_long_period_cloud_against_sampled_literal_oracle(minutes):
    raw = _bars(2 * 365 * 24 * 60 // minutes, minutes, gap=7)
    periods = (512, 2048, 8192, 256)
    cloud = module.visible_ichimoku_cloud(
        raw,
        tenkan_period=periods[0],
        kijun_period=periods[1],
        span_b_period=periods[2],
        displacement=periods[3],
    )
    first = 8192 - 1 + 256
    assert not cloud.ready[: first + 8].any()
    assert cloud.ready[first + 8 :].all()
    for t in (first + 8, first + 9, len(raw.close) // 2, len(raw.close) - 1):
        if t < first + 8:
            continue
        s = t - 256

        def mid(period):
            return (
                max(float(raw.high[k]) for k in range(s - period + 1, s + 1))
                + min(float(raw.low[k]) for k in range(s - period + 1, s + 1))
            ) * 0.5

        assert cloud.span_a[t] == (mid(512) + mid(2048)) * 0.5
        assert cloud.span_b[t] == mid(8192)


def test_unreachable_warmup_does_not_allocate_period_sized_storage():
    raw = _bars(3, 60)
    cloud = module.visible_ichimoku_cloud(raw, span_b_period=10**80)
    assert not cloud.ready.any()
    assert np.isnan(cloud.top).all()
