"""Regime diagnostics must be causal, as-of, source-bound and never backdated."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.features.cloud_regime_context import build_cloud_regime_context
from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
from trade_rl.data.source import RawMarketSeries


def _bars(n: int = 24 * 115, late: tuple[int, int] | None = None) -> RawMarketSeries:
    time = np.datetime64("2022-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    t = np.arange(n, dtype=float)
    price = 100 + 0.02 * t + 1.2 * np.sin(t / 6) + 3 * np.sin(t / 91)
    op = price - 0.05
    high = np.maximum(op, price) + 0.3
    low = np.minimum(op, price) - 0.3
    available = time.copy()
    if late:
        available[late[0]] += np.timedelta64(late[1], "h")
    return RawMarketSeries(
        timestamps=time,
        available_at=available,
        open=op,
        high=high,
        low=low,
        close=price,
        volume=np.full(n, 10_000.0),
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=np.bool_),
    )


def test_context_prefix_causal_and_positive_trend() -> None:
    full = _bars()
    pack = build_divergence_cloud_mtf(full, indicators=("RSI",))
    features = build_cloud_regime_context(full, pack)
    assert not features.ready[: 24 * 80].any()
    assert features.ready[24 * 102]
    assert features.momentum_30d[24 * 102] > 0
    assert np.isfinite(features.cloud_mid_slope_7d[24 * 102])
    assert np.isfinite(features.distance_to_cloud_mid_pct[24 * 102])
    assert np.isfinite(features.realized_vol_30d[24 * 102])
    assert not features.ready.flags.writeable
    assert not features.momentum_30d.flags.writeable
    k = 24 * 105 + 5
    part = _bars(k)
    part_pack = build_divergence_cloud_mtf(part, indicators=("RSI",))
    short = build_cloud_regime_context(part, part_pack)
    for name in (
        "ready",
        "momentum_30d",
        "cloud_mid_slope_7d",
        "distance_to_cloud_mid_pct",
        "realized_vol_30d",
    ):
        np.testing.assert_array_equal(getattr(features, name)[:k], getattr(short, name))


def test_late_candle_blocks_context_until_observable() -> None:
    late_idx = 24 * 95
    delayed = _bars(late=(late_idx, 48))
    pack = build_divergence_cloud_mtf(delayed, indicators=("RSI",))
    result = build_cloud_regime_context(delayed, pack)
    assert not result.ready[late_idx + 12]
    assert not result.ready[late_idx + 24]
    assert result.ready[late_idx + 49]


def test_irregular_or_misaligned_inputs_fail_closed() -> None:
    bars = _bars(24 * 90)
    pack = build_divergence_cloud_mtf(bars, indicators=("RSI",))
    shortened = _bars(24 * 89)
    with pytest.raises(ValueError, match="clock"):
        build_cloud_regime_context(shortened, pack)
    mask = np.arange(len(bars.close)) != 100
    bad = RawMarketSeries(
        timestamps=bars.timestamps[mask],
        open=bars.open[mask],
        high=bars.high[mask],
        low=bars.low[mask],
        close=bars.close[mask],
        volume=bars.volume[mask],
        funding_rate=bars.funding_rate[mask],
        tradable=bars.tradable[mask],
    )
    with pytest.raises(ValueError, match="regular"):
        build_cloud_regime_context(bad, pack)
