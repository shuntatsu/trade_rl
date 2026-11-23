"""Point-in-time 30d market direction and visible-daily-cloud context.

Used only by opt-in research. Does not mutate the frozen MTF pack, feature
rosters, Dataset schema, core indicator semantics or native account.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud
from trade_rl.data.source import RawMarketSeries

_NS_HOUR = 3_600_000_000_000
_MOMENTUM_HOURS = 30 * 24
_CLOUD_SLOPE_DAYS = 7


@dataclass(frozen=True, slots=True)
class CloudRegimeContext:
    timestamps: np.ndarray
    momentum_30d: np.ndarray
    cloud_mid_slope_7d: np.ndarray
    distance_to_cloud_mid_pct: np.ndarray
    realized_vol_30d: np.ndarray
    ready: np.ndarray


def build_cloud_regime_context(
    raw: RawMarketSeries,
    signals: DivergenceCloudMtfPack,
) -> CloudRegimeContext:
    """Build as-of 1h context, including **only completed daily visible clouds**.

    At decision t, each rolling return uses bars in [t-30d, t].
    Cloud centre uses the latest *published and completed* daily bar and
    compares it with the one seven completed daily bars earlier. Source
    publication delays cause a fail-closed mask; never retro-fill signals.
    """
    time = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    if (len(time) > 1 and not np.all(np.diff(time) == _NS_HOUR)) or np.any(
        time % _NS_HOUR != 0
    ):
        raise ValueError("regular hourly UTC close times required")
    if raw.timestamps.shape != signals.timestamps.shape or not np.array_equal(
        raw.timestamps, signals.timestamps
    ):
        raise ValueError("context and MTF signal clocks must agree")
    n = len(time)
    assert raw.available_at is not None
    available = raw.available_at.astype("datetime64[ns]").astype(np.int64)
    latest_required = np.maximum.accumulate(available)
    close = raw.close
    momentum = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)
    distance = np.full(n, np.nan, dtype=np.float64)
    volatility = np.full(n, np.nan, dtype=np.float64)
    ready = np.zeros(n, dtype=np.bool_)

    daily_end = np.flatnonzero(time % (24 * _NS_HOUR) == 0)
    daily_end = daily_end[daily_end >= 23]
    daily_count = len(daily_end)
    cloud = None
    daily_time = None
    daily_known = None
    if daily_count:
        starts = daily_end - 23
        high = np.asarray(
            [raw.high[j : k + 1].max() for j, k in zip(starts, daily_end, strict=True)]
        )
        low = np.asarray(
            [raw.low[j : k + 1].min() for j, k in zip(starts, daily_end, strict=True)]
        )
        volumes = np.asarray(
            [
                raw.volume[j : k + 1].sum(dtype=np.float64)
                for j, k in zip(starts, daily_end, strict=True)
            ]
        )
        if not np.isfinite(volumes).all():
            raise ValueError("daily volume aggregation became nonfinite")
        tradable = np.asarray(
            [
                raw.tradable[j : k + 1].all()
                for j, k in zip(starts, daily_end, strict=True)
            ],
            dtype=np.bool_,
        )
        daily_available = np.asarray(
            [
                raw.available_at[j : k + 1].max()
                for j, k in zip(starts, daily_end, strict=True)
            ]
        )
        daily = RawMarketSeries(
            timestamps=raw.timestamps[daily_end],
            available_at=daily_available,
            open=raw.open[starts],
            high=high,
            low=low,
            close=raw.close[daily_end],
            volume=volumes,
            funding_rate=np.zeros(daily_count, dtype=np.float64),
            tradable=tradable,
        )
        cloud = visible_ichimoku_cloud(daily)
        daily_time = time[daily_end]
        daily_known = np.maximum.accumulate(
            daily_available.astype("datetime64[ns]").astype(np.int64)
        )

    if cloud is not None:
        assert daily_time is not None and daily_known is not None
        cum_inactive = np.r_[0, np.cumsum(~raw.tradable, dtype=np.int64)]
        returns = np.diff(np.log(close))
        for i in range(_MOMENTUM_HOURS, n):
            if (
                not raw.tradable[i]
                or latest_required[i] > time[i]
                or cum_inactive[i + 1] != cum_inactive[i - _MOMENTUM_HOURS]
            ):
                continue
            # Daily publish-time prefix maxima preserve dependency on a late
            # earlier candle even if the latest daily candle arrived on time.
            d = int(np.searchsorted(daily_known, time[i], side="right") - 1)
            if (
                d < _CLOUD_SLOPE_DAYS
                or daily_time[d] > time[i]
                or time[i] - daily_time[d] >= 24 * _NS_HOUR
                or not cloud.ready[d]
                or not cloud.ready[d - _CLOUD_SLOPE_DAYS]
                or not bool(signals.cloud_ready[i])
            ):
                continue
            mid = 0.5 * (cloud.top[d] + cloud.bottom[d])
            previous_mid = 0.5 * (
                cloud.top[d - _CLOUD_SLOPE_DAYS] + cloud.bottom[d - _CLOUD_SLOPE_DAYS]
            )
            if mid <= 0 or previous_mid <= 0:
                raise ValueError("cloud midpoint must remain positive")
            momentum[i] = float(close[i] / close[i - _MOMENTUM_HOURS] - 1)
            slope[i] = float(mid / previous_mid - 1)
            distance[i] = float((close[i] / mid - 1) * 100)
            volatility[i] = float(
                np.std(returns[i - _MOMENTUM_HOURS : i], ddof=1) * np.sqrt(8760)
            )
            if not all(
                np.isfinite(x)
                for x in (momentum[i], slope[i], distance[i], volatility[i])
            ):
                raise ValueError("regime metric became nonfinite")
            ready[i] = True

    for output in (momentum, slope, distance, volatility, ready):
        output.setflags(write=False)
    return CloudRegimeContext(
        timestamps=raw.timestamps,
        momentum_30d=momentum,
        cloud_mid_slope_7d=slope,
        distance_to_cloud_mid_pct=distance,
        realized_vol_30d=volatility,
        ready=ready,
    )


__all__ = ["CloudRegimeContext", "build_cloud_regime_context"]
