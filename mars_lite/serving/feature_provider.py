"""Cached CSV feature snapshots for the online serving plane."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from mars_lite.serving.market_time import (
    SUPPORTED_BASE_TIMEFRAMES,
    resolve_completed_bar_endpoint,
)
from mars_lite.serving.runtime import FeatureSnapshot, ServingRuntime
from mars_lite.serving.snapshot_identity import compute_snapshot_id


def required_history_bars(
    rank_window: int,
    vol_lookback: int,
    trend_config: Mapping[str, int],
) -> int:
    trend_values = (
        trend_config.get("fast_lookback", 0),
        trend_config.get("base_lookback", 0),
        trend_config.get("slow_lookback", 0),
        trend_config.get("rebalance_every", 0),
    )
    if rank_window <= 0 or vol_lookback < 0 or any(value < 0 for value in trend_values):
        raise ValueError(
            "history windows must be non-negative and rank_window positive"
        )
    trend_lookback = max(trend_values[:3])
    trend_rebalance = trend_values[3]
    # Absolute-time TrendFamily evaluates at the last rebalance slot. At an
    # arbitrary inference bar that slot can be up to rebalance_every-1 bars
    # behind the endpoint, so retain both the lookback and that offset.
    trend_history = trend_lookback + max(trend_rebalance, 1)
    return max(rank_window, vol_lookback + 1, trend_history, 2)


class CsvFeatureProvider:
    def __init__(
        self,
        *,
        runtime: ServingRuntime,
        data_dir: str | Path,
        cache_ttl_seconds: float = 30.0,
        clock: Callable[[], np.datetime64] | None = None,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        self.runtime = runtime
        self.data_dir = Path(data_dir)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock or (lambda: np.datetime64("now", "ns"))
        self._lock = threading.RLock()
        self._cached: FeatureSnapshot | None = None
        self._cached_digest: str | None = None
        self._cached_until = 0.0

    def get_snapshot(self) -> FeatureSnapshot:
        bundle = self.runtime.active_bundle()
        if bundle is None:
            raise RuntimeError("serving runtime has no active bundle")
        now = time.monotonic()
        with self._lock:
            if (
                self._cached is not None
                and self._cached_digest == bundle.bundle_digest
                and now < self._cached_until
            ):
                return self._cached

        from mars_lite.data.sources import create_source
        from mars_lite.features.feature_pipeline import FeaturePipeline

        symbols = list(bundle.metadata["symbols"])
        run_config = dict(bundle.metadata.get("run_config") or {})
        base_timeframe = str(run_config.get("base_timeframe", "1h"))
        if base_timeframe not in SUPPORTED_BASE_TIMEFRAMES:
            raise ValueError(f"unsupported bundled base_timeframe: {base_timeframe!r}")
        source = create_source("csv", symbols, data_dir=self.data_dir)
        feature_set = FeaturePipeline(symbols, base_timeframe=base_timeframe).build(
            source
        )
        if tuple(feature_set.symbols) != tuple(symbols):
            raise ValueError(
                "feature provider symbol order does not match active bundle"
            )
        if not feature_set.n_bars:
            raise ValueError("feature provider produced no bars")

        rank_window = int(bundle.preprocessing.get("rank_window", 250))
        post_config = dict(bundle.metadata.get("post_processor") or {})
        vol_lookback = int(post_config.get("vol_lookback", 60))
        raw_trend_config = dict(bundle.metadata.get("trend_family") or {})
        trend_config = {
            key: int(raw_trend_config.get(key, 0))
            for key in (
                "fast_lookback",
                "base_lookback",
                "slow_lookback",
                "rebalance_every",
            )
        }
        history_bars = required_history_bars(
            rank_window,
            vol_lookback,
            trend_config,
        )
        endpoint = resolve_completed_bar_endpoint(
            feature_set.timestamps,
            base_timeframe=base_timeframe,
            now_utc=self._clock(),
        )
        end = endpoint.end_exclusive
        start = max(0, end - history_bars)
        timestamps = feature_set.timestamps[start:end]
        feature_history = feature_set.features[start:end]
        close_history = feature_set.close[start:end]
        feature_history_array = np.asarray(feature_history, dtype=np.float64)
        global_features_array = np.asarray(
            feature_set.global_features[end - 1], dtype=np.float64
        )
        close_history_array = np.asarray(close_history, dtype=np.float64)
        snapshot_id = compute_snapshot_id(
            bundle_digest=bundle.bundle_digest,
            base_timeframe=base_timeframe,
            timestamps=np.asarray(timestamps),
            symbols=tuple(feature_set.symbols),
            feature_names=tuple(feature_set.feature_names),
            global_feature_names=tuple(feature_set.global_feature_names),
            feature_history=feature_history_array,
            global_features=global_features_array,
            close_history=close_history_array,
        )
        snapshot = FeatureSnapshot(
            snapshot_id=snapshot_id,
            symbols=tuple(feature_set.symbols),
            feature_names=tuple(feature_set.feature_names),
            global_feature_names=tuple(feature_set.global_feature_names),
            feature_history=feature_history_array,
            global_features=global_features_array,
            close_history=close_history_array,
            data_age_hours=endpoint.data_age_hours,
            timestamps=np.asarray(timestamps),
        )
        snapshot.validate()
        with self._lock:
            self._cached = snapshot
            self._cached_digest = bundle.bundle_digest
            self._cached_until = now + self.cache_ttl_seconds
        return snapshot
