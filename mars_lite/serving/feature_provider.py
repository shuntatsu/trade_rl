"""Cached CSV feature snapshots for the online serving plane."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from mars_lite.serving.runtime import FeatureSnapshot, ServingRuntime


class CsvFeatureProvider:
    def __init__(
        self,
        *,
        runtime: ServingRuntime,
        data_dir: str | Path,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        self.runtime = runtime
        self.data_dir = Path(data_dir)
        self.cache_ttl_seconds = cache_ttl_seconds
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

        from mars_lite.data.data_utils import TF_TO_MINUTES
        from mars_lite.data.sources import create_source
        from mars_lite.features.feature_pipeline import FeaturePipeline

        symbols = list(bundle.metadata["symbols"])
        run_config = dict(bundle.metadata.get("run_config") or {})
        base_timeframe = str(run_config.get("base_timeframe", "1h"))
        if base_timeframe not in TF_TO_MINUTES:
            raise ValueError(
                f"unsupported bundled base_timeframe: {base_timeframe!r}"
            )
        source = create_source("csv", symbols, data_dir=self.data_dir)
        feature_set = FeaturePipeline(
            symbols, base_timeframe=base_timeframe
        ).build(source)
        if tuple(feature_set.symbols) != tuple(symbols):
            raise ValueError(
                "feature provider symbol order does not match active bundle"
            )
        if not feature_set.n_bars:
            raise ValueError("feature provider produced no bars")

        rank_window = int(bundle.preprocessing.get("rank_window", 250))
        post_config = dict(bundle.metadata.get("post_processor") or {})
        vol_lookback = int(post_config.get("vol_lookback", 60))
        history_bars = max(rank_window, vol_lookback + 1, 2)
        start = max(0, feature_set.n_bars - history_bars)
        timestamps = feature_set.timestamps[start:]
        feature_history = feature_set.features[start:]
        close_history = feature_set.close[start:]
        last_timestamp = np.datetime64(timestamps[-1], "ns")
        now_ns = np.datetime64("now", "ns")
        age_hours = float((now_ns - last_timestamp) / np.timedelta64(1, "h"))
        if age_hours < 0:
            age_hours = 0.0
        identity_source = (
            f"{bundle.bundle_digest}:{str(last_timestamp)}:{feature_set.n_bars}"
        ).encode("utf-8")
        snapshot = FeatureSnapshot(
            snapshot_id=hashlib.sha256(identity_source).hexdigest(),
            symbols=tuple(feature_set.symbols),
            feature_names=tuple(feature_set.feature_names),
            global_feature_names=tuple(feature_set.global_feature_names),
            feature_history=np.asarray(feature_history, dtype=np.float64),
            global_features=np.asarray(
                feature_set.global_features[-1], dtype=np.float64
            ),
            close_history=np.asarray(close_history, dtype=np.float64),
            data_age_hours=age_hours,
        )
        snapshot.validate()
        with self._lock:
            self._cached = snapshot
            self._cached_digest = bundle.bundle_digest
            self._cached_until = now + self.cache_ttl_seconds
        return snapshot
