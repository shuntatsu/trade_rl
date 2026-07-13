from __future__ import annotations

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import trend_following_strategy


def _feature_set(n_bars: int = 100) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    up = np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    down = up[::-1]
    close = np.column_stack([up, down])
    return FeatureSet(
        symbols=["UP", "DOWN"],
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["global"],
    )


def test_trend_rebalance_phase_is_stable_under_contextual_slicing() -> None:
    full = _feature_set()
    contextual = full.slice(12, 80)
    current = np.array([0.7, -0.3], dtype=np.float64)

    full_weights = trend_following_strategy(
        full,
        48,
        current,
        lookback=24,
        rebalance_every=24,
    )
    contextual_weights = trend_following_strategy(
        contextual,
        36,
        current,
        lookback=24,
        rebalance_every=24,
    )

    np.testing.assert_allclose(contextual_weights, full_weights)
