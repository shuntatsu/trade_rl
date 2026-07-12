import numpy as np

from mars_lite.eval.context_window import with_history_context
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _feature_set(n_bars: int = 260) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = np.exp(np.arange(n_bars, dtype=np.float64) * 0.001)
    close = np.column_stack([base, base[::-1]])
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


def test_context_window_preserves_absolute_trend_at_scored_start() -> None:
    fs = _feature_set()
    window = with_history_context(fs, start=180, end=240, history_bars=120)
    family = TrendFamily(
        TrendFamilyConfig(
            fast_lookback=24,
            base_lookback=48,
            slow_lookback=96,
            rebalance_every=24,
        )
    )

    full = family.targets(fs, 180)
    contextual = family.targets(window.feature_set, window.start_idx)

    np.testing.assert_allclose(contextual.fast, full.fast, atol=1e-12)
    np.testing.assert_allclose(contextual.base, full.base, atol=1e-12)
    np.testing.assert_allclose(contextual.slow, full.slow, atol=1e-12)
    assert window.scored_bars == 60


def test_context_bars_are_not_counted_as_scored_bars() -> None:
    fs = _feature_set()
    window = with_history_context(fs, start=30, end=90, history_bars=120)

    assert window.start_idx == 30
    assert window.feature_set.n_bars == 90
    assert window.scored_bars == 60
