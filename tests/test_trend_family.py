import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _feature_set(n_bars: int = 220) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
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
        global_feature_names=["dummy_global"],
    )


def test_targets_do_not_depend_on_portfolio_state() -> None:
    fs = _feature_set()
    family = TrendFamily(TrendFamilyConfig(rebalance_every=24))

    first = family.targets(fs, 180)
    second = family.targets(fs, 180)

    np.testing.assert_allclose(first.base, second.base)
    assert np.abs(first.base).sum() <= 1.0 + 1e-12


def test_same_timestamp_matches_across_slices_with_enough_history() -> None:
    fs = _feature_set()
    family = TrendFamily(TrendFamilyConfig(rebalance_every=24))
    full_index = 192
    sliced = fs.slice(48, fs.n_bars)
    sliced_index = full_index - 48

    full = family.targets(fs, full_index)
    partial = family.targets(sliced, sliced_index)

    np.testing.assert_allclose(full.fast, partial.fast, atol=1e-12)
    np.testing.assert_allclose(full.base, partial.base, atol=1e-12)
    np.testing.assert_allclose(full.slow, partial.slow, atol=1e-12)


def test_targets_are_held_between_absolute_rebalance_slots() -> None:
    fs = _feature_set()
    family = TrendFamily(TrendFamilyConfig(rebalance_every=24))

    at_slot = family.targets(fs, 192)
    between_slots = family.targets(fs, 200)

    np.testing.assert_allclose(at_slot.base, between_slots.base, atol=1e-12)


def test_fast_base_slow_use_distinct_lookbacks() -> None:
    fs = _feature_set()
    family = TrendFamily(
        TrendFamilyConfig(
            fast_lookback=24,
            base_lookback=48,
            slow_lookback=96,
            rebalance_every=24,
        )
    )

    targets = family.targets(fs, 192)

    assert not np.allclose(targets.fast, targets.slow)
    for weights in (targets.fast, targets.base, targets.slow):
        assert np.all(np.isfinite(weights))
        assert np.abs(weights).sum() <= 1.0 + 1e-12
