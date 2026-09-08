from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.supervised import build_causal_forecast_training_set


def market(*, future_multiplier: float = 1.0) -> MarketDataset:
    n_bars = 12
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(0.01 * phase)).reshape(-1, 1)
    close[8:, 0] *= future_multiplier
    features = np.stack((phase, phase**2), axis=-1).reshape(n_bars, 1, 2)
    return MarketDataset(
        dataset_id="6" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features.astype(np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 2), dtype=np.bool_),
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def pooled_market() -> MarketDataset:
    n_bars = 12
    phase = np.arange(n_bars, dtype=np.float64)
    close = np.stack(
        (
            100.0 * np.exp(0.01 * phase),
            200.0 * np.exp(-0.005 * phase),
        ),
        axis=1,
    )
    features = np.empty((n_bars, 2, 2), dtype=np.float32)
    features[:, 0, 0] = phase
    features[:, 0, 1] = phase**2
    features[:, 1, 0] = 100.0 + phase
    features[:, 1, 1] = 10_000.0 + phase**2
    available = np.ones_like(features, dtype=np.bool_)
    available[:3, 1, :] = False
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=available,
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_training_set_is_strictly_cutoff_causal_and_future_invariant() -> None:
    cutoff = np.datetime64("2026-01-01T08:00:00", "ns")
    baseline = build_causal_forecast_training_set(
        market(),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
    )
    mutated = build_causal_forecast_training_set(
        market(future_multiplier=7.0),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
    )

    assert baseline.n_samples == 6
    assert baseline.label_end_times.max() < cutoff
    np.testing.assert_array_equal(baseline.features, mutated.features)
    np.testing.assert_array_equal(baseline.labels, mutated.labels)
    np.testing.assert_array_equal(baseline.sample_weights, np.ones(6))
    assert baseline.features.flags.writeable is False
    assert baseline.labels.flags.writeable is False
    assert baseline.label_end_times.flags.writeable is False
    assert baseline.sample_weights.flags.writeable is False


def test_multi_symbol_rows_use_no_symbol_feature_and_equal_symbol_weight() -> None:
    training = build_causal_forecast_training_set(
        pooled_market(),
        feature_indices=(0, 1),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    assert training.features.shape == (9, 2)
    assert training.n_samples == 9
    np.testing.assert_allclose(training.sample_weights[:6], np.full(6, 0.75))
    np.testing.assert_allclose(training.sample_weights[6:], np.full(3, 1.5))
    assert training.sample_weights[:6].sum() == 4.5
    assert training.sample_weights[6:].sum() == 4.5
    assert training.sample_weights.sum() == 9.0
    assert training.label_end_times.max() < training.fit_cutoff
