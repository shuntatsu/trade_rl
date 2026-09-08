from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)


def market(*, terminal_multiplier: float = 1.0) -> MarketDataset:
    n_bars = 12
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(0.01 * phase)).reshape(-1, 1)
    close[8:, 0] *= terminal_multiplier
    features = np.stack((phase, phase**2), axis=-1).reshape(n_bars, 1, 2)
    return MarketDataset(
        dataset_id="7" * 64,
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


def test_fit_cutoff_excludes_labels_ending_at_or_after_cutoff() -> None:
    cutoff = np.datetime64("2026-01-01T08:00:00", "ns")
    baseline = fit_ridge_forecast(
        market(),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
        alpha=1.0,
    )
    mutated_future = fit_ridge_forecast(
        market(terminal_multiplier=5.0),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
        alpha=1.0,
    )

    assert baseline.n_samples == 6
    assert mutated_future.n_samples == 6
    np.testing.assert_allclose(baseline.coefficients, mutated_future.coefficients)
    np.testing.assert_allclose(baseline.feature_mean, mutated_future.feature_mean)
    np.testing.assert_allclose(baseline.feature_scale, mutated_future.feature_scale)
    assert baseline.intercept == mutated_future.intercept


def test_fitted_scaler_and_model_arrays_are_read_only() -> None:
    model = fit_ridge_forecast(
        market(),
        feature_indices=(0, 1),
        fit_cutoff=np.datetime64("2026-01-01T10:00:00", "ns"),
        horizon_hours=2,
        alpha=1.0,
    )

    assert model.feature_mean.flags.writeable is False
    assert model.feature_scale.flags.writeable is False
    assert model.coefficients.flags.writeable is False


def observation(value: float, *, available: bool = True) -> StrategyObservation:
    return StrategyObservation(
        index=10,
        timestamp=np.datetime64("2026-01-01T10:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([value, 0.0]),
        feature_available=np.asarray([available, True]),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_ridge_strategy_uses_shared_forecast_controller_and_fails_closed() -> None:
    mean = np.asarray([0.0])
    scale = np.asarray([1.0])
    coefficients = np.asarray([1.0])
    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=0.0,
        horizon_hours=24,
        alpha=1.0,
        n_samples=100,
        fit_cutoff=np.datetime64("2025-12-31T00:00:00", "ns"),
    )
    strategy = RidgeForecastStrategy(
        model,
        entry_threshold=0.10,
        exit_threshold=0.02,
    )

    assert strategy.decide(observation(0.20)) is PositionIntent.LONG
    assert strategy.decide(observation(-0.20)) is PositionIntent.SHORT
    assert strategy.decide(observation(0.20, available=False)) is PositionIntent.FLAT
