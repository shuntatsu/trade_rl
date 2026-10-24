from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.strategies.forecasts.lightgbm as lightgbm_module
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.lightgbm import (
    LightGBMForecastModel,
    LightGBMForecastStrategy,
    fit_lightgbm_forecast,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class FakeRegressor:
    last: FakeRegressor | None = None

    def __init__(self, **params: object) -> None:
        self.params = params
        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.sample_weights: np.ndarray | None = None
        FakeRegressor.last = self

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> FakeRegressor:
        self.features = np.asarray(features).copy()
        self.labels = np.asarray(labels).copy()
        self.sample_weights = (
            None if sample_weight is None else np.asarray(sample_weight).copy()
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(features).shape[0], 0.20, dtype=np.float64)


def market() -> MarketDataset:
    n_bars = 12
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(0.01 * phase)).reshape(-1, 1)
    features = np.stack((phase, phase**2), axis=-1).reshape(n_bars, 1, 2)
    return MarketDataset(
        dataset_id="5" * 64,
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


def observation(
    *,
    available: bool = True,
    current_intent: PositionIntent = PositionIntent.FLAT,
    current_weight: float = 0.0,
    current_position_quantity: float | None = None,
    intent_gross_budget: float | None = None,
) -> StrategyObservation:
    return StrategyObservation(
        index=10,
        timestamp=np.datetime64("2026-01-01T10:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([10.0, 100.0]),
        feature_available=np.asarray([available, True]),
        feature_staleness=np.asarray(
            [0.0 if available else 1.0, 0.0], dtype=np.float32
        ),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=current_intent,
        current_weight=current_weight,
        current_position_quantity=current_position_quantity,
        intent_gross_budget=intent_gross_budget,
    )


def test_fit_uses_shared_causal_rows_and_one_shallow_configuration(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(LGBMRegressor=FakeRegressor),
    )
    model = fit_lightgbm_forecast(
        market(),
        feature_indices=(0, 1),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
        random_state=7,
    )

    regressor = FakeRegressor.last
    assert regressor is not None
    assert regressor.features is not None
    assert regressor.labels is not None
    assert regressor.sample_weights is not None
    assert regressor.features.shape == (6, 2)
    assert regressor.labels.shape == (6,)
    np.testing.assert_array_equal(regressor.sample_weights, np.ones(6))
    assert regressor.params["n_estimators"] == 64
    assert regressor.params["max_depth"] == 3
    assert regressor.params["num_leaves"] == 7
    assert regressor.params["random_state"] == 7
    assert model.n_samples == 6


def test_fit_rejects_unrepresentable_cutoff_before_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_estimator_is_created() -> object:
        pytest.fail("LightGBM estimator was created before fit_cutoff validation")

    monkeypatch.setattr(
        lightgbm_module,
        "_lightgbm_regressor_factory",
        fail_if_estimator_is_created,
    )

    with pytest.raises(ValueError, match=r"fit_cutoff.*datetime64\[ns\]"):
        fit_lightgbm_forecast(
            market(),
            feature_indices=(0, 1),
            fit_cutoff=np.datetime64("1677-01-01T00:00:00", "s"),
            horizon_hours=1,
        )


def test_lightgbm_strategy_uses_shared_forecast_controller() -> None:
    model = LightGBMForecastModel(
        feature_indices=(0, 1),
        predictor=FakeRegressor(),
        horizon_hours=24,
        n_samples=100,
        fit_cutoff=np.datetime64("2025-12-31T00:00:00", "ns"),
    )
    strategy = LightGBMForecastStrategy(
        model,
        entry_threshold=0.10,
        exit_threshold=0.02,
    )

    assert strategy.decide(observation()) is PositionIntent.LONG
    assert strategy.decide(observation(available=False)) is PositionIntent.FLAT


def test_cost_aware_lightgbm_uses_filled_position_only_when_budget_is_present() -> None:
    weak_positive_forecast = float(np.log1p(0.005))

    class WeakRegressor:
        def __init__(self, forecast: float) -> None:
            self.forecast = forecast

        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(np.asarray(features).shape[0], self.forecast)

    model = LightGBMForecastModel(
        feature_indices=(0,),
        predictor=WeakRegressor(weak_positive_forecast),
        horizon_hours=24,
        n_samples=100,
        fit_cutoff=np.datetime64("2025-12-31T00:00:00", "ns"),
    )
    strategy = LightGBMForecastStrategy(
        model,
        entry_threshold=0.10,
        exit_threshold=0.02,
        one_way_switch_cost=0.01,
    )

    assert (
        strategy.decide(
            observation(
                current_intent=PositionIntent.LONG,
                current_weight=0.0,
                current_position_quantity=0.0,
                intent_gross_budget=0.5,
            )
        )
        is PositionIntent.FLAT
    )
    assert (
        strategy.decide(
            observation(
                current_intent=PositionIntent.LONG,
                current_weight=0.0,
                current_position_quantity=0.0,
            )
        )
        is PositionIntent.LONG
    )

    baseline = LightGBMForecastStrategy(
        LightGBMForecastModel(
            feature_indices=(0,),
            predictor=WeakRegressor(0.05),
            horizon_hours=24,
            n_samples=100,
            fit_cutoff=np.datetime64("2025-12-31T00:00:00", "ns"),
        ),
        entry_threshold=0.10,
        exit_threshold=0.02,
    )
    assert (
        baseline.decide(
            observation(
                current_intent=PositionIntent.LONG,
                current_weight=0.0,
                current_position_quantity=0.0,
                intent_gross_budget=0.5,
            )
        )
        is PositionIntent.LONG
    )
