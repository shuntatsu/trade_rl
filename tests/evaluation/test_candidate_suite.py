from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import trade_rl.evaluation.candidate_suite as candidate_suite
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.strategy_comparison import StrategyComparison
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def market() -> MarketDataset:
    n = 60
    close = np.linspace(100.0, 130.0, n).reshape(n, 1)
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.linspace(-0.2, 0.2, n, dtype=np.float32).reshape(n, 1, 1),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 1), 1_000_000.0),
        funding_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_suite_builds_fixed_candidates_and_delegates_one_comparison(monkeypatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        candidate_suite,
        "fit_ridge_forecast",
        lambda *args, **kwargs: SimpleNamespace(feature_indices=(0,)),
    )
    monkeypatch.setattr(
        candidate_suite,
        "fit_lightgbm_forecast",
        lambda *args, **kwargs: SimpleNamespace(feature_indices=(0,)),
    )
    monkeypatch.setattr(
        candidate_suite,
        "fit_ppo_strategy",
        lambda *args, **kwargs: ConstantIntentStrategy(PositionIntent.FLAT),
    )
    monkeypatch.setattr(
        candidate_suite,
        "RidgeForecastStrategy",
        lambda *args, **kwargs: ConstantIntentStrategy(PositionIntent.FLAT),
    )
    monkeypatch.setattr(
        candidate_suite,
        "LightGBMForecastStrategy",
        lambda *args, **kwargs: ConstantIntentStrategy(PositionIntent.FLAT),
    )

    def fake_compare(dataset, strategies, **kwargs):
        calls["dataset"] = dataset
        calls["names"] = tuple(strategies)
        calls["kwargs"] = kwargs
        return StrategyComparison(entries=())

    monkeypatch.setattr(candidate_suite, "compare_strategies", fake_compare)
    dataset = market()
    config = candidate_suite.LeanCandidateConfig(
        signal_index=0,
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-03T06:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=7,
    )

    result = candidate_suite.run_lean_candidate_suite(
        dataset,
        config,
        start_index=54,
        stop_index=59,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    assert result.entries == ()
    assert calls["dataset"] is dataset
    assert calls["names"] == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
    )
    assert calls["kwargs"] == {
        "start_index": 54,
        "stop_index": 59,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "execution_cost": None,
        "risk": None,
    }
