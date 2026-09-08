from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import trade_rl.evaluation.runs.candidate_suite as candidate_suite
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import UniversalStrategyComparison
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def market() -> MarketDataset:
    n = 60
    base = np.linspace(100.0, 130.0, n)
    close = np.stack((base, base * 1.5), axis=1)
    signal = np.linspace(-0.2, 0.2, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_suite_fits_one_universal_candidate_set_and_compares_every_symbol(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {"ridge": 0, "lightgbm": 0, "ppo": 0}

    def fake_ridge(*args, **kwargs):
        calls["ridge"] = int(calls["ridge"]) + 1
        calls["ridge_dataset"] = args[0]
        calls["ridge_kwargs"] = kwargs
        return SimpleNamespace(feature_indices=(0,))

    def fake_lightgbm(*args, **kwargs):
        calls["lightgbm"] = int(calls["lightgbm"]) + 1
        calls["lightgbm_dataset"] = args[0]
        calls["lightgbm_kwargs"] = kwargs
        return SimpleNamespace(feature_indices=(0,))

    def fake_ppo(*args, **kwargs):
        calls["ppo"] = int(calls["ppo"]) + 1
        calls["ppo_dataset"] = args[0]
        calls["ppo_kwargs"] = kwargs
        return ConstantIntentStrategy(PositionIntent.FLAT)

    monkeypatch.setattr(candidate_suite, "fit_ridge_forecast", fake_ridge)
    monkeypatch.setattr(candidate_suite, "fit_lightgbm_forecast", fake_lightgbm)
    monkeypatch.setattr(candidate_suite, "fit_ppo_strategy", fake_ppo)
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
        calls["comparison_dataset"] = dataset
        calls["names"] = tuple(strategies)
        calls["kwargs"] = kwargs
        return UniversalStrategyComparison(by_symbol=())

    monkeypatch.setattr(candidate_suite, "compare_strategies_by_symbol", fake_compare)
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
        fit_symbol_indices=(0,),
    )

    result = candidate_suite.run_lean_candidate_suite(
        dataset,
        config,
        start_index=54,
        stop_index=59,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    assert result.by_symbol == ()
    assert calls["ridge"] == 1
    assert calls["lightgbm"] == 1
    assert calls["ppo"] == 1
    assert calls["ridge_dataset"] is dataset
    assert calls["lightgbm_dataset"] is dataset
    assert calls["ppo_dataset"] is dataset
    assert calls["ridge_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["lightgbm_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["ppo_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["comparison_dataset"] is dataset
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
