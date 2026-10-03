from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.evaluation.runs.candidate_suite as candidate_suite
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import UniversalStrategyComparison
from trade_rl.evaluation.experiments.contracts import StudyPlan
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3


def test_universal_comparison_records_training_and_suppressed_counts() -> None:
    comparison = UniversalStrategyComparison(
        (),
        ppo_training_timesteps=2048,
        ppo_training_minimum_hold_suppressed_count=6,
    )

    assert comparison.ppo_training_timesteps == 2048
    assert comparison.ppo_training_minimum_hold_suppressed_count == 6
    assert comparison.shared_cash_ppo is None


def market(*, bar_hours: int = 1) -> MarketDataset:
    n = 60
    base = np.linspace(100.0, 130.0, n)
    close = np.stack((base, base * 1.5), axis=1)
    signal = np.linspace(-0.2, 0.2, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(bar_hours, "h"),
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
        periods_per_year=8_760 // bar_hours,
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
        return SimpleNamespace(
            policy=SimpleNamespace(num_timesteps=512),
            feature_indices=(0,),
            feature_names=("signal",),
            feature_normalizer=None,
            observation_schema=PPO_OBSERVATION_SCHEMA_V3,
            minimum_hold_bars=168,
            training_minimum_hold_suppressed_count=6,
        )

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

    def fake_ppo_wrapper(*args, **kwargs):
        calls["ppo_wrapper_kwargs"] = kwargs
        return ConstantIntentStrategy(PositionIntent.LONG)

    monkeypatch.setattr(candidate_suite, "PPOIntentStrategy", fake_ppo_wrapper)

    def fake_compare(dataset, factories, **kwargs):
        calls["comparison_dataset"] = dataset
        calls["names"] = tuple(factories)
        calls["fresh_instances"] = {
            name: (factory(), factory()) for name, factory in factories.items()
        }
        calls["kwargs"] = kwargs
        return UniversalStrategyComparison(by_symbol=())

    monkeypatch.setattr(
        candidate_suite,
        "compare_strategy_factories_by_symbol",
        fake_compare,
    )
    dataset = market()
    close = np.full(dataset.close.shape, 100.0, dtype=np.float64)
    open_prices = close.copy()
    open_prices[56, 0] = 20.0
    dataset = replace(
        dataset,
        open=open_prices,
        high=np.maximum(open_prices, close),
        low=np.minimum(open_prices, close),
        close=close,
        mark_price=close.copy(),
    )
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
        ppo_training_layout="interleaved",
        ppo_rollout_steps_per_env=512,
        fit_symbol_indices=(0,),
        ppo_minimum_hold_bars=168,
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=True,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        )
    )

    result = candidate_suite.run_lean_candidate_suite(
        dataset,
        config,
        start_index=54,
        stop_index=59,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        include_ppo_shared_cash_replay=True,
    )

    assert result.by_symbol == ()
    assert result.shared_cash_ppo is not None
    assert result.shared_cash_ppo.name == "ppo"
    assert result.shared_cash_ppo.replay.book.portfolio_value == 1_000.0
    assert result.shared_cash_ppo.metrics.total_return == 0.0
    assert result.shared_cash_ppo.metrics.max_drawdown == pytest.approx(0.2)
    assert result.shared_cash_ppo.replay.ledger_evidence is not None
    assert len(result.shared_cash_ppo.replay.returns.values) == 5
    assert result.ppo_training_timesteps == 512
    assert result.ppo_training_minimum_hold_suppressed_count == 6
    assert calls["ridge"] == 1
    assert calls["lightgbm"] == 1
    assert calls["ppo"] == 1
    assert calls["ridge_dataset"] is dataset
    assert calls["lightgbm_dataset"] is dataset
    assert calls["ppo_dataset"] is dataset
    assert calls["ridge_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["lightgbm_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["ppo_kwargs"]["fit_symbol_indices"] == (0,)
    assert calls["ppo_kwargs"]["training_layout"] == "interleaved"
    assert calls["ppo_kwargs"]["rollout_steps_per_env"] == 512
    assert calls["ppo_kwargs"]["minimum_hold_bars"] == 168
    assert calls["ppo_kwargs"]["observation_schema"] == PPO_OBSERVATION_SCHEMA_V3
    assert calls["ppo_kwargs"]["settle_terminal_position"] is True
    assert calls["ppo_kwargs"]["risk_config"] == risk.config
    assert calls["ppo_wrapper_kwargs"]["feature_names"] == ("signal",)
    assert calls["ppo_wrapper_kwargs"]["observation_schema"] == (
        PPO_OBSERVATION_SCHEMA_V3
    )
    assert calls["ppo_wrapper_kwargs"]["minimum_hold_bars"] == 168
    assert calls["ppo_wrapper_kwargs"]["training_minimum_hold_suppressed_count"] == 6
    assert calls["comparison_dataset"] is dataset
    assert calls["names"] == StudyPlan.STRATEGY_NAMES
    for first, second in calls["fresh_instances"].values():
        assert first is not second
    assert calls["kwargs"] == {
        "start_index": 54,
        "stop_index": 59,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "execution_cost": None,
        "risk": risk,
        "settle_terminal_position": True,
    }


def test_age_aware_suite_fails_closed_without_explicit_twenty_percent_risk() -> None:
    config = candidate_suite.LeanCandidateConfig(
        signal_index=0,
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-03T06:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        fit_symbol_indices=(0,),
        ppo_minimum_hold_bars=168,
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=True,
    )

    with pytest.raises(ValueError, match="explicit.*risk"):
        candidate_suite.run_lean_candidate_suite(
            market(),
            config,
            start_index=54,
            stop_index=59,
            gross_budget=0.5,
            risk=None,
        )


def test_age_aware_suite_rejects_non_hourly_data_before_fitting(monkeypatch) -> None:
    config = candidate_suite.LeanCandidateConfig(
        signal_index=0,
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-03T06:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        fit_symbol_indices=(0,),
        ppo_minimum_hold_bars=168,
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=True,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.1,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        )
    )

    def fail_if_fitted(*args, **kwargs):
        raise AssertionError("model fitting must not run for non-hourly data")

    monkeypatch.setattr(candidate_suite, "fit_ridge_forecast", fail_if_fitted)
    monkeypatch.setattr(candidate_suite, "fit_lightgbm_forecast", fail_if_fitted)
    monkeypatch.setattr(candidate_suite, "fit_ppo_strategy", fail_if_fitted)

    with pytest.raises(ValueError, match="exactly regular one-hour bars"):
        candidate_suite.run_lean_candidate_suite(
            market(bar_hours=4),
            config,
            start_index=54,
            stop_index=59,
            gross_budget=0.5,
            risk=risk,
        )


def test_age_aware_suite_rejects_missing_terminal_settlement() -> None:
    config = candidate_suite.LeanCandidateConfig(
        signal_index=0,
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-03T06:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        fit_symbol_indices=(0,),
        ppo_minimum_hold_bars=168,
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=False,
    )

    with pytest.raises(ValueError, match="terminal settlement"):
        candidate_suite.run_lean_candidate_suite(
            market(),
            config,
            start_index=54,
            stop_index=59,
            gross_budget=0.5,
            risk=PreTradeRisk(
                PreTradeRiskConfig(
                    max_gross=0.5,
                    max_abs_weight=0.1,
                    drawdown_start=0.1,
                    drawdown_stop=0.2,
                )
            ),
        )
