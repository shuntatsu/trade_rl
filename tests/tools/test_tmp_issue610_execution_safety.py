from __future__ import annotations

from dataclasses import replace

import pytest

from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    candidate_run_config_from_resolved,
    execute_candidate_after_precompute_gate,
)
from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def _baseline() -> ResolvedRunConfig:
    return ResolvedRunConfig(
        signal_name="1h__log_return_24bar",
        signal_index=0,
        feature_names=("1h__log_return_24bar", "1h__funding_rate"),
        feature_indices=(0, 1),
        fit_symbol_names=_SYMBOLS,
        fit_symbol_indices=(0, 1, 2, 3, 4),
        fit_cutoff="2021-01-01T00:00:00.000000000",
        rule_entry_threshold=0.01,
        rule_exit_threshold=0.001,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.001,
        ppo_total_timesteps=100_000,
        ppo_seed=0,
        evaluation_start="2021-01-01T00:00:00.000000000",
        evaluation_stop_exclusive="2023-01-01T00:00:00.000000000",
        gross_budget=1.0,
        initial_capital=10_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA,
        ppo_global_feature_names=PPO_GLOBAL_FEATURE_NAMES,
        schema_version="resolved_run_config_v2",
    )


def _candidate() -> ResolvedRunConfig:
    return replace(
        _baseline(),
        ppo_observation_schema=PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
        schema_version="resolved_run_config_v3",
    )


def test_candidate_run_config_projection_preserves_executable_semantics() -> None:
    resolved = _candidate()
    projected = candidate_run_config_from_resolved(resolved)

    assert projected.signal_name == resolved.signal_name
    assert projected.feature_names == resolved.feature_names
    assert projected.fit_symbol_names == resolved.fit_symbol_names
    assert str(projected.fit_cutoff) == resolved.fit_cutoff
    assert str(projected.evaluation_start) == resolved.evaluation_start
    assert (
        str(projected.evaluation_stop_exclusive) == resolved.evaluation_stop_exclusive
    )
    assert projected.rule_entry_threshold == resolved.rule_entry_threshold
    assert projected.rule_exit_threshold == resolved.rule_exit_threshold
    assert projected.forecast_entry_threshold == resolved.forecast_entry_threshold
    assert projected.forecast_exit_threshold == resolved.forecast_exit_threshold
    assert projected.ppo_total_timesteps == resolved.ppo_total_timesteps
    assert projected.ppo_seed == resolved.ppo_seed
    assert projected.gross_budget == resolved.gross_budget
    assert projected.initial_capital == resolved.initial_capital
    assert projected.ppo_global_context == PPO_GLOBAL_BTC_REGIME_CONTEXT


def test_precompute_failure_never_calls_candidate_executor() -> None:
    calls = 0

    def executor() -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(RuntimeError, match="precompute gate failed"):
        execute_candidate_after_precompute_gate(
            precompute_violations=("authority mismatch",),
            candidate_executor=executor,
        )

    assert calls == 0


def test_candidate_executor_is_called_exactly_once_after_green_precompute_gate() -> (
    None
):
    calls = 0
    sentinel = object()

    def executor() -> object:
        nonlocal calls
        calls += 1
        return sentinel

    actual = execute_candidate_after_precompute_gate(
        precompute_violations=(),
        candidate_executor=executor,
    )

    assert actual is sentinel
    assert calls == 1
