from __future__ import annotations

import numpy as np
import pytest

from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ResolvedRunConfig,
)
from trade_rl.evaluation.experiments.delta import FACTOR_RULES
from trade_rl.evaluation.runs.config import parse_candidate_run_config
from trade_rl.strategies.rl import ppo


def _raw_config() -> dict[str, object]:
    return {
        "signal_name": "1h__log_return_24bar",
        "feature_names": ["1h__log_return_24bar", "local"],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_cutoff": "2025-01-01T00:00:00.000000000",
        "evaluation_start": "2025-01-02T00:00:00.000000000",
        "evaluation_stop_exclusive": "2025-01-03T00:00:00.000000000",
        "rule_entry_threshold": 0.1,
        "rule_exit_threshold": 0.05,
        "forecast_entry_threshold": 0.1,
        "forecast_exit_threshold": 0.05,
        "ppo_total_timesteps": 128,
        "ppo_seed": 0,
        "gross_budget": 0.5,
        "initial_capital": 100_000.0,
    }


def _resolved_baseline() -> ResolvedRunConfig:
    return ResolvedRunConfig(
        signal_name="1h__log_return_24bar",
        signal_index=0,
        feature_names=("1h__log_return_24bar", "local"),
        feature_indices=(0, 1),
        fit_symbol_names=("BTCUSDT", "ETHUSDT"),
        fit_symbol_indices=(0, 1),
        fit_cutoff=str(np.datetime64("2025-01-01", "ns")),
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.05,
        forecast_entry_threshold=0.1,
        forecast_exit_threshold=0.05,
        ppo_total_timesteps=128,
        ppo_seed=0,
        evaluation_start=str(np.datetime64("2025-01-02", "ns")),
        evaluation_stop_exclusive=str(np.datetime64("2025-01-03", "ns")),
        gross_budget=0.5,
        initial_capital=100_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
        ppo_observation_schema="ppo_observation_v2",
        ppo_global_feature_names=(),
        schema_version="resolved_run_config_v2",
    )


def test_baseline_candidate_json_bytes_contract_remains_extension_free() -> None:
    config = parse_candidate_run_config(_raw_config())
    assert config.to_json_payload() == _raw_config()
    assert "ppo_global_context" not in config.to_json_payload()
    assert config.ppo_global_context is None


def test_candidate_config_accepts_only_the_sealed_global_context() -> None:
    candidate = _raw_config()
    candidate["ppo_global_context"] = "ppo_global_btc_regime_context"
    config = parse_candidate_run_config(candidate)
    assert config.ppo_global_context == ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT
    assert config.to_json_payload()["ppo_global_context"] == (
        "ppo_global_btc_regime_context"
    )

    bad = _raw_config()
    bad["ppo_global_context"] = "symbol_embedding"
    with pytest.raises(ValueError, match="unsupported PPO global context"):
        parse_candidate_run_config(bad)


def test_resolved_v3_binds_candidate_schema_without_dataset_global_features() -> None:
    baseline = _resolved_baseline()
    payload = baseline.to_payload()
    payload.update(
        {
            "schema_version": "resolved_run_config_v3",
            "ppo_observation_schema": "ppo_observation_v3_global_btc_regime",
            "ppo_global_context": "ppo_global_btc_regime_context",
        }
    )
    candidate = ResolvedRunConfig(
        signal_name=baseline.signal_name,
        signal_index=baseline.signal_index,
        feature_names=baseline.feature_names,
        feature_indices=baseline.feature_indices,
        fit_symbol_names=baseline.fit_symbol_names,
        fit_symbol_indices=baseline.fit_symbol_indices,
        fit_cutoff=baseline.fit_cutoff,
        rule_entry_threshold=baseline.rule_entry_threshold,
        rule_exit_threshold=baseline.rule_exit_threshold,
        forecast_entry_threshold=baseline.forecast_entry_threshold,
        forecast_exit_threshold=baseline.forecast_exit_threshold,
        ppo_total_timesteps=baseline.ppo_total_timesteps,
        ppo_seed=baseline.ppo_seed,
        evaluation_start=baseline.evaluation_start,
        evaluation_stop_exclusive=baseline.evaluation_stop_exclusive,
        gross_budget=baseline.gross_budget,
        initial_capital=baseline.initial_capital,
        execution_overlay=baseline.execution_overlay,
        ppo_observation_schema="ppo_observation_v3_global_btc_regime",
        ppo_global_feature_names=(),
        ppo_global_context="ppo_global_btc_regime_context",
        schema_version="resolved_run_config_v3",
    )
    assert candidate.to_payload() == payload


def test_feature_set_factor_admits_only_frozen_global_context_paths_as_extension() -> None:
    allowed = FACTOR_RULES[ControlledFactor.FEATURE_SET].allowed_paths
    assert ("feature_names",) in allowed
    assert ("feature_indices",) in allowed
    assert ("schema_version",) in allowed
    assert ("ppo_observation_schema",) in allowed
    assert ("ppo_global_context",) in allowed
    assert ("ppo_total_timesteps",) not in allowed
    assert ("ppo_seed",) not in allowed
    assert ("gross_budget",) not in allowed
