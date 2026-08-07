from __future__ import annotations

from dataclasses import fields

import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.training_run import TrainingRunConfig

# Literal by design: changing the public execution fields requires a new schema version.
_V4_EXECUTION_FIELDS = frozenset(
    {
        "allow_short",
        "borrow_rate_multiplier",
        "collateral_haircut",
        "fee_rate",
        "impact_rate",
        "limit_offset_rate",
        "lot_size",
        "maintenance_margin_rate",
        "maker_fee_rate",
        "margin_mode",
        "max_leverage",
        "max_participation_rate",
        "minimum_notional",
        "multiplier",
        "order_latency_bars",
        "order_type",
        "partial_fill_carry",
        "path_mode",
        "processing_bar_volume_capacity",
        "random_seed",
        "slippage_std",
        "spread_rate",
        "tail_slippage_multiplier",
        "tail_slippage_probability",
        "taker_fee_rate",
        "tick_size",
        "trigger_volume_fractions",
    }
)


def _mapping() -> dict[str, object]:
    return {
        "schema_version": "training_run_config_v4",
        "training": {
            "timesteps": 8,
            "gamma": 0.99,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
            "policy_actor_head": "standard_continuous_v1",
            "hierarchical_gate_temperature": 1.0,
            "behavior_cloning_gate_loss_weight": 1.0,
            "behavior_cloning_target_loss_weight": 1.0,
            "behavior_cloning_composed_loss_weight": 1.0,
            "behavior_cloning_gate_change_threshold": 0.05,
            "behavior_cloning_max_positive_class_weight": 20.0,
            "behavior_cloning_min_gate_precision": 0.0,
            "behavior_cloning_min_gate_recall": 0.0,
            "behavior_cloning_max_active_target_rmse": 1.0,
            "behavior_cloning_min_activity_ratio": 0.0,
            "behavior_cloning_max_activity_ratio": 1.0,
            "behavior_cloning_min_causal_holdout_trades": 0,
            "behavior_cloning_max_causal_holdout_regret": 0.0,
            "behavior_cloning_causal_holdout_bootstrap_resamples": 2_000,
            "behavior_cloning_causal_holdout_confidence_level": 0.95,
        },
        "environment": {
            "episode_bars": 4,
            "decision_every": 1,
            "initial_capital": 1_000.0,
        },
        "risk": {},
        "reward": {},
        "trend": {"fast_lookback": 1, "base_lookback": 2, "slow_lookback": 3},
        "action": {"alpha_enabled": False, "n_factors": 0},
    }


def test_v4_execution_schema_matches_public_execution_fields() -> None:
    dataclass_field_names = {
        item.name for item in fields(ExecutionCostConfig) if item.init
    }
    assert dataclass_field_names == _V4_EXECUTION_FIELDS
    assert set(complete_execution_config()) == _V4_EXECUTION_FIELDS


def test_training_config_rejects_v3_with_migration_message() -> None:
    raw = _mapping()
    raw["schema_version"] = "training_run_config_v3"
    raw["execution"] = complete_execution_config()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="training_run_config_v4"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_execution_section() -> None:
    raw = _mapping()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="execution"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_execution_field() -> None:
    raw = _mapping()
    execution = complete_execution_config()
    del execution["maker_fee_rate"]
    raw["execution"] = execution
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="maker_fee_rate"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_reward_preroll_contract() -> None:
    raw = _mapping()
    raw["execution"] = complete_execution_config()

    with pytest.raises(ValueError, match="require_full_reward_preroll"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_accepts_complete_explicit_semantics() -> None:
    raw = _mapping()
    raw["execution"] = complete_execution_config()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    config = TrainingRunConfig.from_mapping(raw)

    assert config.schema_version == "training_run_config_v4"
    assert config.environment.require_full_reward_preroll is True
    assert config.environment.execution_cost == ExecutionCostConfig()
