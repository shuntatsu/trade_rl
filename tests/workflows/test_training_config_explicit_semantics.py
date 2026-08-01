from __future__ import annotations

import pytest

from trade_rl.workflows.training_run import TrainingRunConfig


def _mapping() -> dict[str, object]:
    return {
        "schema_version": "training_run_config_v3",
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


def _complete_execution() -> dict[str, object]:
    return {
        "fee_rate": 0.0005,
        "maker_fee_rate": 0.0,
        "taker_fee_rate": 0.0,
        "spread_rate": 0.0002,
        "impact_rate": 0.0001,
        "multiplier": 1.0,
        "max_participation_rate": 0.05,
        "slippage_std": 0.0,
        "tail_slippage_probability": 0.0,
        "tail_slippage_multiplier": 5.0,
        "random_seed": 0,
        "minimum_notional": 0.0,
        "lot_size": 0.0,
        "tick_size": 0.0,
        "allow_short": True,
        "borrow_rate_multiplier": 1.0,
        "max_leverage": 1.0,
        "maintenance_margin_rate": 0.25,
        "collateral_haircut": 1.0,
        "margin_mode": "cross",
        "order_latency_bars": 0,
        "order_type": "market",
        "limit_offset_rate": 0.0005,
        "path_mode": "conservative",
        "processing_bar_volume_capacity": True,
        "partial_fill_carry": True,
        "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
    }


def test_training_config_rejects_omitted_execution_section() -> None:
    with pytest.raises(ValueError, match="execution"):
        TrainingRunConfig.from_mapping(_mapping())


def test_training_config_rejects_omitted_execution_field() -> None:
    raw = _mapping()
    execution = _complete_execution()
    del execution["maker_fee_rate"]
    raw["execution"] = execution

    with pytest.raises(ValueError, match="maker_fee_rate"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_reward_preroll_contract() -> None:
    raw = _mapping()
    raw["execution"] = _complete_execution()

    with pytest.raises(ValueError, match="require_full_reward_preroll"):
        TrainingRunConfig.from_mapping(raw)
