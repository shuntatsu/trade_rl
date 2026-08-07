from __future__ import annotations

import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.workflows.training_run import TrainingRunConfig


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
            "require_full_reward_preroll": False,
        },
        "execution": complete_execution_config(),
        "risk": {},
        "reward": {},
        "trend": {"fast_lookback": 1, "base_lookback": 2, "slow_lookback": 3},
        "action": {"alpha_enabled": False, "n_factors": 0},
    }


def test_training_config_rejects_disabled_full_reward_preroll() -> None:
    """Training refuses an explicitly disabled full-preroll contract."""

    with pytest.raises(ValueError, match="require_full_reward_preroll"):
        TrainingRunConfig.from_mapping(_mapping())
