from __future__ import annotations

from copy import deepcopy

import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.workflows.training_run import TrainingRunConfig


def _pure_growth_mapping() -> dict[str, object]:
    return {
        "schema_version": "training_run_config_v4",
        "training": {
            "algorithm": "ppo",
            "timesteps": 8,
            "gamma": 1.0,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
            "policy_actor_head": None,
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
            "liquidate_on_end": False,
            "require_full_reward_preroll": True,
        },
        "execution": complete_execution_config(),
        "risk": {},
        "reward": {
            "scale": 100.0,
            "absolute_growth_weight": 1.0,
            "excess_growth_weight": 0.0,
            "incremental_drawdown_weight": 0.0,
            "baseline_underperformance_weight": 0.0,
            "projection_penalty_weight": 0.0,
            "terminal_equity_weight": 0.0,
            "margin_deficit_weight": 0.0,
        },
        "trend": {
            "fast_lookback": 1,
            "base_lookback": 2,
            "slow_lookback": 3,
        },
        "action": {
            "mode": "target_weight",
            "alpha_enabled": False,
            "risk_tilt_enabled": False,
            "n_factors": 0,
            "target_weight_count": 1,
        },
    }


def test_training_config_accepts_target_weight_pure_growth() -> None:
    config = TrainingRunConfig.from_mapping(_pure_growth_mapping())

    assert config.reward.is_pure_net_log_growth() is True
    assert config.training.gamma == pytest.approx(1.0)
    assert config.action.mode.value == "target_weight"


@pytest.mark.parametrize(
    "field",
    [
        "excess_growth_weight",
        "incremental_drawdown_weight",
        "baseline_underperformance_weight",
        "projection_penalty_weight",
    ],
)
def test_terminal_and_margin_disabled_contract_rejects_objective_mixing(
    field: str,
) -> None:
    raw = deepcopy(_pure_growth_mapping())
    reward = dict(raw["reward"])  # type: ignore[arg-type]
    reward[field] = 0.1
    raw["reward"] = reward

    with pytest.raises(ValueError, match="pure_net_log_growth"):
        TrainingRunConfig.from_mapping(raw)


def test_terminal_and_margin_disabled_contract_requires_unit_growth_weight() -> None:
    raw = deepcopy(_pure_growth_mapping())
    reward = dict(raw["reward"])  # type: ignore[arg-type]
    reward["absolute_growth_weight"] = 0.5
    raw["reward"] = reward

    with pytest.raises(ValueError, match="pure_net_log_growth"):
        TrainingRunConfig.from_mapping(raw)


def test_legacy_shaping_remains_backward_compatible() -> None:
    raw = _pure_growth_mapping()
    reward = dict(raw["reward"])  # type: ignore[arg-type]
    reward.pop("terminal_equity_weight")
    reward.pop("margin_deficit_weight")
    reward["baseline_underperformance_weight"] = 0.1
    raw["reward"] = reward

    config = TrainingRunConfig.from_mapping(raw)

    assert config.reward.is_pure_net_log_growth() is False
    assert config.reward.terminal_equity_weight == pytest.approx(1.0)
    assert config.reward.margin_deficit_weight == pytest.approx(1.0)
