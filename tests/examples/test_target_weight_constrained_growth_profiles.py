from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.rl.algorithm_configs import LagrangianPPOConfig, build_algorithm_config
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"

PPO = "training-target-weight-growth-ppo.json"
LAGRANGIAN = "training-target-weight-constrained-growth.json"
DISCOUNTED = "training-target-weight-constrained-growth-discounted.json"
WALK_FORWARD = "walk-forward-target-weight-constrained-growth.json"

EXPECTED_BUDGETS = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.03)
EXPECTED_DUAL_LEARNING_RATES = (0.001, 0.01, 0.001, 0.01, 0.001, 0.001, 0.001)
EXPECTED_MINIMUM_SUPPORT = (1, 20, 1, 20, 1, 1, 1)
EXPECTED_NAMES = (
    "target-weight-growth-gamma-one-ppo",
    "target-weight-constrained-growth-gamma-one",
    "target-weight-constrained-growth-discounted-168h",
)


def _load(name: str) -> TrainingRunConfig:
    return TrainingRunConfig.from_json(EXAMPLE_ROOT / name)


def _common_contract(config: TrainingRunConfig) -> None:
    assert config.action.mode.value == "target_weight"
    assert config.action.target_weight_count == 3
    assert config.action.alpha_enabled is False
    assert config.action.risk_tilt_enabled is False
    assert config.reward.is_pure_net_log_growth() is True
    assert config.environment.episode_hours == pytest.approx(720.0)
    assert config.environment.finite_horizon_observation is False
    assert config.environment.liquidate_on_end is False
    assert config.risk.max_abs_weight == pytest.approx(0.45)
    assert config.risk.max_gross == pytest.approx(1.0)
    assert config.risk.drawdown_start == pytest.approx(0.10)
    assert config.risk.drawdown_stop == pytest.approx(0.20)

    training = config.training
    assert training.seeds == (0, 1, 2)
    assert training.timesteps == 524_288
    assert training.n_steps == 256
    assert training.n_envs == 4
    assert training.batch_size == 256
    assert training.n_epochs == 10
    assert training.behavior_cloning_epochs == 15
    assert training.behavior_cloning_teacher == "oracle"
    assert training.behavior_cloning_validation_fraction == pytest.approx(0.1)
    assert training.policy_net_arch == (384, 256, 128)
    assert training.value_net_arch == (512, 384, 256)
    assert training.observation_encoder.value == "hierarchical_sequence_v2"
    assert training.sequence_d_model == 336
    assert training.sequence_compile is False
    assert training.vector_environment_mode == "subprocess"


def _without_discount(payload: dict[str, object]) -> dict[str, object]:
    resolved = deepcopy(payload)
    training = resolved["training"]
    assert isinstance(training, dict)
    training.pop("gamma", None)
    training.pop("discount_half_life_hours", None)
    return resolved


def _without_algorithm_specific_training(
    payload: dict[str, object],
) -> dict[str, object]:
    resolved = deepcopy(payload)
    training = resolved["training"]
    assert isinstance(training, dict)
    for name in tuple(training):
        if name == "algorithm" or name.startswith(("cost_", "lagrangian_")):
            training.pop(name)
    resolved.pop("cost_critic", None)
    resolved.pop("lagrangian", None)
    return resolved


def test_target_weight_growth_ppo_is_gamma_one_control() -> None:
    config = _load(PPO)

    _common_contract(config)
    assert config.training.algorithm == "ppo"
    assert config.training.gamma == pytest.approx(1.0)
    assert config.training.discount_half_life_hours is None


def test_target_weight_lagrangian_uses_same_growth_recipe_and_all_costs() -> None:
    ppo = _load(PPO)
    constrained = _load(LAGRANGIAN)

    _common_contract(constrained)
    assert constrained.action == ppo.action
    assert constrained.environment == ppo.environment
    assert constrained.risk == ppo.risk
    assert constrained.reward == ppo.reward
    assert constrained.portfolio_risk == ppo.portfolio_risk
    assert _without_algorithm_specific_training(
        ppo.candidate_digest_payload()
    ) == _without_algorithm_specific_training(constrained.candidate_digest_payload())
    assert constrained.training.algorithm == "lagrangian_ppo"
    assert constrained.training.gamma == pytest.approx(1.0)
    assert constrained.training.discount_half_life_hours is None
    assert constrained.training.lagrangian_budgets == EXPECTED_BUDGETS
    assert (
        constrained.training.lagrangian_dual_learning_rates
        == EXPECTED_DUAL_LEARNING_RATES
    )
    assert (
        constrained.training.lagrangian_minimum_completed_episodes
        == EXPECTED_MINIMUM_SUPPORT
    )

    algorithm = build_algorithm_config(constrained.training)
    assert isinstance(algorithm, LagrangianPPOConfig)
    assert algorithm.cost_schema.names == CONSTRAINT_COST_NAMES
    assert algorithm.lagrangian_schema.names == CONSTRAINT_COST_NAMES


def test_discounted_profile_changes_only_real_time_discount() -> None:
    canonical = _load(LAGRANGIAN)
    discounted = _load(DISCOUNTED)

    _common_contract(discounted)
    assert discounted.training.algorithm == "lagrangian_ppo"
    assert discounted.training.gamma == pytest.approx(0.998969062762624)
    assert discounted.training.discount_half_life_hours == pytest.approx(168.0)
    assert _without_discount(canonical.candidate_digest_payload()) == _without_discount(
        discounted.candidate_digest_payload()
    )


def test_walk_forward_references_canonical_standalone_profiles() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / WALK_FORWARD,
        n_bars=55_392,
    )

    assert tuple(candidate.name for candidate in config.candidates) == EXPECTED_NAMES
    standalone = {
        EXPECTED_NAMES[0]: _load(PPO),
        EXPECTED_NAMES[1]: _load(LAGRANGIAN),
        EXPECTED_NAMES[2]: _load(DISCOUNTED),
    }
    for candidate in config.candidates:
        assert (
            candidate.run.candidate_digest_payload()
            == standalone[candidate.name].candidate_digest_payload()
        )

    scenario_names = tuple(
        scenario.name for scenario in config.execution_sensitivity.scenarios
    )
    assert scenario_names == (
        "nominal",
        "tick_2x",
        "lot_2x",
        "minimum_notional_2x",
        "joint_2x",
        "joint_5x",
        "joint_3x",
    )
    extensions = tuple(
        scenario
        for scenario in config.execution_sensitivity.scenarios
        if scenario.name not in {
            "nominal",
            "tick_2x",
            "lot_2x",
            "minimum_notional_2x",
            "joint_2x",
            "joint_5x",
        }
    )
    assert tuple(item.name for item in extensions) == ("joint_3x",)
    assert all(item.report_only for item in extensions)
    assert config.execution_sensitivity.required_scenario == "joint_2x"
    assert config.workflow.max_folds == 6
    assert config.workflow.selection_bars == 2_880
    assert config.workflow.test_bars == 2_880


def test_walk_forward_resolves_run_files_from_a_relative_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    relative_path = Path("examples/binance-multitimeframe") / WALK_FORWARD

    config = MarketWalkForwardConfig.from_json(relative_path, n_bars=55_392)

    assert tuple(candidate.name for candidate in config.candidates) == EXPECTED_NAMES
    assert (
        config.candidates[0].run.candidate_digest_payload()
        == _load(PPO).candidate_digest_payload()
    )
