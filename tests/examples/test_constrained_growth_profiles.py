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

CONTROL = "training-constrained-growth-control.json"
CANONICAL = "training-constrained-growth.json"
GAE_097 = "training-constrained-growth-gae097.json"
DISCOUNTED = "training-constrained-growth-discounted.json"
WALK_FORWARD = "walk-forward-constrained-growth.json"

EXPECTED_CANDIDATE_NAMES = (
    "growth-optimal-ppo-pr-d-control",
    "constrained-growth-canonical",
    "constrained-growth-gae097",
    "constrained-growth-discounted-gamma09995",
)
EXPECTED_BUDGETS = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.03)
EXPECTED_DUAL_LEARNING_RATES = (0.001, 0.01, 0.001, 0.01, 0.001, 0.001, 0.001)
EXPECTED_EMA_BETAS = (0.95,) * 7
EXPECTED_INITIAL_MULTIPLIERS = (0.0,) * 7
EXPECTED_MAX_MULTIPLIERS = (100.0,) * 7
EXPECTED_WARMUP_ROLLOUTS = (4,) * 7
EXPECTED_UPDATE_INTERVALS = (1,) * 7
EXPECTED_MINIMUM_SUPPORT = (1, 20, 1, 20, 1, 1, 1)


def _load_training(name: str) -> TrainingRunConfig:
    return TrainingRunConfig.from_json(EXAMPLE_ROOT / name)


def _payload_without_training_field(
    config: TrainingRunConfig,
    field: str,
) -> dict[str, object]:
    payload = deepcopy(config.candidate_digest_payload())
    training = payload["training"]
    assert isinstance(training, dict)
    training.pop(field)
    return payload


def _assert_common_full_scale_contract(config: TrainingRunConfig) -> None:
    training = config.training
    assert training.seeds == (0, 1, 2)
    assert training.timesteps == 524_288
    assert training.batch_size == 128
    assert training.n_steps == 256
    assert training.n_envs == 4
    assert training.n_epochs == 10
    assert training.device == "cuda"
    assert training.learning_rate == pytest.approx(0.00012)
    assert training.learning_rate_schedule == "linear"
    assert training.learning_rate_final_ratio == pytest.approx(0.1)
    assert training.policy_net_arch == (384, 256, 128)
    assert training.value_net_arch == (512, 384, 256)
    assert training.sequence_d_model == 336
    assert training.sequence_timeframe_attention_heads == 8
    assert training.sequence_timeframe_attention_layers == 2
    assert training.sequence_asset_attention_heads == 8
    assert training.sequence_asset_attention_layers == 2
    assert training.sequence_compile is True
    assert training.sequence_compile_mode == "reduce-overhead"
    assert training.sequence_transfer_mode == "pinned_non_blocking"
    assert training.vector_environment_mode == "subprocess"
    assert config.reward.absolute_growth_weight == pytest.approx(1.0)
    assert config.reward.excess_growth_weight == pytest.approx(0.0)
    assert config.reward.incremental_drawdown_weight == pytest.approx(0.0)
    assert config.reward.baseline_underperformance_weight == pytest.approx(0.0)
    assert config.reward.projection_penalty_weight == pytest.approx(0.0)
    assert config.reward.terminal_equity_weight == pytest.approx(0.0)
    assert config.reward.margin_deficit_weight == pytest.approx(0.0)


def test_pr_d_control_is_explicit_and_preserves_growth_objective() -> None:
    control = _load_training(CONTROL)

    _assert_common_full_scale_contract(control)
    assert control.training.algorithm == "ppo"
    assert control.training.gamma == pytest.approx(1.0)
    assert control.training.gae_lambda == pytest.approx(0.95)
    assert "cost_critic" not in control.training.digest_payload()
    assert "lagrangian" not in control.training.digest_payload()


def test_canonical_constrained_profile_closes_all_seven_costs() -> None:
    config = _load_training(CANONICAL)

    _assert_common_full_scale_contract(config)
    training = config.training
    assert training.algorithm == "lagrangian_ppo"
    assert training.gamma == pytest.approx(1.0)
    assert training.gae_lambda == pytest.approx(0.95)
    assert training.cost_continuous_gae_lambda == pytest.approx(0.95)
    assert training.cost_event_gae_lambda == pytest.approx(0.95)
    assert training.lagrangian_budgets == EXPECTED_BUDGETS
    assert training.lagrangian_dual_learning_rates == EXPECTED_DUAL_LEARNING_RATES
    assert training.lagrangian_ema_betas == EXPECTED_EMA_BETAS
    assert training.lagrangian_initial_multipliers == EXPECTED_INITIAL_MULTIPLIERS
    assert training.lagrangian_max_multipliers == EXPECTED_MAX_MULTIPLIERS
    assert training.lagrangian_warmup_rollouts == EXPECTED_WARMUP_ROLLOUTS
    assert training.lagrangian_update_interval_rollouts == EXPECTED_UPDATE_INTERVALS
    assert training.lagrangian_minimum_completed_episodes == EXPECTED_MINIMUM_SUPPORT
    assert training.lagrangian_probe_episodes == 20
    assert training.lagrangian_probe_max_steps_per_episode == 2_880

    algorithm = build_algorithm_config(training)
    assert isinstance(algorithm, LagrangianPPOConfig)
    assert algorithm.cost_schema.names == CONSTRAINT_COST_NAMES
    assert algorithm.lagrangian_schema.names == CONSTRAINT_COST_NAMES
    assert tuple(spec.budget for spec in algorithm.lagrangian_schema.specs) == (
        EXPECTED_BUDGETS
    )
    assert (
        tuple(
            spec.minimum_completed_episodes
            for spec in algorithm.lagrangian_schema.specs
        )
        == EXPECTED_MINIMUM_SUPPORT
    )
    assert tuple(spec.gamma for spec in algorithm.cost_schema.specs) == (1.0,) * 7


def test_gae_ablation_changes_only_reward_gae_lambda() -> None:
    canonical = _load_training(CANONICAL)
    ablation = _load_training(GAE_097)

    assert canonical.candidate_digest_payload() != ablation.candidate_digest_payload()
    assert ablation.training.gamma == pytest.approx(1.0)
    assert ablation.training.gae_lambda == pytest.approx(0.97)
    assert ablation.training.cost_continuous_gae_lambda == pytest.approx(0.95)
    assert ablation.training.cost_event_gae_lambda == pytest.approx(0.95)
    assert _payload_without_training_field(
        canonical, "gae_lambda"
    ) == _payload_without_training_field(ablation, "gae_lambda")


def test_discounted_ablation_changes_only_reward_gamma() -> None:
    canonical = _load_training(CANONICAL)
    ablation = _load_training(DISCOUNTED)

    assert canonical.candidate_digest_payload() != ablation.candidate_digest_payload()
    assert ablation.training.gamma == pytest.approx(0.9995)
    assert ablation.training.gae_lambda == pytest.approx(0.95)
    algorithm = build_algorithm_config(ablation.training)
    assert isinstance(algorithm, LagrangianPPOConfig)
    assert tuple(spec.gamma for spec in algorithm.cost_schema.specs) == (1.0,) * 7
    assert _payload_without_training_field(
        canonical, "gamma"
    ) == _payload_without_training_field(ablation, "gamma")


def test_walk_forward_profile_binds_all_four_candidates_and_joint_stress() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / WALK_FORWARD,
        n_bars=55_392,
    )

    assert tuple(candidate.name for candidate in config.candidates) == (
        EXPECTED_CANDIDATE_NAMES
    )
    standalone = {
        EXPECTED_CANDIDATE_NAMES[0]: _load_training(CONTROL),
        EXPECTED_CANDIDATE_NAMES[1]: _load_training(CANONICAL),
        EXPECTED_CANDIDATE_NAMES[2]: _load_training(GAE_097),
        EXPECTED_CANDIDATE_NAMES[3]: _load_training(DISCOUNTED),
    }
    for candidate in config.candidates:
        assert (
            candidate.run.candidate_digest_payload()
            == standalone[candidate.name].candidate_digest_payload()
        )

    scenario_names = tuple(
        scenario.name for scenario in config.execution_sensitivity.scenarios
    )
    assert "nominal" in scenario_names
    assert "joint_2x" in scenario_names
    assert config.execution_sensitivity.required_scenario == "joint_2x"
    assert config.workflow.max_folds == 6
    assert config.workflow.selection_bars == 2_880
    assert config.workflow.test_bars == 2_880
    assert config.sealed_test_ledger_mode.value == "durable_postgres"
