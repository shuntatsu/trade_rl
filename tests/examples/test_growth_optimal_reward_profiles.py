import json
from pathlib import Path

import pytest

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _assert_growth_optimal_contract(config: TrainingRunConfig) -> None:
    assert config.training.gamma == pytest.approx(1.0)
    assert config.training.discount_half_life_hours is None
    assert config.environment.finite_horizon_observation is True
    assert config.environment.episode_hours == pytest.approx(720.0)
    assert config.reward.absolute_growth_weight == pytest.approx(1.0)
    assert config.reward.excess_growth_weight == pytest.approx(0.0)
    assert config.reward.incremental_drawdown_weight == pytest.approx(0.0)
    assert config.reward.baseline_underperformance_weight == pytest.approx(0.0)
    assert config.reward.projection_penalty_weight == pytest.approx(0.0)
    assert config.reward.terminal_equity_weight == pytest.approx(0.0)
    assert config.reward.margin_deficit_weight == pytest.approx(0.0)
    assert config.environment.execution_cost.fee_rate > 0.0
    assert config.environment.execution_cost.spread_rate > 0.0
    assert config.environment.execution_cost.impact_rate > 0.0
    assert config.risk.drawdown_stop == pytest.approx(0.20)


def test_growth_optimal_full_training_profile_is_explicit_and_parseable() -> None:
    config = TrainingRunConfig.from_json(EXAMPLE_ROOT / "training-growth-optimal.json")

    _assert_growth_optimal_contract(config)


def test_growth_optimal_walk_forward_profile_uses_same_objective() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / "walk-forward-growth-optimal.json",
        n_bars=55_392,
    )

    assert [candidate.name for candidate in config.candidates] == [
        "growth-optimal-ppo-15m"
    ]
    _assert_growth_optimal_contract(config.candidates[0].run)
    assert config.maximum_selection_drawdown == pytest.approx(0.20)
    assert config.minimum_selection_uplift == pytest.approx(0.001)
    assert config.execution_sensitivity.required_scenario == "joint_2x"


def test_growth_optimal_walk_forward_candidate_matches_standalone_profile() -> None:
    standalone = json.loads(
        (EXAMPLE_ROOT / "training-growth-optimal.json").read_text(encoding="utf-8")
    )
    walk_forward = json.loads(
        (EXAMPLE_ROOT / "walk-forward-growth-optimal.json").read_text(encoding="utf-8")
    )

    assert walk_forward["candidates"][0]["run"] == standalone
