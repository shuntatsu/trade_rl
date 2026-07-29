from __future__ import annotations

from pathlib import Path

from trade_rl.workflows._market_walk_forward_core import (
    _maintained_training_environment,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    normalize_training_run_config,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    ROOT
    / "examples/binance-multitimeframe/training-target-weight-constrained-growth.json"
)


def test_standalone_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    normalized = normalize_training_run_config(config)

    assert normalized.environment.liquidate_on_end is False
    assert normalized.environment.terminal_accounting_mode == "mark_to_market"
    assert normalized.environment.finite_horizon_observation is False


def test_walk_forward_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    environment = _maintained_training_environment(
        config.environment,
        episode_bars=1_024,
    )

    assert environment.episode_bars == 1_024
    assert environment.episode_hour_choices == ()
    assert environment.liquidate_on_end is False
    assert environment.terminal_accounting_mode == "mark_to_market"
    assert environment.require_full_reward_preroll is True
