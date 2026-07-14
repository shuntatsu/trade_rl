from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]


def test_binance_training_smoke_is_small_and_disables_long_reward_preroll() -> None:
    config = TrainingRunConfig.from_json(
        ROOT / "examples" / "binance" / "training-smoke.json"
    )

    assert config.training.algorithm == "ppo"
    assert config.training.device == "cpu"
    assert config.training.seeds == (0,)
    assert config.training.timesteps <= 128
    assert config.reward.baseline_underperformance_weight == 0.0
    assert config.environment.episode_hours <= 24.0
    assert config.trend.slow_hours <= 12.0


def test_binance_walk_forward_smoke_constructs_one_complete_fold() -> None:
    config = MarketWalkForwardConfig.from_json(
        ROOT / "examples" / "binance" / "walk-forward-smoke.json",
        n_bars=672,
    )

    folds = config.workflow.build_folds()
    assert len(folds) == 1
    assert len(config.candidates) == 1
    assert config.candidates[0].run.reward.baseline_underperformance_weight == 0.0


def test_binance_smoke_runner_executes_all_authoritative_cli_stages() -> None:
    content = (ROOT / "examples" / "binance" / "run_e2e_smoke.py").read_text(
        encoding="utf-8"
    )

    assert '"data", "binance"' in content
    assert '"train", "run"' in content
    assert '"walk-forward", "run"' in content
    assert "dataset_id" in content
    assert "manifest.json" in content
