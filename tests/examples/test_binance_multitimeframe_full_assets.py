from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def test_full_training_config_is_not_a_smoke_run() -> None:
    config = TrainingRunConfig.from_json(EXAMPLE_ROOT / "training-full.json")

    assert config.training.algorithm == "ppo"
    assert config.training.device == "cpu"
    assert config.training.seeds == (0, 1, 2)
    assert config.training.timesteps >= 131_072
    assert config.training.n_steps == 2_048
    assert config.training.batch_size == 64
    assert config.training.n_epochs == 10
    assert config.training.policy_net_arch == (128, 128)
    assert config.environment.decision_hours == 1.0
    assert config.environment.episode_hours >= 720.0
    execution = config.environment.execution_cost
    assert execution.fee_rate > 0.0
    assert execution.spread_rate > 0.0
    assert execution.impact_rate > 0.0
    assert config.portfolio_risk.max_abs_weight is not None
    assert config.portfolio_risk.max_abs_weight <= 0.5


def test_full_walk_forward_config_has_two_material_folds() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / "walk-forward-full.json",
        n_bars=13_104,
    )

    folds = config.workflow.build_folds()
    assert len(folds) == 2
    assert len(config.candidates) == 1
    candidate = config.candidates[0].run
    assert candidate.training.seeds == (0, 1, 2)
    assert candidate.training.timesteps >= 32_768
    assert candidate.environment.decision_hours == 1.0


def test_full_runner_uses_three_assets_and_four_native_timeframes() -> None:
    content = (EXAMPLE_ROOT / "run_full_research.py").read_text(encoding="utf-8")

    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        assert symbol in content
    for timeframe in ("15m", "1h", "4h", "1d"):
        assert timeframe in content
    assert "2024-12-01T00:00:00Z" in content
    assert "2026-06-01T00:00:00Z" in content
    assert "dataset_id" in content
    assert "artifact_digest" in content
    assert '"train", "run"' in content
    assert '"walk-forward", "run"' in content
