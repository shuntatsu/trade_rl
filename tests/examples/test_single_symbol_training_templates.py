from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"

TRAINING_PROFILES = (
    "training-full.json",
    "training-target-weight-growth-ppo.json",
    "training-target-weight-constrained-growth.json",
    "training-target-weight-constrained-growth-discounted.json",
)
WALK_FORWARD_PROFILES = (
    "walk-forward-full.json",
    "walk-forward-target-weight-constrained-growth.json",
    "walk-forward-constrained-growth.json",
)


def _assert_single_symbol_run(config: TrainingRunConfig) -> None:
    assert config.schema_version == "training_run_config_v4"
    assert config.action.target_weight_count == 1
    assert config.action.names_for_symbols(("BTCUSDT",)) == (
        "target_weight:BTCUSDT",
    )
    assert config.risk.max_gross == pytest.approx(1.0)
    assert config.risk.max_abs_weight == pytest.approx(1.0)
    assert config.portfolio_risk.max_abs_weight == pytest.approx(1.0)
    assert config.environment.execution_cost.max_leverage == pytest.approx(1.0)


@pytest.mark.parametrize("name", TRAINING_PROFILES)
def test_maintained_training_profile_uses_one_btc_action(name: str) -> None:
    _assert_single_symbol_run(TrainingRunConfig.from_json(EXAMPLE_ROOT / name))


@pytest.mark.parametrize("name", WALK_FORWARD_PROFILES)
def test_maintained_walk_forward_candidates_use_one_btc_action(name: str) -> None:
    config = MarketWalkForwardConfig.from_json(EXAMPLE_ROOT / name, n_bars=55_392)

    assert config.candidates
    for candidate in config.candidates:
        _assert_single_symbol_run(candidate.run)
