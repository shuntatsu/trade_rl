from __future__ import annotations

import math
from pathlib import Path

from trade_rl.rl.environment_config import EpisodeBoundaryMode
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def test_pure_growth_profiles_use_boundary_semantics_matching_discounting() -> None:
    gamma_one: list[str] = []
    discounted: list[str] = []

    for path in sorted(PROFILE_ROOT.glob("training*.json")):
        config = TrainingRunConfig.from_json(path)
        if not config.reward.is_pure_net_log_growth():
            continue

        assert config.environment.liquidate_on_end is False, path.name
        if math.isclose(config.training.gamma, 1.0, rel_tol=0.0, abs_tol=1e-12):
            gamma_one.append(path.name)
            assert (
                config.environment.episode_boundary_mode
                is EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION
            ), path.name
            assert config.environment.finite_horizon_observation is True, path.name
        else:
            discounted.append(path.name)
            assert (
                config.environment.episode_boundary_mode
                is EpisodeBoundaryMode.EXTERNAL_TRUNCATION
            ), path.name
            assert config.environment.finite_horizon_observation is False, path.name

    assert gamma_one
    assert discounted
