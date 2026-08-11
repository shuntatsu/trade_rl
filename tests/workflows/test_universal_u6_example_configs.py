from __future__ import annotations

from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_full_research_training import (
    prepare_universal_full_research_training_configs,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm


_ROOT = Path("examples/binance-multitimeframe")
_PATHS = {
    FullResearchAlgorithm.PPO: _ROOT / "universal-u6-ppo.json",
    FullResearchAlgorithm.LAGRANGIAN: _ROOT / "universal-u6-lagrangian.json",
    FullResearchAlgorithm.DISCOUNTED: _ROOT / "universal-u6-discounted.json",
}


def _non_training_digest(config: TrainingRunConfig) -> str:
    payload = dict(config.candidate_digest_payload())
    payload.pop("training", None)
    return content_digest(payload)


def test_universal_u6_example_configs_close_the_maintained_comparison() -> None:
    configs = {
        algorithm: TrainingRunConfig.from_json(path)
        for algorithm, path in _PATHS.items()
    }

    assert len({_non_training_digest(config) for config in configs.values()}) == 1
    for config in configs.values():
        training = config.training
        assert training.behavior_cloning_seed is not None
        assert training.behavior_cloning_critic_warm_start_steps > 0
        assert training.behavior_cloning_joint_warm_start_steps > 0
        assert training.behavior_cloning_teacher == "oracle"
        assert training.gamma > 0.0

    prepared = prepare_universal_full_research_training_configs(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        algorithm_configs={
            algorithm: config.training for algorithm, config in configs.items()
        },
    )
    assert tuple(item.algorithm for item in prepared) == tuple(FullResearchAlgorithm)
    assert len({item.fixed_condition_digest for item in prepared}) == 1

    discounted = configs[FullResearchAlgorithm.DISCOUNTED].training
    assert discounted.discount_half_life_hours is not None
    assert 0.0 < discounted.gamma < 1.0


def test_start_uses_canonical_universal_u6_example_configs() -> None:
    start = Path("START.md").read_text(encoding="utf-8")
    for path in _PATHS.values():
        assert str(path) in start
