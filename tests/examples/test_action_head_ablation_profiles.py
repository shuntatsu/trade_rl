from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig


_ROOT = Path(__file__).parents[2] / "examples" / "binance-multitimeframe"
_GATE = _ROOT / "training-action-head-ablation-gate.json"
_DIRECT = _ROOT / "training-action-head-ablation-direct.json"
_WALK_FORWARD = _ROOT / "walk-forward-action-head-ablation.json"


def _mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_action_head_ablation_training_profiles_differ_only_by_actor_head() -> None:
    gate = deepcopy(_mapping(_GATE))
    direct = deepcopy(_mapping(_DIRECT))
    gate_training = gate["training"]
    direct_training = direct["training"]
    assert isinstance(gate_training, dict)
    assert isinstance(direct_training, dict)

    assert gate_training.pop("policy_actor_head") == "hierarchical_gate_target_v1"
    assert direct_training.pop("policy_actor_head") == "shared_target_v1"
    assert gate == direct

    assert TrainingRunConfig.from_json(_GATE).training.policy_actor_head == (
        "hierarchical_gate_target_v1"
    )
    assert TrainingRunConfig.from_json(_DIRECT).training.policy_actor_head == (
        "shared_target_v1"
    )


def test_action_head_ablation_walk_forward_resolves_exact_paired_candidates() -> None:
    raw = _mapping(_WALK_FORWARD)
    candidates = raw["candidates"]
    assert candidates == [
        {
            "name": "target-weight-gate-head-ppo",
            "run_file": "training-action-head-ablation-gate.json",
        },
        {
            "name": "target-weight-direct-head-ppo",
            "run_file": "training-action-head-ablation-direct.json",
        },
    ]

    config = MarketWalkForwardConfig.from_json(_WALK_FORWARD)
    assert tuple(candidate.name for candidate in config.candidates) == (
        "target-weight-gate-head-ppo",
        "target-weight-direct-head-ppo",
    )
    assert tuple(
        candidate.run.training.policy_actor_head for candidate in config.candidates
    ) == (
        "hierarchical_gate_target_v1",
        "shared_target_v1",
    )
