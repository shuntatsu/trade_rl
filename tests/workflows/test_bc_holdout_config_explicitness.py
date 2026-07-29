from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "examples/quickstart/training.json"


def test_v3_preserves_explicit_bc_holdout_statistical_settings() -> None:
    config = TrainingRunConfig.from_json(PROFILE)
    identity = config.training.digest_payload()

    assert config.training.behavior_cloning_causal_holdout_bootstrap_resamples == 2_000
    assert (
        config.training.behavior_cloning_causal_holdout_confidence_level
        == pytest.approx(0.95)
    )
    assert identity["behavior_cloning_causal_holdout_bootstrap_resamples"] == 2_000
    assert identity[
        "behavior_cloning_causal_holdout_confidence_level"
    ] == pytest.approx(0.95)


@pytest.mark.parametrize(
    "field",
    (
        "behavior_cloning_causal_holdout_bootstrap_resamples",
        "behavior_cloning_causal_holdout_confidence_level",
    ),
)
def test_v3_requires_explicit_bc_holdout_statistical_settings(field: str) -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    modified = deepcopy(payload)
    training = modified["training"]
    assert isinstance(training, dict)
    training.pop(field)

    with pytest.raises(ValueError, match=field):
        TrainingRunConfig.from_mapping(modified)
