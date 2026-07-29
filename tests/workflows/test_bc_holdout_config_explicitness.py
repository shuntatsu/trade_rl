from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "examples/quickstart/training.json"


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
