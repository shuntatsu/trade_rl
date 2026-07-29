from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.learning.evaluation import deterministic_bootstrap_upper_bound


def test_bootstrap_upper_bound_is_deterministic_and_one_sided() -> None:
    values = np.array([0.01, 0.03, 0.02, 0.08, 0.04])
    first = deterministic_bootstrap_upper_bound(values, confidence_level=0.95, resamples=2_000, seed_material="a" * 64)
    second = deterministic_bootstrap_upper_bound(values, confidence_level=0.95, resamples=2_000, seed_material="a" * 64)
    assert first == second
    assert first >= float(np.mean(values))


def test_maintained_profiles_require_nontrivial_causal_evidence() -> None:
    root = Path(__file__).resolve().parents[2] / "examples/binance-multitimeframe"
    for name in (
        "training-target-weight-growth-ppo.json",
        "training-target-weight-constrained-growth.json",
        "training-target-weight-constrained-growth-discounted.json",
    ):
        training = json.loads((root / name).read_text())["training"]
        assert training["behavior_cloning_required_relative_improvement"] > 0.0
        assert training["behavior_cloning_min_causal_holdout_trades"] >= 30
        assert training["behavior_cloning_causal_holdout_bootstrap_resamples"] >= 2_000
        assert training["behavior_cloning_causal_holdout_confidence_level"] >= 0.95
