from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ControlledVerificationStatus,
    define_experiment,
    load_evidence_set,
    run_experiment,
    verify_experiment,
)


def test_study_seed_binding_does_not_change_frozen_candidate_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Study-owned seed binding is not a controlled factor.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    run_experiment(root, 1, dataset_root=dataset_root)

    loaded = load_evidence_set(root / "experiments" / "0001" / "candidate" / "evidence")
    assert loaded.semantic_config["ppo_seed"] == snapshot.plan.ppo_seeds[0]
    assert definition.candidate_config.ppo_seed == snapshot.plan.ppo_seeds[0]

    verification = verify_experiment(root, 1)
    assert verification.status is ControlledVerificationStatus.CONTROLLED
    assert verification.violations == ()
