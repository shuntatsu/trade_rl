from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    define_experiment,
    inspect_study,
    run_experiment,
)


def test_candidate_analysis_failure_keeps_definition_but_no_partial_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import workflow as workflow_module

    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Candidate bundle publication must be atomic.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )

    def fail_analysis(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic candidate analysis failure")

    monkeypatch.setattr(workflow_module, "analyze_evidence_set", fail_analysis)

    with pytest.raises(RuntimeError, match="synthetic candidate analysis failure"):
        run_experiment(root, 1, dataset_root=dataset_root)

    experiment_root = root / "experiments" / "0001"
    assert (experiment_root / "definition.json").is_file()
    assert not (experiment_root / "candidate").exists()
    assert inspect_study(root).experiment_sequences == (1,)
