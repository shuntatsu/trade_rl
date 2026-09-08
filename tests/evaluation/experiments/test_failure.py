from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    InvalidExperimentStateError,
    define_experiment,
    inspect_study,
    record_experiment_failure,
    run_experiment,
)


def _defined_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Operational failure before complete evidence is terminal.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    return root, dataset_root, snapshot, definition


def test_failure_is_terminal_budgeted_and_does_not_enter_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot, definition = _defined_experiment(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    failure = record_experiment_failure(
        root,
        1,
        reason="synthetic worker crash before candidate publication",
        recorded_by="researcher",
        recorded_at=datetime(2026, 9, 9, tzinfo=UTC),
    )

    assert failure.experiment_digest == definition.digest
    assert (root / "experiments" / "0001" / "failure.json").is_file()
    rebuilt = inspect_study(root)
    assert rebuilt.experiment_sequences == (1,)
    assert rebuilt.lineage_evidence_digests == (snapshot.baseline.fingerprint,)

    with pytest.raises(InvalidExperimentStateError, match="FAILED"):
        run_experiment(root, 1, dataset_root=dataset_root)
    with pytest.raises(InvalidExperimentStateError, match="failure"):
        record_experiment_failure(
            root,
            1,
            reason="second failure must not overwrite",
            recorded_by="researcher",
            recorded_at=datetime(2026, 9, 9, tzinfo=UTC),
        )

    second = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="FAILED attempt consumed sequence 1 and budget.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=96),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    assert second.sequence == 2


def test_failure_cannot_replace_complete_candidate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, _, _ = _defined_experiment(tmp_path, monkeypatch)
    run_experiment(root, 1, dataset_root=dataset_root)

    with pytest.raises(InvalidExperimentStateError, match="candidate"):
        record_experiment_failure(
            root,
            1,
            reason="complete candidate evidence cannot be rewritten as FAILED",
            recorded_by="researcher",
            recorded_at=datetime(2026, 9, 9, tzinfo=UTC),
        )

    assert not (root / "experiments" / "0001" / "failure.json").exists()
