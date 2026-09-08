from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ExperimentDecisionKind,
    InvalidExperimentStateError,
    compare_experiment,
    decide_experiment,
    define_experiment,
    inspect_study,
    run_experiment,
    verify_experiment,
)


def _decided_first_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: ExperimentDecisionKind,
):
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        hypothesis="Create one candidate EvidenceSet for lineage testing.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    candidate = run_experiment(root, 1, dataset_root=dataset_root)
    verification = verify_experiment(root, 1)
    comparison = compare_experiment(root, 1)
    decided = decide_experiment(
        root,
        1,
        decision=decision,
        rationale=f"Lineage test decision: {decision.value}",
        decided_by="researcher",
        decided_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert decided.comparison_digest == comparison.digest
    return root, dataset_root, snapshot, candidate, verification


def test_accept_candidate_enters_reachable_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, snapshot, candidate, _ = _decided_first_experiment(
        tmp_path,
        monkeypatch,
        decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
    )
    assert snapshot.baseline is not None

    rebuilt = inspect_study(root)
    assert rebuilt.lineage_evidence_digests == (
        snapshot.baseline.fingerprint,
        candidate.fingerprint,
    )
    definition = define_experiment(
        root,
        hypothesis="Continue from the accepted candidate with one more registered step.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=96),
        baseline_evidence_digest=candidate.fingerprint,
    )
    assert definition.sequence == 2
    assert definition.baseline_evidence_digest == candidate.fingerprint


@pytest.mark.parametrize(
    "decision",
    (
        ExperimentDecisionKind.KEEP_BASELINE,
        ExperimentDecisionKind.INCONCLUSIVE,
    ),
)
def test_nonaccepted_decision_candidate_never_enters_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ExperimentDecisionKind,
) -> None:
    root, _, snapshot, candidate, _ = _decided_first_experiment(
        tmp_path,
        monkeypatch,
        decision=decision,
    )
    assert snapshot.baseline is not None
    assert inspect_study(root).lineage_evidence_digests == (
        snapshot.baseline.fingerprint,
    )

    with pytest.raises(InvalidExperimentStateError, match="lineage"):
        define_experiment(
            root,
            hypothesis="Rejected lineage must not be resurrected.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=96),
            baseline_evidence_digest=candidate.fingerprint,
        )


def test_invalid_candidate_never_enters_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        hypothesis="Declare wrong factor so the attempt becomes INVALID.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), rule_entry_threshold=0.20),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    candidate = run_experiment(root, 1, dataset_root=dataset_root)
    verification = verify_experiment(root, 1)
    assert verification.status.value == "INVALID"
    assert inspect_study(root).lineage_evidence_digests == (
        snapshot.baseline.fingerprint,
    )

    with pytest.raises(InvalidExperimentStateError, match="lineage"):
        define_experiment(
            root,
            hypothesis="INVALID evidence cannot be a baseline.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=96),
            baseline_evidence_digest=candidate.fingerprint,
        )
