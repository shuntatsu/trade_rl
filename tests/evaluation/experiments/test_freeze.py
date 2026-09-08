from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_lineage import _decided_first_experiment
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ContractViolationError,
    ControlledFactor,
    ExperimentDecisionKind,
    InvalidExperimentStateError,
    StudyFrozenError,
    StudyOutcome,
    compare_experiment,
    decide_experiment,
    define_experiment,
    freeze_study,
    inspect_study,
    record_experiment_failure,
    run_baseline,
    run_experiment,
    verify_experiment,
)

_FREEZE_AT = datetime(2026, 9, 9, 8, 30, tzinfo=UTC)


def test_no_winner_freeze_is_terminal_and_inspectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    frozen = freeze_study(
        root,
        outcome=StudyOutcome.NO_WINNER,
        rationale="No accepted candidate is supportable from development evidence.",
        frozen_by="researcher",
        frozen_at=_FREEZE_AT,
    )

    assert frozen.outcome is StudyOutcome.NO_WINNER
    assert (root / "freeze.json").is_file()
    rebuilt = inspect_study(root)
    assert rebuilt.freeze == frozen

    mutation_calls = (
        lambda: run_baseline(root, dataset_root=dataset_root),
        lambda: define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="post-freeze mutation",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=64),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        ),
        lambda: run_experiment(root, 1, dataset_root=dataset_root),
        lambda: verify_experiment(root, 1),
        lambda: compare_experiment(root, 1),
        lambda: decide_experiment(
            root,
            1,
            decision=ExperimentDecisionKind.KEEP_BASELINE,
            rationale="post-freeze mutation",
            decided_by="researcher",
            decided_at=_FREEZE_AT,
        ),
        lambda: record_experiment_failure(
            root,
            1,
            reason="post-freeze mutation",
            recorded_by="researcher",
            recorded_at=_FREEZE_AT,
        ),
    )
    for mutate in mutation_calls:
        with pytest.raises(StudyFrozenError):
            mutate()

    with pytest.raises(StudyFrozenError):
        freeze_study(
            root,
            outcome=StudyOutcome.NO_WINNER,
            rationale="freeze is one-shot",
            frozen_by="researcher",
            frozen_at=_FREEZE_AT,
        )


def test_freeze_rejects_open_nonterminal_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Open experiment prevents freeze.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )

    with pytest.raises(InvalidExperimentStateError, match="nonterminal"):
        freeze_study(
            root,
            outcome=StudyOutcome.NO_WINNER,
            rationale="Cannot freeze around an open attempt.",
            frozen_by="researcher",
            frozen_at=_FREEZE_AT,
        )
    assert not (root / "freeze.json").exists()


def test_winner_requires_accepted_candidate_lineage_and_candidate_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _, candidate, _ = _decided_first_experiment(
        tmp_path,
        monkeypatch,
        decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
    )

    frozen = freeze_study(
        root,
        outcome=StudyOutcome.WINNER,
        selected_evidence_digest=candidate.fingerprint,
        selected_strategy="ppo",
        rationale="Accepted candidate evidence supports PPO as the development winner.",
        frozen_by="researcher",
        frozen_at=_FREEZE_AT,
    )
    assert frozen.selected_evidence_digest == candidate.fingerprint
    assert frozen.selected_strategy == "ppo"


def test_winner_rejects_control_and_rejected_candidate_resurrection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _, candidate, _ = _decided_first_experiment(
        tmp_path,
        monkeypatch,
        decision=ExperimentDecisionKind.KEEP_BASELINE,
    )

    with pytest.raises(InvalidExperimentStateError, match="ACCEPT"):
        freeze_study(
            root,
            outcome=StudyOutcome.WINNER,
            selected_evidence_digest=candidate.fingerprint,
            selected_strategy="ppo",
            rationale="Rejected candidate cannot be resurrected.",
            frozen_by="researcher",
            frozen_at=_FREEZE_AT,
        )
    with pytest.raises(ContractViolationError, match="candidate strategy"):
        freeze_study(
            root,
            outcome=StudyOutcome.WINNER,
            selected_evidence_digest=candidate.fingerprint,
            selected_strategy="cash",
            rationale="Controls are benchmarks only.",
            frozen_by="researcher",
            frozen_at=_FREEZE_AT,
        )
