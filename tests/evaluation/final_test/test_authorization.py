from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_lineage import _decided_first_experiment
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ArtifactIntegrityError,
    ContractViolationError,
    ExperimentDecisionKind,
    InvalidExperimentStateError,
    StudyOutcome,
    freeze_study,
)
from trade_rl.evaluation.final_test import (
    authorize_final_evaluation,
    inspect_final_evaluation_authorization,
)

_AUTHORIZED_AT = datetime(2026, 9, 13, 11, 45, tzinfo=UTC)
_FINAL_START = "2026-01-01T20:00:00.000000000"
_FINAL_STOP = "2026-01-02T20:00:00.000000000"


def _study_file_digests(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _winner_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    root, _, _, candidate, _ = _decided_first_experiment(
        tmp_path,
        monkeypatch,
        decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
    )
    freeze_study(
        root,
        outcome=StudyOutcome.WINNER,
        selected_evidence_digest=candidate.fingerprint,
        selected_strategy="ppo",
        rationale="Accepted development evidence supports final-test authorization.",
        frozen_by="researcher",
        frozen_at=datetime(2026, 9, 13, 11, 30, tzinfo=UTC),
    )
    return root, candidate.fingerprint


def test_winner_study_can_issue_one_sealed_authorization_without_mutating_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, winner_digest = _winner_study(tmp_path, monkeypatch)
    before = _study_file_digests(study_root)
    output = tmp_path / "final-authorization"

    authorization = authorize_final_evaluation(
        output,
        study_root=study_root,
        final_evaluation_start=_FINAL_START,
        final_evaluation_stop_exclusive=_FINAL_STOP,
        authorized_by="final-gate",
        authorized_at=_AUTHORIZED_AT,
    )

    assert authorization.winner_evidence_digest == winner_digest
    assert authorization.winner_strategy == "ppo"
    assert authorization.development_evaluation_stop_exclusive == _FINAL_START
    assert authorization.final_evaluation_start == _FINAL_START
    assert authorization.final_evaluation_stop_exclusive == _FINAL_STOP
    assert set(path.name for path in output.iterdir()) == {"authorization.json"}
    assert (
        inspect_final_evaluation_authorization(output, study_root=study_root)
        == authorization
    )
    assert _study_file_digests(study_root) == before

    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        authorize_final_evaluation(
            output,
            study_root=study_root,
            final_evaluation_start=_FINAL_START,
            final_evaluation_stop_exclusive=_FINAL_STOP,
            authorized_by="second-attempt",
            authorized_at=_AUTHORIZED_AT,
        )


def test_authorization_rejects_unfrozen_and_no_winner_studies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, _, _ = _with_baseline(tmp_path, monkeypatch)

    with pytest.raises(InvalidExperimentStateError, match="WINNER"):
        authorize_final_evaluation(
            tmp_path / "unfrozen-authorization",
            study_root=study_root,
            final_evaluation_start=_FINAL_START,
            final_evaluation_stop_exclusive=_FINAL_STOP,
            authorized_by="final-gate",
            authorized_at=_AUTHORIZED_AT,
        )

    freeze_study(
        study_root,
        outcome=StudyOutcome.NO_WINNER,
        rationale="Development evidence supports no winner.",
        frozen_by="researcher",
        frozen_at=datetime(2026, 9, 13, 11, 30, tzinfo=UTC),
    )
    with pytest.raises(InvalidExperimentStateError, match="WINNER"):
        authorize_final_evaluation(
            tmp_path / "no-winner-authorization",
            study_root=study_root,
            final_evaluation_start=_FINAL_START,
            final_evaluation_stop_exclusive=_FINAL_STOP,
            authorized_by="final-gate",
            authorized_at=_AUTHORIZED_AT,
        )


def test_authorization_rejects_final_window_overlap_or_empty_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, _ = _winner_study(tmp_path, monkeypatch)

    with pytest.raises(ContractViolationError, match="development"):
        authorize_final_evaluation(
            tmp_path / "overlap",
            study_root=study_root,
            final_evaluation_start="2026-01-01T19:00:00.000000000",
            final_evaluation_stop_exclusive=_FINAL_STOP,
            authorized_by="final-gate",
            authorized_at=_AUTHORIZED_AT,
        )
    with pytest.raises(ContractViolationError, match="strictly later"):
        authorize_final_evaluation(
            tmp_path / "empty",
            study_root=study_root,
            final_evaluation_start=_FINAL_START,
            final_evaluation_stop_exclusive=_FINAL_START,
            authorized_by="final-gate",
            authorized_at=_AUTHORIZED_AT,
        )


def test_inspection_rejects_authorization_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, _ = _winner_study(tmp_path, monkeypatch)
    output = tmp_path / "authorization"
    authorize_final_evaluation(
        output,
        study_root=study_root,
        final_evaluation_start=_FINAL_START,
        final_evaluation_stop_exclusive=_FINAL_STOP,
        authorized_by="final-gate",
        authorized_at=_AUTHORIZED_AT,
    )

    artifact_path = output / "authorization.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["authorization"]["final_evaluation_stop_exclusive"] = (
        "2026-01-03T20:00:00.000000000"
    )
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="digest"):
        inspect_final_evaluation_authorization(output, study_root=study_root)


def test_inspection_rejects_different_bound_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, _ = _winner_study(tmp_path / "first", monkeypatch)
    second_root, _ = _winner_study(tmp_path / "second", monkeypatch)
    output = tmp_path / "authorization"
    authorize_final_evaluation(
        output,
        study_root=first_root,
        final_evaluation_start=_FINAL_START,
        final_evaluation_stop_exclusive=_FINAL_STOP,
        authorized_by="final-gate",
        authorized_at=_AUTHORIZED_AT,
    )

    with pytest.raises(ArtifactIntegrityError, match="Study"):
        inspect_final_evaluation_authorization(output, study_root=second_root)


def test_authorization_rejects_symlink_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, _ = _winner_study(tmp_path, monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "authorization-link"
    try:
        symlink.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises((ArtifactIntegrityError, InvalidExperimentStateError)):
        authorize_final_evaluation(
            symlink,
            study_root=study_root,
            final_evaluation_start=_FINAL_START,
            final_evaluation_stop_exclusive=_FINAL_STOP,
            authorized_by="final-gate",
            authorized_at=_AUTHORIZED_AT,
        )
