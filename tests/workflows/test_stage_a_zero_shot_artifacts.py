from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import (
    build_sealed_test_access_record,
)
from trade_rl.workflows import stage_a_zero_shot_artifacts as artifacts_module
from trade_rl.workflows.stage_a_zero_shot_artifacts import (
    StageAZeroShotArtifactPublisher,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageASealedTestAccessRecord,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_marker(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_validation_publication_contains_only_complete_phase_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        artifacts_module,
        "write_stage_a_evaluation_evidence",
        lambda path, _evidence: _write_marker(path, "evidence"),
    )
    monkeypatch.setattr(
        artifacts_module,
        "write_stage_a_validation_selection",
        lambda path, _selection: _write_marker(path, "selection"),
    )
    run = SimpleNamespace(evidence=object(), selection=object())

    final = StageAZeroShotArtifactPublisher(tmp_path).publish_validation(run)

    assert final == tmp_path / "validation"
    assert sorted(path.name for path in final.iterdir()) == [
        "evidence.json",
        "selection.json",
    ]
    assert not list(tmp_path.glob(".validation-*"))
    assert not (tmp_path / ".validation.lock").exists()


def test_sealed_test_publication_includes_canonical_access_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        artifacts_module,
        "write_stage_a_evaluation_evidence",
        lambda path, _evidence: _write_marker(path, "evidence"),
    )
    monkeypatch.setattr(
        artifacts_module,
        "write_stage_a_sealed_test_decision",
        lambda path, _decision: _write_marker(path, "decision"),
    )
    record = build_sealed_test_access_record(
        experiment_plan_digest=_digest("plan"),
        dataset_id=_digest("dataset"),
        fold_index=0,
        test_range=IndexRange(100, 120),
        selected_configuration="candidate-a",
        selected_policy_digest=_digest("candidate"),
    )
    stage_a_record = StageASealedTestAccessRecord(
        evaluation_dataset_manifest_digest=_digest("manifest"),
        triplet_id=_digest("triplet"),
        dataset_id=record.dataset_id,
        fold=record.fold_index,
        test_range=record.test_range,
        ledger_record=record,
    )
    run = SimpleNamespace(
        evidence=object(),
        decision=object(),
        access_records=(stage_a_record,),
    )

    final = StageAZeroShotArtifactPublisher(tmp_path).publish_sealed_test(run)

    assert sorted(path.name for path in final.iterdir()) == [
        "access-records.json",
        "decision.json",
        "evidence.json",
    ]
    payload = json.loads((final / "access-records.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "stage_a_sealed_test_access_records_v2"
    assert payload["records"][0]["access_digest"] == stage_a_record.access_digest
    assert payload["records"][0]["ledger_access_digest"] == record.access_digest
    assert payload["records"][0]["triplet_id"] == stage_a_record.triplet_id
    assert payload["records"][0]["test_range"] == [100, 120]


def test_failed_publication_removes_staging_and_keeps_final_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_first_file(path: Path, _evidence: object) -> None:
        _write_marker(path, "partial")
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(
        artifacts_module,
        "write_stage_a_evaluation_evidence",
        fail_after_first_file,
    )
    run = SimpleNamespace(evidence=object(), selection=object())

    with pytest.raises(RuntimeError, match="injected writer failure"):
        StageAZeroShotArtifactPublisher(tmp_path).publish_validation(run)

    assert not (tmp_path / "validation").exists()
    assert not list(tmp_path.glob(".validation-*"))
    assert not (tmp_path / ".validation.lock").exists()


def test_existing_phase_package_is_immutable(tmp_path: Path) -> None:
    final = tmp_path / "validation"
    final.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        StageAZeroShotArtifactPublisher(tmp_path).publish_validation(
            SimpleNamespace(evidence=object(), selection=object())
        )

    assert final.is_dir()


def test_foreign_publication_lock_is_not_removed(tmp_path: Path) -> None:
    lock = tmp_path / ".validation.lock"
    lock.write_text("other process", encoding="utf-8")

    with pytest.raises(FileExistsError):
        StageAZeroShotArtifactPublisher(tmp_path).publish_validation(
            SimpleNamespace(evidence=object(), selection=object())
        )

    assert lock.read_text(encoding="utf-8") == "other process"
