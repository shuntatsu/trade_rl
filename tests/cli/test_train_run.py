from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from trade_rl.cli import extended
from trade_rl.workflows.training_run import TrainingRunResult


def test_train_run_command_emits_published_run_json(
    monkeypatch, tmp_path: Path
) -> None:
    published = tmp_path / "runs" / "run-001"

    def fake_execute(**kwargs):
        assert kwargs == {
            "config_path": Path("config.json"),
            "dataset_path": Path("dataset"),
            "store_root": Path("artifacts"),
            "run_id": "run-001",
            "selection_proposal_path": None,
            "selection_authorization_path": None,
            "selection_public_keys_path": None,
            "require_selection_authorization": False,
            "execution_evidence_path": None,
        }
        return TrainingRunResult(
            run_id="run-001",
            status="published",
            path=published,
            run_digest="a" * 64,
            policy_digest="b" * 64,
            dataset_id="c" * 64,
        )

    monkeypatch.setattr(extended, "execute_training_run", fake_execute)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "train",
            "run",
            "--config",
            "config.json",
            "--dataset",
            "dataset",
            "--output",
            "artifacts",
            "--run-id",
            "run-001",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload == {
        "artifact_path": str(published),
        "dataset_id": "c" * 64,
        "policy_digest": "b" * 64,
        "production_status": "NO-GO",
        "run_digest": "a" * 64,
        "run_id": "run-001",
        "run_kind": "research_exploratory",
        "schema": "training_run_result_v1",
        "selection_authorization_digest": None,
        "selection_proposal_digest": None,
        "status": "published",
    }


def test_train_run_command_returns_structured_failure(monkeypatch) -> None:
    def fail(**kwargs):
        del kwargs
        raise ValueError("dataset digest mismatch")

    monkeypatch.setattr(extended, "execute_training_run", fail)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "train",
            "run",
            "--config",
            "config.json",
            "--dataset",
            "dataset",
            "--output",
            "artifacts",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload == {
        "error": "dataset digest mismatch",
        "error_type": "ValueError",
        "production_status": "NO-GO",
        "schema": "training_run_error_v1",
        "status": "failed",
    }
