from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.run_manifest import TrainingRunManifest, write_training_run_manifest
from trade_rl.studio.evidence import inspect_run_evidence


def build_run(root: Path, *, selected_final: bool = False) -> Path:
    root.mkdir(parents=True)
    files = {
        "dataset-reference.json": {"dataset_id": "a" * 64, "artifact_digest": "1" * 64},
        "training-config.json": {"training": {"algorithm": "ppo"}},
        "ensemble.json": {"digest": "c" * 64},
    }
    if selected_final:
        files["selection-proposal.json"] = {"digest": "2" * 64}
        files["selection-authorization.json"] = {"authorization_digest": "3" * 64}
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest = TrainingRunManifest.build(
        root=root,
        run_id=root.name,
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=tuple(sorted(files)),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 19, 12, 5, tzinfo=UTC),
        run_kind="research_selected_final" if selected_final else "research_exploratory",
        selection_proposal_digest="2" * 64 if selected_final else None,
        selection_authorization_digest="3" * 64 if selected_final else None,
        walk_forward_run_digest="4" * 64 if selected_final else None,
        gate_evidence_digest="5" * 64 if selected_final else None,
    )
    write_training_run_manifest(root, manifest)
    return root


def test_exploratory_evidence_marks_authorization_chain_optional(tmp_path: Path) -> None:
    root = build_run(tmp_path / "run-exploratory")

    report = inspect_run_evidence(root)

    nodes = {node.key: node for node in report.nodes}
    assert report.status == "VALID"
    assert nodes["run_manifest"].status == "VERIFIED"
    assert nodes["dataset_reference"].status == "VERIFIED"
    assert nodes["selection_proposal"].required is False
    assert nodes["selection_proposal"].status == "ABSENT"
    assert report.files.status == "VERIFIED"
    assert report.files.declared_count == 3


def test_selected_final_evidence_requires_complete_bound_chain(tmp_path: Path) -> None:
    root = build_run(tmp_path / "run-selected", selected_final=True)

    report = inspect_run_evidence(root)

    nodes = {node.key: node for node in report.nodes}
    assert nodes["selection_proposal"].required is True
    assert nodes["selection_proposal"].status == "VERIFIED"
    assert nodes["selection_authorization"].status == "VERIFIED"
    assert nodes["walk_forward"].status == "PRESENT"
    assert nodes["gate_evidence"].status == "PRESENT"
    assert nodes["confirmation_evidence"].status == "ABSENT"


def test_evidence_preserves_manifest_validation_failure(tmp_path: Path) -> None:
    root = build_run(tmp_path / "run-invalid")
    (root / "ensemble.json").write_text('{"digest":"tampered"}', encoding="utf-8")

    report = inspect_run_evidence(root)

    nodes = {node.key: node for node in report.nodes}
    assert report.status == "INVALID"
    assert report.validation_error is not None
    assert "mismatch" in report.validation_error
    assert nodes["run_manifest"].status == "INVALID"
    assert report.files.status == "INVALID"
