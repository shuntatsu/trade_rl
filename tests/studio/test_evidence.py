from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
    write_execution_promotion_report,
)
from trade_rl.studio.evidence import inspect_run_evidence
from trade_rl.studio.resource_ids import resource_id
from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    SelectionProposal,
)


def _runtime_report():
    return build_execution_promotion_report(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=ExecutionPromotionEvidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=False,
            determinism_passed=False,
            performance_approved=False,
        ),
    )


def build_run(
    root: Path,
    *,
    selected_final: bool = False,
    proposal_dataset_id: str = "a" * 64,
    runtime_promotion: bool = False,
    runtime_promotion_digest_only: bool = False,
    unbound_runtime_sidecar: bool = False,
) -> tuple[Path, str]:
    root.mkdir(parents=True)
    files: dict[str, object] = {
        "dataset-reference.json": {"dataset_id": "a" * 64, "artifact_digest": "1" * 64},
        "training-config.json": {"training": {"algorithm": "ppo"}},
        "ensemble.json": {"digest": "c" * 64},
    }
    report = (
        _runtime_report()
        if runtime_promotion or runtime_promotion_digest_only or unbound_runtime_sidecar
        else None
    )
    proposal = None
    authorization = None
    if selected_final:
        proposal = SelectionProposal.create(
            walk_forward_run_digest="4" * 64,
            gate_evidence_digest="5" * 64,
            execution_sensitivity_digest="6" * 64,
            dataset_id=proposal_dataset_id,
            selected_configuration="candidate-a",
            candidate_config_digest="7" * 64,
            seeds=(0, 1),
            git_commit="8" * 40,
            dependency_digest="9" * 64,
            resume_checkpoint_digests=(),
            runtime_promotion_report_digest=(
                None if report is None or unbound_runtime_sidecar else report.digest
            ),
        )
        approved = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
        authorization = SelectionAuthorization(
            proposal_digest=proposal.digest,
            approver="researcher",
            approved_at=approved,
            expires_at=approved + timedelta(days=1),
            key_id="test-key",
            signature="test-signature",
        )
        files["selection-proposal.json"] = proposal.to_mapping()
        files["selection-authorization.json"] = authorization.to_mapping()
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload, default=str), encoding="utf-8")
    artifact_paths = list(files)
    if report is not None and not runtime_promotion_digest_only:
        write_execution_promotion_report(root / "runtime-promotion-report.json", report)
        artifact_paths.append("runtime-promotion-report.json")
    manifest = TrainingRunManifest.build(
        root=root,
        run_id=root.name,
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=tuple(sorted(artifact_paths)),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 19, 12, 5, tzinfo=UTC),
        run_kind="research_selected_final"
        if selected_final
        else "research_exploratory",
        selection_proposal_digest=None if proposal is None else proposal.digest,
        selection_authorization_digest=(
            None if authorization is None else authorization.authorization_digest
        ),
        walk_forward_run_digest="4" * 64 if selected_final else None,
        gate_evidence_digest="5" * 64 if selected_final else None,
    )
    write_training_run_manifest(root, manifest)
    return root, resource_id("run", root.name, manifest.digest)


def test_exploratory_evidence_marks_authorization_optional(tmp_path: Path) -> None:
    root, identity = build_run(tmp_path / "run-exploratory")

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    assert report.run_resource_id == identity
    assert report.status == "VALID"
    assert nodes["run_manifest"].status == "VERIFIED"
    assert nodes["selection_proposal"].status == "ABSENT"


def test_selected_final_evidence_verifies_internal_bindings(tmp_path: Path) -> None:
    root, identity = build_run(tmp_path / "run-selected", selected_final=True)

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    assert report.status == "VALID"
    assert nodes["selection_proposal"].status == "VERIFIED"
    assert nodes["selection_authorization"].status == "VERIFIED"
    assert nodes["walk_forward"].status == "VERIFIED"
    assert nodes["gate_evidence"].status == "VERIFIED"
    assert nodes["runtime_promotion"].status == "ABSENT"
    assert not nodes["runtime_promotion"].required


def test_selected_final_evidence_verifies_runtime_promotion_binding(
    tmp_path: Path,
) -> None:
    root, identity = build_run(
        tmp_path / "run-runtime-promotion",
        selected_final=True,
        runtime_promotion=True,
    )

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    runtime = nodes["runtime_promotion"]
    assert report.status == "VALID"
    assert runtime.status == "VERIFIED"
    assert runtime.required
    assert runtime.path == "runtime-promotion-report.json"
    assert runtime.digest is not None
    assert "runtime authority" not in runtime.detail.lower()


def test_selected_final_evidence_rejects_missing_bound_runtime_promotion(
    tmp_path: Path,
) -> None:
    root, identity = build_run(
        tmp_path / "run-runtime-missing",
        selected_final=True,
        runtime_promotion_digest_only=True,
    )

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    runtime = nodes["runtime_promotion"]
    assert report.status == "INVALID"
    assert runtime.status == "INVALID"
    assert runtime.required
    assert "missing" in runtime.detail


def test_selected_final_evidence_rejects_unbound_runtime_sidecar(
    tmp_path: Path,
) -> None:
    root, identity = build_run(
        tmp_path / "run-runtime-unbound",
        selected_final=True,
        unbound_runtime_sidecar=True,
    )

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    runtime = nodes["runtime_promotion"]
    assert report.status == "INVALID"
    assert runtime.status == "INVALID"
    assert runtime.required
    assert "does not authorize" in runtime.detail


def test_selected_final_evidence_rejects_proposal_binding_mismatch(
    tmp_path: Path,
) -> None:
    root, identity = build_run(
        tmp_path / "run-mismatch",
        selected_final=True,
        proposal_dataset_id="f" * 64,
    )

    report = inspect_run_evidence(root, run_resource_id=identity)

    nodes = {node.key: node for node in report.nodes}
    assert report.status == "INVALID"
    assert nodes["selection_proposal"].status == "INVALID"
    assert "dataset" in nodes["selection_proposal"].detail


def test_evidence_preserves_manifest_validation_failure(tmp_path: Path) -> None:
    root, identity = build_run(tmp_path / "run-invalid")
    (root / "ensemble.json").write_text('{"digest":"tampered"}', encoding="utf-8")

    report = inspect_run_evidence(root, run_resource_id=identity)

    assert report.status == "INVALID"
    assert report.validation_error is not None
    assert report.files.status == "INVALID"
