from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.selection_authorization import (
    SelectionProposal,
    write_selection_proposal,
)
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
    write_execution_promotion_report,
)
from trade_rl.studio.serving_monitor import inspect_serving
from trade_rl.studio.settings import StudioSettings


def settings_for(root: Path) -> StudioSettings:
    return StudioSettings(
        project_root=root,
        dataset_roots=(root / "datasets",),
        run_roots=(root / "research",),
        config_roots=(root / "configs",),
        job_root=root / "jobs",
        serving_root=root / "serving",
        paper_snapshot_path=root / "paper-inference.json",
    )


def _publish_pointer(root: Path, manifest: ServingBundleManifest) -> None:
    pointer = {
        "bundle_digest": manifest.bundle_digest,
        "path": f"versions/{'f' * 64}",
        "schema": "serving_registry_pointer_v1",
    }
    (root / "serving" / "active.json").write_text(json.dumps(pointer), encoding="utf-8")


def build_active_bundle(root: Path) -> ServingBundleManifest:
    bundle_root = root / "serving" / "versions" / ("f" * 64)
    bundle_root.mkdir(parents=True)
    (bundle_root / "baseline.json").write_text('{"strategy":"flat"}', encoding="utf-8")
    manifest = ServingBundleManifest.build(
        root=bundle_root,
        dataset_id="a" * 64,
        action_schema="target_weights_v1",
        observation_schema="observation_v4",
        observation_size=8,
        environment_digest="b" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.BASELINE_ONLY,
        policy_digest=None,
        signal_digest="c" * 64,
        selection_digest="d" * 64,
        artifact_paths=("baseline.json",),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        action_size=2,
        action_names=("BTCUSDT", "CASH"),
        action_spec_digest="e" * 64,
    )
    write_serving_bundle_manifest(bundle_root, manifest)
    _publish_pointer(root, manifest)
    return manifest


def _allowed_runtime_report():
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


def build_selected_final_bundle(
    root: Path,
    *,
    bind_runtime_report: bool,
    include_runtime_report: bool,
) -> tuple[ServingBundleManifest, SelectionProposal]:
    bundle_root = root / "serving" / "versions" / ("f" * 64)
    bundle_root.mkdir(parents=True)
    (bundle_root / "policy.bin").write_bytes(b"policy")
    report = _allowed_runtime_report()
    proposal = SelectionProposal.create(
        walk_forward_run_digest="1" * 64,
        gate_evidence_digest="2" * 64,
        execution_sensitivity_digest="3" * 64,
        dataset_id="a" * 64,
        selected_configuration="candidate-a",
        candidate_config_digest="4" * 64,
        seeds=(7, 11),
        git_commit="5" * 40,
        dependency_digest="6" * 64,
        resume_checkpoint_digests=(),
        runtime_promotion_report_digest=(
            report.digest if bind_runtime_report else None
        ),
    )
    write_selection_proposal(bundle_root / "selection-proposal.json", proposal)
    artifact_paths = ["policy.bin", "selection-proposal.json"]
    if include_runtime_report:
        write_execution_promotion_report(
            bundle_root / "runtime-promotion-report.json",
            report,
        )
        artifact_paths.append("runtime-promotion-report.json")
    manifest = ServingBundleManifest.build(
        root=bundle_root,
        dataset_id="a" * 64,
        action_schema="target_weights_v1",
        observation_schema="observation_v4",
        observation_size=8,
        environment_digest="b" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="7" * 64,
        signal_digest="c" * 64,
        selection_digest="d" * 64,
        artifact_paths=tuple(artifact_paths),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        action_size=2,
        action_names=("BTCUSDT", "CASH"),
        action_spec_digest="e" * 64,
        training_run_digest="8" * 64,
        run_kind="research_selected_final",
        selection_proposal_digest=proposal.digest,
        selection_authorization_digest="9" * 64,
        walk_forward_run_digest=proposal.walk_forward_run_digest,
        gate_evidence_digest=proposal.gate_evidence_digest,
        confirmation_evidence_digest="0" * 64,
    )
    write_serving_bundle_manifest(bundle_root, manifest)
    _publish_pointer(root, manifest)
    return manifest, proposal


def test_missing_serving_registry_is_idle(tmp_path: Path) -> None:
    report = inspect_serving(settings_for(tmp_path))

    assert report.state == "IDLE"
    assert report.active_bundle_digest is None
    assert report.validation_error is None
    assert report.production_status == "NO-GO"


def test_valid_active_bundle_reports_identity_and_optional_snapshot(
    tmp_path: Path,
) -> None:
    manifest = build_active_bundle(tmp_path)
    snapshot = {
        "schema_version": "studio_paper_inference_v1",
        "recorded_at": "2026-07-19T12:10:00+00:00",
        "bundle_digest": manifest.bundle_digest,
        "dataset_id": manifest.dataset_id,
        "decision_index": 42,
        "target_weights": {"BTCUSDT": 0.25, "CASH": 0.75},
        "latency_ms": 12.5,
    }
    snapshot["snapshot_digest"] = content_digest(snapshot)
    tmp_path.joinpath("paper-inference.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )

    report = inspect_serving(settings_for(tmp_path))

    assert report.state == "VALID"
    assert report.active_bundle_digest == manifest.bundle_digest
    assert report.dataset_id == manifest.dataset_id
    assert report.run_kind == "baseline_release"
    assert report.release_attestation_present is False
    assert report.paper_snapshot is not None
    assert report.paper_snapshot.decision_index == 42
    assert report.paper_snapshot.target_weights["BTCUSDT"] == 0.25
    assert all(check.status != "FAIL" for check in report.checks)


def test_selected_final_serving_reports_bound_runtime_promotion(
    tmp_path: Path,
) -> None:
    build_selected_final_bundle(
        tmp_path,
        bind_runtime_report=True,
        include_runtime_report=True,
    )

    report = inspect_serving(settings_for(tmp_path))

    runtime = next(check for check in report.checks if check.key == "runtime_promotion")
    assert report.state == "VALID"
    assert runtime.status == "PASS"
    assert "binding verified" in runtime.detail
    assert "authority remains external" in runtime.detail


def test_selected_final_serving_rejects_missing_bound_runtime_promotion(
    tmp_path: Path,
) -> None:
    build_selected_final_bundle(
        tmp_path,
        bind_runtime_report=True,
        include_runtime_report=False,
    )

    report = inspect_serving(settings_for(tmp_path))

    runtime = next(check for check in report.checks if check.key == "runtime_promotion")
    assert report.state == "INVALID"
    assert runtime.status == "FAIL"
    assert "missing" in runtime.detail


def test_selected_final_serving_rejects_unbound_runtime_promotion(
    tmp_path: Path,
) -> None:
    build_selected_final_bundle(
        tmp_path,
        bind_runtime_report=False,
        include_runtime_report=True,
    )

    report = inspect_serving(settings_for(tmp_path))

    runtime = next(check for check in report.checks if check.key == "runtime_promotion")
    assert report.state == "INVALID"
    assert runtime.status == "FAIL"
    assert "does not authorize" in runtime.detail


def test_invalid_pointer_or_bundle_fails_closed(tmp_path: Path) -> None:
    build_active_bundle(tmp_path)
    pointer_path = tmp_path / "serving" / "active.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["path"] = "../escape"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    report = inspect_serving(settings_for(tmp_path))

    assert report.state == "INVALID"
    assert report.validation_error is not None
    assert "escapes" in report.validation_error
    assert any(check.status == "FAIL" for check in report.checks)


def test_snapshot_digest_mismatch_is_reported_without_invalidating_bundle(
    tmp_path: Path,
) -> None:
    manifest = build_active_bundle(tmp_path)
    snapshot = {
        "schema_version": "studio_paper_inference_v1",
        "recorded_at": "2026-07-19T12:10:00+00:00",
        "bundle_digest": manifest.bundle_digest,
        "dataset_id": manifest.dataset_id,
        "decision_index": 42,
        "target_weights": {"BTCUSDT": 0.25, "CASH": 0.75},
        "latency_ms": 12.5,
        "snapshot_digest": "9" * 64,
    }
    tmp_path.joinpath("paper-inference.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )

    report = inspect_serving(settings_for(tmp_path))

    assert report.state == "VALID"
    assert report.paper_snapshot is None
    paper_check = next(
        check for check in report.checks if check.key == "paper_snapshot"
    )
    assert paper_check.status == "FAIL"
    assert "digest" in paper_check.detail
