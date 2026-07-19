from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
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
    pointer = {
        "bundle_digest": manifest.bundle_digest,
        "path": f"versions/{'f' * 64}",
        "schema": "serving_registry_pointer_v1",
    }
    (root / "serving" / "active.json").write_text(json.dumps(pointer), encoding="utf-8")
    return manifest


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
        "snapshot_digest": "9" * 64,
    }
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
