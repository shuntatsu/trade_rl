from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.serving.test_package import (
    COMPLETED,
    PRIVATE_KEY,
    PUBLIC_KEY,
    _confirmation,
    _training_run,
)
from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.data.metadata_promotion import (
    MetadataPromotionEvidence,
    write_metadata_promotion_evidence,
)
from trade_rl.evaluation.confirmation import write_confirmation_evidence
from trade_rl.evaluation.offline_confirmation import create_fresh_confirmation_evidence
from trade_rl.release.offline_signing import public_key_bytes
from trade_rl.serving import package as serving_package
from trade_rl.serving.package import package_selected_training_run
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    write_execution_evidence,
)


def _rebuild_manifest(
    root: Path,
    source: TrainingRunManifest,
) -> TrainingRunManifest:
    manifest = TrainingRunManifest.build(
        root=root,
        run_id=source.run_id,
        dataset_id=source.dataset_id,
        environment_digest=source.environment_digest,
        ensemble_digest=source.ensemble_digest,
        training_config_digest=source.training_config_digest,
        provenance_digest=source.provenance_digest,
        artifact_paths=tuple(item.path for item in source.files),
        created_at=source.created_at,
        completed_at=source.completed_at,
        run_kind=source.run_kind,
        selection_proposal_digest=source.selection_proposal_digest,
        selection_authorization_digest=source.selection_authorization_digest,
        walk_forward_run_digest=source.walk_forward_run_digest,
        gate_evidence_digest=source.gate_evidence_digest,
    )
    write_training_run_manifest(root, manifest)
    return manifest


def _write_confirmation(
    path: Path,
    manifest: TrainingRunManifest,
    **overrides: object,
) -> None:
    values: dict[str, object] = {
        "dataset_id": manifest.dataset_id,
        "environment_digest": manifest.environment_digest,
        "policy_digest": manifest.ensemble_digest,
        "training_run_digest": manifest.digest,
        "git_commit": "5" * 40,
        "dependency_digest": "6" * 64,
        "required_after": manifest.completed_at,
        "start_time": manifest.completed_at,
        "end_time": manifest.completed_at + timedelta(days=30),
        "returns": (0.001,) * 30,
        "return_period_hours": 24.0,
        "order_log_digest": "7" * 64,
        "fill_log_digest": "8" * 64,
        "reconciliation_digest": "9" * 64,
        "created_at": manifest.completed_at + timedelta(days=30),
        "key_id": PUBLIC_KEY.key_id,
        "private_key": PRIVATE_KEY,
    }
    values.update(overrides)
    evidence = create_fresh_confirmation_evidence(**values)  # type: ignore[arg-type]
    write_confirmation_evidence(path, evidence)


def _package(
    root: Path,
    manifest: TrainingRunManifest,
    confirmation_path: Path,
    output: Path,
) -> None:
    package_selected_training_run(
        training_root=root,
        confirmation_path=confirmation_path,
        output_root=output,
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=manifest.completed_at + timedelta(days=30),
    )


@pytest.mark.parametrize(
    ("function", "value", "message"),
    [
        (serving_package._mapping, [], "field must be a mapping"),
        (serving_package._string, 1, "field must be a string"),
        (serving_package._integer, True, "field must be an integer"),
        (serving_package._number, True, "field must be numeric"),
    ],
)
def test_package_primitive_decoders_reject_wrong_types(
    function: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(value, field="field")  # type: ignore[operator]


def test_package_optional_string_accepts_none_and_rejects_other_types() -> None:
    assert serving_package._optional_string(None, field="field") is None
    with pytest.raises(ValueError, match="field must be a string"):
        serving_package._optional_string(1, field="field")


def test_package_rejects_metadata_dataset_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    metadata_path = root / "metadata-promotion.json"
    metadata_path.unlink()
    write_metadata_promotion_evidence(
        metadata_path,
        MetadataPromotionEvidence(
            dataset_id="0" * 64,
            mode="historical_signed",
            metadata_evidence_digest="6" * 64,
            source_payload_digest="7" * 64,
            point_in_time=True,
            authentication="ed25519",
            coverage_application="effective-dated-full-interval",
            limitations=(),
            promotable=True,
        ),
    )
    manifest = _rebuild_manifest(root, manifest)
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)

    with pytest.raises(ValueError, match="metadata promotion dataset identity"):
        _package(root, manifest, confirmation_path, tmp_path / "bundle")


def test_package_rejects_execution_evidence_dataset_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    execution_path = root / "execution-evidence.json"
    execution_path.unlink()
    write_execution_evidence(
        execution_path,
        ExecutionEvidence(
            dataset_id="0" * 64,
            execution_policy_digest=serving_package._execution_cost(
                root
            ).execution_policy_digest,
            path_mode="conservative",
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
            trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
            order_event_count=4,
            complete_order_evidence=True,
            sensitivity_path_modes=("conservative",),
        ),
    )
    manifest = _rebuild_manifest(root, manifest)
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)

    with pytest.raises(ValueError, match="execution evidence dataset identity"):
        _package(root, manifest, confirmation_path, tmp_path / "bundle")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("digest", "0" * 64, "ensemble digest"),
        ("dataset_id", "0" * 64, "dataset identity"),
        ("environment_digest", "0" * 64, "environment identity"),
    ],
)
def test_package_rejects_ensemble_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    ensemble_path = root / "ensemble.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble[field] = value
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")
    manifest = _rebuild_manifest(root, manifest)
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)

    with pytest.raises(ValueError, match=message):
        _package(root, manifest, confirmation_path, tmp_path / "bundle")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_run_digest", "0" * 64, "training run identity"),
        ("dataset_id", "0" * 64, "dataset identity"),
        ("environment_digest", "0" * 64, "environment identity"),
        ("policy_digest", "0" * 64, "policy identity"),
    ],
)
def test_package_rejects_confirmation_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _write_confirmation(confirmation_path, manifest, **{field: value})

    with pytest.raises(ValueError, match=message):
        _package(root, manifest, confirmation_path, tmp_path / "bundle")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"order_log_digest": "0" * 64}, "order log digest"),
        ({"fill_log_digest": "0" * 64}, "fill log digest"),
        ({"training_run_digest": "0" * 64}, "training run identity"),
        ({"environment_digest": "0" * 64}, "environment identity"),
        ({"policy_digest": "0" * 64}, "policy identity"),
        ({"start_time": COMPLETED + timedelta(hours=1)}, "start time"),
        ({"end_time": COMPLETED + timedelta(days=29)}, "end time"),
    ],
)
def test_package_rejects_reconciliation_binding_mismatch(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(
        confirmation_path,
        manifest,
        reconciliation_overrides=overrides,
    )

    with pytest.raises(ValueError, match=message):
        _package(root, manifest, confirmation_path, tmp_path / "bundle")


def test_package_rejects_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _package(root, manifest, confirmation_path, output)


def test_package_removes_stale_stage_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)
    output = tmp_path / "bundle"
    stage = tmp_path / ".bundle.staging"
    stage.mkdir()
    (stage / "stale.txt").write_text("stale", encoding="utf-8")

    _package(root, manifest, confirmation_path, output)

    assert output.is_dir()
    assert not stage.exists()


def test_package_cleans_stage_after_manifest_input_failure(tmp_path: Path) -> None:
    root = tmp_path / "training"
    manifest = _training_run(root, run_kind="research_selected_final")
    ensemble_path = root / "ensemble.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble["action_names"] = ["valid", 1]
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")
    manifest = _rebuild_manifest(root, manifest)
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, manifest)
    output = tmp_path / "bundle"
    stage = tmp_path / ".bundle.staging"

    with pytest.raises(ValueError, match="action_names"):
        _package(root, manifest, confirmation_path, output)

    assert not output.exists()
    assert not stage.exists()


def test_public_key_fixture_remains_bound_to_expected_private_key() -> None:
    assert PUBLIC_KEY.public_key == public_key_bytes(PRIVATE_KEY)
