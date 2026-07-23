from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
from trade_rl.evaluation.paper_reconciliation import (
    PAPER_RECONCILIATION_FILE_NAME,
    PaperReconciliationEvidence,
    write_paper_reconciliation_evidence,
)
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.offline_signing import public_key_bytes
from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.serving.package import package_selected_training_run
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    write_execution_evidence,
)

COMPLETED = datetime(2026, 7, 1, tzinfo=UTC)
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x44" * 32)
PUBLIC_KEY = PublicVerificationKey(
    key_id="confirmation-key",
    public_key=public_key_bytes(PRIVATE_KEY),
    purpose="fresh-confirmation",
    valid_from=COMPLETED - timedelta(days=1),
    valid_until=COMPLETED + timedelta(days=365),
)


def _training_run(
    root: Path,
    *,
    run_kind: str,
    execution_path_mode: str = "conservative",
    execution_policy_digest: str | None = None,
) -> TrainingRunManifest:
    root.mkdir()
    ensemble = {
        "action_names": ["fast", "slow"],
        "action_schema": "portfolio_action_v3",
        "action_size": 2,
        "action_spec_digest": "a" * 64,
        "alpha_artifact_digest": None,
        "created_at": COMPLETED.isoformat(),
        "dataset_id": "b" * 64,
        "digest": "c" * 64,
        "environment_digest": "d" * 64,
        "factor_artifact_digest": None,
        "initial_capital": 100_000.0,
        "normalizer_digest": None,
        "observation_schema": "portfolio_observation_v3",
        "observation_size": 8,
    }
    (root / "ensemble.json").write_text(json.dumps(ensemble), encoding="utf-8")
    (root / "policy-loader.json").write_text("{}", encoding="utf-8")
    (root / "policy.zip").write_bytes(b"policy")
    execution_cost = ExecutionCostConfig(path_mode="conservative")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "environment": {"execution_cost": asdict(execution_cost)},
                "schema_version": "training_environment_v2",
            }
        ),
        encoding="utf-8",
    )
    write_execution_evidence(
        root / "execution-evidence.json",
        ExecutionEvidence(
            dataset_id="b" * 64,
            execution_policy_digest=(
                execution_cost.execution_policy_digest
                if execution_policy_digest is None
                else execution_policy_digest
            ),
            path_mode=execution_path_mode,
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
            trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
            order_event_count=4,
            complete_order_evidence=True,
            sensitivity_path_modes=("conservative",),
        ),
    )
    write_metadata_promotion_evidence(
        root / "metadata-promotion.json",
        MetadataPromotionEvidence(
            dataset_id="b" * 64,
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
    selected = run_kind == "research_selected_final"
    manifest = TrainingRunManifest.build(
        root=root,
        run_id="selected",
        dataset_id="b" * 64,
        environment_digest="d" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="e" * 64,
        provenance_digest="f" * 64,
        artifact_paths=(
            "ensemble.json",
            "environment.json",
            "execution-evidence.json",
            "metadata-promotion.json",
            "policy-loader.json",
            "policy.zip",
        ),
        created_at=COMPLETED - timedelta(hours=1),
        completed_at=COMPLETED,
        run_kind=run_kind,
        selection_proposal_digest=("1" * 64 if selected else None),
        selection_authorization_digest=("2" * 64 if selected else None),
        walk_forward_run_digest=("3" * 64 if selected else None),
        gate_evidence_digest=("4" * 64 if selected else None),
    )
    write_training_run_manifest(root, manifest)
    return manifest


def _paper_reconciliation(
    path: Path,
    manifest: TrainingRunManifest,
    **overrides: object,
) -> PaperReconciliationEvidence:
    values: dict[str, Any] = {
        "dataset_id": manifest.dataset_id,
        "environment_digest": manifest.environment_digest,
        "policy_digest": manifest.ensemble_digest,
        "training_run_digest": manifest.digest,
        "start_time": manifest.completed_at,
        "end_time": manifest.completed_at + timedelta(days=30),
        "created_at": manifest.completed_at + timedelta(days=30, minutes=1),
        "order_log_digest": "7" * 64,
        "fill_log_digest": "8" * 64,
        "submitted_order_count": 120,
        "terminal_order_count": 120,
        "observed_fill_count": 100,
        "matched_fill_count": 100,
        "unknown_order_fill_count": 0,
        "duplicate_fill_count": 0,
        "open_order_count": 0,
        "maximum_position_notional_difference_fraction": 0.0,
        "maximum_cash_difference_fraction": 0.0,
        "maximum_equity_difference_fraction": 0.0,
        "position_notional_tolerance_fraction": 1e-8,
        "cash_tolerance_fraction": 1e-8,
        "equity_tolerance_fraction": 1e-8,
    }
    values.update(overrides)
    evidence = PaperReconciliationEvidence.create(**values)
    write_paper_reconciliation_evidence(path, evidence)
    return evidence


def _confirmation(
    path: Path,
    manifest: TrainingRunManifest,
    *,
    write_reconciliation: bool = True,
    reconciliation_digest: str | None = None,
    reconciliation_overrides: dict[str, object] | None = None,
) -> None:
    reconciliation = None
    if write_reconciliation:
        reconciliation = _paper_reconciliation(
            path.with_name(PAPER_RECONCILIATION_FILE_NAME),
            manifest,
            **(reconciliation_overrides or {}),
        )
    resolved_reconciliation_digest = reconciliation_digest or (
        reconciliation.evidence_digest if reconciliation is not None else "9" * 64
    )
    evidence = create_fresh_confirmation_evidence(
        dataset_id=manifest.dataset_id,
        environment_digest=manifest.environment_digest,
        policy_digest=manifest.ensemble_digest,
        training_run_digest=manifest.digest,
        git_commit="5" * 40,
        dependency_digest="6" * 64,
        required_after=manifest.completed_at,
        start_time=manifest.completed_at,
        end_time=manifest.completed_at + timedelta(days=30),
        returns=(0.001,) * 30,
        return_period_hours=24.0,
        order_log_digest="7" * 64,
        fill_log_digest="8" * 64,
        reconciliation_digest=resolved_reconciliation_digest,
        created_at=manifest.completed_at + timedelta(days=30),
        key_id=PUBLIC_KEY.key_id,
        private_key=PRIVATE_KEY,
    )
    write_confirmation_evidence(path, evidence)


def test_package_selected_final_binds_training_and_confirmation(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(training_root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)

    manifest = package_selected_training_run(
        training_root=training_root,
        confirmation_path=confirmation_path,
        output_root=tmp_path / "bundle",
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=training.completed_at + timedelta(days=30),
    )
    loaded = load_serving_bundle(tmp_path / "bundle")
    assert manifest.training_run_digest == training.digest
    assert manifest.confirmation_evidence_digest is not None
    assert loaded.manifest == manifest
    assert (loaded.root / "training-run.json").is_file()
    assert (loaded.root / "confirmation-evidence.json").is_file()
    assert (loaded.root / PAPER_RECONCILIATION_FILE_NAME).is_file()


def test_package_rejects_missing_paper_reconciliation(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(training_root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training, write_reconciliation=False)

    with pytest.raises((FileNotFoundError, ValueError), match="reconciliation"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )


def test_package_rejects_reconciliation_digest_mismatch(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(training_root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training, reconciliation_digest="9" * 64)

    with pytest.raises(ValueError, match="reconciliation digest"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )


def test_package_rejects_failed_paper_reconciliation(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(training_root, run_kind="research_selected_final")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(
        confirmation_path,
        training,
        reconciliation_overrides={
            "matched_fill_count": 99,
            "unknown_order_fill_count": 1,
        },
    )

    with pytest.raises(ValueError, match="did not pass"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )


def test_package_rejects_exploratory_training(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(training_root, run_kind="research_exploratory")
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    with pytest.raises(ValueError, match="selected-final"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )


def test_package_rejects_non_conservative_execution_evidence(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(
        training_root,
        run_kind="research_selected_final",
        execution_path_mode="optimistic",
    )
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    with pytest.raises(ValueError, match="conservative"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )


def test_package_rejects_execution_policy_digest_mismatch(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training = _training_run(
        training_root,
        run_kind="research_selected_final",
        execution_policy_digest="f" * 64,
    )
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    with pytest.raises(ValueError, match="policy digest"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )
