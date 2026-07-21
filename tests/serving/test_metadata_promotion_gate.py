from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.serving.test_package import COMPLETED, PUBLIC_KEY, _confirmation
from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.serving.package import package_selected_training_run


def _selected_training_run(root: Path, *, metadata_mode: str) -> TrainingRunManifest:
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
    historical = metadata_mode == "historical_signed"
    promotion = {
        "authentication": "ed25519" if historical else "none",
        "coverage_application": (
            "effective-dated-full-interval" if historical else "static-full-interval"
        ),
        "dataset_id": "b" * 64,
        "limitations": [] if historical else ["static-full-interval"],
        "metadata_evidence_digest": "6" * 64,
        "mode": metadata_mode,
        "point_in_time": historical,
        "promotable": historical,
        "schema_version": "metadata_promotion_evidence_v1",
        "source_payload_digest": "7" * 64,
    }
    (root / "metadata-promotion.json").write_text(
        json.dumps(promotion), encoding="utf-8"
    )
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
            "metadata-promotion.json",
            "policy-loader.json",
            "policy.zip",
        ),
        created_at=COMPLETED,
        completed_at=COMPLETED,
        run_kind="research_selected_final",
        selection_proposal_digest="1" * 64,
        selection_authorization_digest="2" * 64,
        walk_forward_run_digest="3" * 64,
        gate_evidence_digest="4" * 64,
    )
    write_training_run_manifest(root, manifest)
    return manifest


@pytest.mark.parametrize("metadata_mode", ["frozen_snapshot", "conservative_static"])
def test_package_rejects_nonhistorical_metadata_promotion(
    tmp_path: Path,
    metadata_mode: str,
) -> None:
    training_root = tmp_path / "training"
    training = _selected_training_run(training_root, metadata_mode=metadata_mode)
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)

    with pytest.raises(ValueError, match="historical_signed"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )
