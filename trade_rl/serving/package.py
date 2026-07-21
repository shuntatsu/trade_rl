"""Package a verified selected-final training run into a serving bundle."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.data.metadata_promotion import (
    METADATA_PROMOTION_FILE_NAME,
    load_metadata_promotion_evidence,
)
from trade_rl.domain.common import require_sha256
from trade_rl.domain.selection import PolicyMode
from trade_rl.evaluation.confirmation import load_confirmation_evidence
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    write_serving_bundle_manifest,
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def package_selected_training_run(
    *,
    training_root: Path,
    confirmation_path: Path,
    output_root: Path,
    signal_digest: str,
    selection_digest: str,
    trusted_confirmation_keys: Mapping[str, PublicVerificationKey],
    trusted_now: datetime,
) -> ServingBundleManifest:
    """Validate and copy a selected-final run into an immutable bundle directory."""

    require_sha256(signal_digest, field="signal_digest")
    require_sha256(selection_digest, field="selection_digest")
    manifest = validate_training_run_directory(training_root)
    if manifest.run_kind != "research_selected_final":
        raise ValueError("only selected-final training runs can be packaged")
    if any(
        value is None
        for value in (
            manifest.selection_proposal_digest,
            manifest.selection_authorization_digest,
            manifest.walk_forward_run_digest,
            manifest.gate_evidence_digest,
        )
    ):
        raise ValueError("selected-final training manifest lacks authorization chain")
    metadata_promotion = load_metadata_promotion_evidence(
        training_root / METADATA_PROMOTION_FILE_NAME
    )
    if metadata_promotion.dataset_id != manifest.dataset_id:
        raise ValueError("metadata promotion dataset identity mismatch")
    metadata_promotion.require_promotable()

    ensemble_raw = _mapping(
        json.loads((training_root / "ensemble.json").read_text(encoding="utf-8")),
        field="ensemble",
    )
    ensemble_digest = _string(ensemble_raw.get("digest"), field="ensemble.digest")
    if ensemble_digest != manifest.ensemble_digest:
        raise ValueError("training manifest ensemble digest mismatch")
    if (
        _string(ensemble_raw.get("dataset_id"), field="ensemble.dataset_id")
        != manifest.dataset_id
    ):
        raise ValueError("training manifest dataset identity mismatch")
    if (
        _string(
            ensemble_raw.get("environment_digest"), field="ensemble.environment_digest"
        )
        != manifest.environment_digest
    ):
        raise ValueError("training manifest environment identity mismatch")

    confirmation = load_confirmation_evidence(confirmation_path)
    confirmation.verify(
        trusted_confirmation_keys,
        expected_required_after=manifest.completed_at,
        trusted_now=trusted_now,
    )
    if confirmation.training_run_digest != manifest.digest:
        raise ValueError("confirmation training run identity mismatch")
    if confirmation.dataset_id != manifest.dataset_id:
        raise ValueError("confirmation dataset identity mismatch")
    if confirmation.environment_digest != manifest.environment_digest:
        raise ValueError("confirmation environment identity mismatch")
    if confirmation.policy_digest != ensemble_digest:
        raise ValueError("confirmation policy identity mismatch")
    if confirmation.days < 30.0:
        raise ValueError("confirmation evidence must cover at least 30 days")
    if confirmation.total_return <= 0.0:
        raise ValueError("confirmation evidence must have positive total return")
    if confirmation.maximum_drawdown > 0.20:
        raise ValueError("confirmation drawdown exceeds release packaging limit")

    if output_root.exists():
        raise FileExistsError("serving bundle output already exists")
    stage = output_root.with_name(f".{output_root.name}.staging")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        artifact_paths: list[str] = []
        for item in manifest.files:
            source = training_root / item.path
            destination = stage / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            artifact_paths.append(item.path)
        shutil.copy2(training_root / "run.json", stage / "training-run.json")
        artifact_paths.append("training-run.json")
        shutil.copy2(confirmation_path, stage / "confirmation-evidence.json")
        artifact_paths.append("confirmation-evidence.json")

        action_names_raw = ensemble_raw.get("action_names")
        if not isinstance(action_names_raw, list) or any(
            not isinstance(item, str) for item in action_names_raw
        ):
            raise ValueError("ensemble.action_names must be a list of strings")
        created_at_raw = _string(
            ensemble_raw.get("created_at"), field="ensemble.created_at"
        )
        bundle_manifest = ServingBundleManifest.build(
            root=stage,
            dataset_id=manifest.dataset_id,
            action_schema=_string(
                ensemble_raw.get("action_schema"), field="ensemble.action_schema"
            ),
            observation_schema=_string(
                ensemble_raw.get("observation_schema"),
                field="ensemble.observation_schema",
            ),
            observation_size=_integer(
                ensemble_raw.get("observation_size"),
                field="ensemble.observation_size",
            ),
            environment_digest=manifest.environment_digest,
            initial_capital=_number(
                ensemble_raw.get("initial_capital"),
                field="ensemble.initial_capital",
            ),
            policy_mode=PolicyMode.RESIDUAL_POLICY,
            policy_digest=ensemble_digest,
            signal_digest=signal_digest,
            selection_digest=selection_digest,
            artifact_paths=tuple(sorted(artifact_paths)),
            created_at=datetime.fromisoformat(created_at_raw.replace("Z", "+00:00")),
            action_size=_integer(
                ensemble_raw.get("action_size"), field="ensemble.action_size"
            ),
            action_names=tuple(action_names_raw),
            action_spec_digest=_string(
                ensemble_raw.get("action_spec_digest"),
                field="ensemble.action_spec_digest",
            ),
            alpha_artifact_digest=_optional_string(
                ensemble_raw.get("alpha_artifact_digest"),
                field="ensemble.alpha_artifact_digest",
            ),
            factor_artifact_digest=_optional_string(
                ensemble_raw.get("factor_artifact_digest"),
                field="ensemble.factor_artifact_digest",
            ),
            normalizer_digest=_optional_string(
                ensemble_raw.get("normalizer_digest"),
                field="ensemble.normalizer_digest",
            ),
            training_run_digest=manifest.digest,
            run_kind=manifest.run_kind,
            selection_proposal_digest=manifest.selection_proposal_digest,
            selection_authorization_digest=manifest.selection_authorization_digest,
            walk_forward_run_digest=manifest.walk_forward_run_digest,
            gate_evidence_digest=manifest.gate_evidence_digest,
            confirmation_evidence_digest=confirmation.evidence_digest,
        )
        write_serving_bundle_manifest(stage, bundle_manifest)
        os.replace(stage, output_root)
        return bundle_manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["package_selected_training_run"]
