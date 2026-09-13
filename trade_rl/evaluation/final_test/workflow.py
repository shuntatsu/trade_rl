"""Read-only Study gate and atomic publication for final-test authorization."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.contracts import StudyOutcome
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    ContractViolationError,
    InvalidExperimentStateError,
)
from trade_rl.evaluation.experiments.inspection import StudySnapshot, inspect_study
from trade_rl.evaluation.final_test.contracts import FinalEvaluationAuthorization

_ARTIFACT_SCHEMA = "final_evaluation_authorization_artifact_v1"
_ARTIFACT_NAME = "authorization.json"
_ARTIFACT_KEYS = frozenset({"schema_version", "authorization_digest", "authorization"})


def _winner_snapshot(study_root: str | Path) -> StudySnapshot:
    snapshot = inspect_study(study_root)
    freeze = snapshot.freeze
    if freeze is None or freeze.outcome is not StudyOutcome.WINNER:
        raise InvalidExperimentStateError(
            "final evaluation authorization requires a frozen WINNER Study"
        )
    if freeze.selected_evidence_digest is None or freeze.selected_strategy is None:
        raise ArtifactIntegrityError("WINNER Study freeze lacks selected evidence")
    if freeze.selected_evidence_digest not in snapshot.lineage_evidence_digests:
        raise ArtifactIntegrityError(
            "WINNER evidence is absent from accepted Study lineage"
        )
    return snapshot


def _authorization_from_study(
    snapshot: StudySnapshot,
    *,
    final_evaluation_start: str,
    final_evaluation_stop_exclusive: str,
    authorized_by: str,
    authorized_at: datetime,
) -> FinalEvaluationAuthorization:
    freeze = snapshot.freeze
    if freeze is None or freeze.outcome is not StudyOutcome.WINNER:
        raise InvalidExperimentStateError(
            "final evaluation authorization requires a frozen WINNER Study"
        )
    if freeze.selected_evidence_digest is None or freeze.selected_strategy is None:
        raise ArtifactIntegrityError("WINNER Study freeze lacks selected evidence")
    return FinalEvaluationAuthorization(
        study_digest=snapshot.plan.digest,
        study_freeze_digest=freeze.digest,
        winner_evidence_digest=freeze.selected_evidence_digest,
        winner_strategy=freeze.selected_strategy,
        development_evaluation_stop_exclusive=(
            snapshot.plan.baseline_config.evaluation_stop_exclusive
        ),
        final_evaluation_start=final_evaluation_start,
        final_evaluation_stop_exclusive=final_evaluation_stop_exclusive,
        authorized_by=authorized_by,
        authorized_at=authorized_at,
    )


def _artifact_payload(
    authorization: FinalEvaluationAuthorization,
) -> dict[str, object]:
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "authorization_digest": authorization.digest,
        "authorization": authorization.to_payload(),
    }


def _checked_new_root(output_root: str | Path) -> Path:
    root = Path(output_root)
    if root.name in {"", ".", ".."}:
        raise InvalidExperimentStateError("authorization output root is invalid")
    absolute = Path(os.path.abspath(root))
    if absolute.exists() or absolute.is_symlink():
        raise InvalidExperimentStateError("authorization output root already exists")
    parent = absolute.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ArtifactIntegrityError(
            "authorization output parent must be a regular directory"
        )
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError(
            "authorization output parent cannot be trusted"
        ) from error
    if resolved_parent != parent:
        raise ArtifactIntegrityError(
            "authorization output path must not traverse symlinks"
        )
    return absolute


def _publish_once(
    output_root: str | Path,
    authorization: FinalEvaluationAuthorization,
) -> Path:
    target = _checked_new_root(output_root)
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    if staging.exists() or staging.is_symlink():
        raise ArtifactIntegrityError("authorization staging path already exists")
    staging.mkdir()
    try:
        artifact_path = staging / _ARTIFACT_NAME
        with artifact_path.open("xb") as handle:
            handle.write(canonical_json_bytes(_artifact_payload(authorization)))
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise InvalidExperimentStateError("authorization output root already exists")
        staging.rename(target)
    finally:
        if staging.exists() or staging.is_symlink():
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            else:
                staging.unlink(missing_ok=True)
    return target


def _read_artifact(output_root: str | Path) -> FinalEvaluationAuthorization:
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactIntegrityError("authorization root must be a regular directory")
    try:
        names = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise ArtifactIntegrityError("authorization root cannot be read") from error
    if names != {_ARTIFACT_NAME}:
        raise ArtifactIntegrityError("authorization root contents are malformed")
    artifact_path = root / _ARTIFACT_NAME
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ArtifactIntegrityError("authorization artifact must be a regular file")
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            "authorization artifact JSON is malformed"
        ) from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ArtifactIntegrityError(
            "authorization artifact must contain a JSON object"
        )
    payload = cast(dict[str, object], raw)
    if (
        set(payload) != _ARTIFACT_KEYS
        or payload.get("schema_version") != _ARTIFACT_SCHEMA
    ):
        raise ArtifactIntegrityError("authorization artifact schema is malformed")
    digest = payload.get("authorization_digest")
    body = payload.get("authorization")
    if not isinstance(digest, str) or not isinstance(body, dict):
        raise ArtifactIntegrityError("authorization artifact fields are malformed")
    try:
        authorization = FinalEvaluationAuthorization.from_payload(
            cast(dict[str, object], body)
        )
    except (ContractViolationError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("authorization contract is malformed") from error
    if authorization.digest != digest:
        raise ArtifactIntegrityError("authorization digest mismatch")
    return authorization


def _validate_study_binding(
    authorization: FinalEvaluationAuthorization,
    snapshot: StudySnapshot,
) -> None:
    freeze = snapshot.freeze
    if freeze is None or freeze.outcome is not StudyOutcome.WINNER:
        raise ArtifactIntegrityError("authorization Study is not frozen WINNER")
    if authorization.study_digest != snapshot.plan.digest:
        raise ArtifactIntegrityError("authorization Study plan binding mismatch")
    if authorization.study_freeze_digest != freeze.digest:
        raise ArtifactIntegrityError("authorization Study freeze binding mismatch")
    if authorization.winner_evidence_digest != freeze.selected_evidence_digest:
        raise ArtifactIntegrityError("authorization Study winner evidence mismatch")
    if authorization.winner_strategy != freeze.selected_strategy:
        raise ArtifactIntegrityError("authorization Study winner strategy mismatch")
    if (
        authorization.development_evaluation_stop_exclusive
        != snapshot.plan.baseline_config.evaluation_stop_exclusive
    ):
        raise ArtifactIntegrityError("authorization Study development window mismatch")


def inspect_final_evaluation_authorization(
    output_root: str | Path,
    *,
    study_root: str | Path,
) -> FinalEvaluationAuthorization:
    """Read and re-bind one sealed authorization to its frozen development Study."""

    authorization = _read_artifact(output_root)
    snapshot = _winner_snapshot(study_root)
    _validate_study_binding(authorization, snapshot)
    return authorization


def authorize_final_evaluation(
    output_root: str | Path,
    *,
    study_root: str | Path,
    final_evaluation_start: str,
    final_evaluation_stop_exclusive: str,
    authorized_by: str,
    authorized_at: datetime,
) -> FinalEvaluationAuthorization:
    """Issue one authorization without reading final data or mutating the Study."""

    snapshot = _winner_snapshot(study_root)
    authorization = _authorization_from_study(
        snapshot,
        final_evaluation_start=final_evaluation_start,
        final_evaluation_stop_exclusive=final_evaluation_stop_exclusive,
        authorized_by=authorized_by,
        authorized_at=authorized_at,
    )
    published = _publish_once(output_root, authorization)
    rebuilt = inspect_final_evaluation_authorization(
        published,
        study_root=study_root,
    )
    if rebuilt != authorization:
        raise ArtifactIntegrityError("published authorization did not reconstruct")
    return rebuilt


__all__ = [
    "authorize_final_evaluation",
    "inspect_final_evaluation_authorization",
]
