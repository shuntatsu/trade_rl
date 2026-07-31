"""Maintained production workflow for replay-bound execution promotion artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    execution_evidence_from_cost,
    load_execution_evidence,
    validate_execution_promotion,
)
from trade_rl.simulation.execution_replay import (
    ExecutionEventArtifact,
    build_execution_event_artifact,
    load_execution_event_artifact,
    write_execution_event_artifact_content_addressed,
)
from trade_rl.simulation.orders import OrderBookState, OrderEvent

EXECUTION_PROMOTION_ARTIFACTS_SCHEMA: Final = "execution_promotion_artifacts_v1"
_REPLAY_DIR: Final = "replays"
_EVIDENCE_DIR: Final = "evidence"
_MANIFEST_DIR: Final = "by-replay"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_idempotent(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        return path
    except FileExistsError:
        with open_regular_binary(path, field=field) as handle:
            existing = handle.read()
        if existing != payload:
            raise ValueError(f"{field} already exists with different bytes") from None
        return path


def _relative_path(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field} must be a normalized relative path")
    return path


@dataclass(frozen=True, slots=True)
class ExecutionPromotionArtifacts:
    """Verified replay and evidence paths for one exact evaluation cell."""

    root: Path
    replay_path: Path
    replay_digest: str
    evidence_path: Path
    evidence_digest: str
    manifest_path: Path
    artifact: ExecutionEventArtifact
    evidence: ExecutionEvidence

    def __post_init__(self) -> None:
        root = Path(self.root)
        replay_path = Path(self.replay_path)
        evidence_path = Path(self.evidence_path)
        manifest_path = Path(self.manifest_path)
        for field, digest in (
            ("replay_digest", self.replay_digest),
            ("evidence_digest", self.evidence_digest),
        ):
            require_sha256(digest, field=field)
        if self.artifact.digest != self.replay_digest:
            raise ValueError("execution replay digest mismatch")
        if self.evidence.digest != self.evidence_digest:
            raise ValueError("execution evidence digest mismatch")
        expected_replay = root / _REPLAY_DIR / (
            f"{self.replay_digest}.execution-replay.json"
        )
        expected_evidence = root / _EVIDENCE_DIR / (
            f"{self.evidence_digest}.execution-evidence.json"
        )
        expected_manifest = root / _MANIFEST_DIR / f"{self.replay_digest}.json"
        if replay_path != expected_replay:
            raise ValueError("execution replay path is not canonical")
        if evidence_path != expected_evidence:
            raise ValueError("execution evidence path is not canonical")
        if manifest_path != expected_manifest:
            raise ValueError("execution promotion manifest path is not canonical")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "replay_path", replay_path)
        object.__setattr__(self, "evidence_path", evidence_path)
        object.__setattr__(self, "manifest_path", manifest_path)


def _manifest_payload(artifacts: ExecutionPromotionArtifacts) -> dict[str, object]:
    identity = artifacts.artifact.replay_identity
    return {
        "candidate_config_digest": identity.candidate_config_digest,
        "dataset_id": artifacts.artifact.dataset_id,
        "evaluation_run_digest": identity.evaluation_run_digest,
        "evidence_digest": artifacts.evidence_digest,
        "evidence_path": artifacts.evidence_path.relative_to(artifacts.root).as_posix(),
        "execution_policy_digest": artifacts.artifact.execution_policy_digest,
        "fold": identity.fold,
        "replay_digest": artifacts.replay_digest,
        "replay_path": artifacts.replay_path.relative_to(artifacts.root).as_posix(),
        "schema_version": EXECUTION_PROMOTION_ARTIFACTS_SCHEMA,
        "seed": identity.seed,
    }


def _manifest_bytes(artifacts: ExecutionPromotionArtifacts) -> bytes:
    payload = _manifest_payload(artifacts)
    return canonical_json_bytes(
        {"digest": content_digest(payload), **payload}
    ) + b"\n"


def write_execution_promotion_artifacts(
    *,
    root: str | Path,
    candidate_config_digest: str,
    evaluation_run_digest: str,
    fold: int,
    seed: int,
    dataset_id: str,
    cost: ExecutionCostConfig,
    actions: Sequence[Sequence[float]],
    observation_digests: Sequence[str],
    equity_curve: Sequence[float],
    order_events: Sequence[OrderEvent],
    terminal_book: BookState,
    terminal_order_book: OrderBookState,
    sensitivity_path_modes: tuple[str, ...] = (),
) -> ExecutionPromotionArtifacts:
    """Build, publish, reload, and revalidate one promotion artifact root."""

    resolved_root = Path(root)
    artifact = build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=evaluation_run_digest,
        fold=fold,
        seed=seed,
        dataset_id=dataset_id,
        execution_policy_digest=cost.execution_policy_digest,
        actions=actions,
        observation_digests=observation_digests,
        equity_curve=equity_curve,
        order_events=order_events,
        terminal_book=terminal_book,
        terminal_order_book=terminal_order_book,
    )
    replay_path = write_execution_event_artifact_content_addressed(
        resolved_root / _REPLAY_DIR,
        artifact,
    )
    reloaded_artifact = load_execution_event_artifact(replay_path)
    if reloaded_artifact != artifact:
        raise ValueError("published execution replay differs from source artifact")
    evidence = execution_evidence_from_cost(
        dataset_id=dataset_id,
        cost=cost,
        sensitivity_path_modes=sensitivity_path_modes,
        order_event_artifact_path=replay_path,
    )
    evidence_path = (
        resolved_root
        / _EVIDENCE_DIR
        / f"{evidence.digest}.execution-evidence.json"
    )
    _write_idempotent(
        evidence_path,
        canonical_json_bytes(evidence.to_mapping()) + b"\n",
        field="execution evidence artifact",
    )
    reloaded_evidence = load_execution_evidence(evidence_path)
    validate_execution_promotion(
        reloaded_evidence,
        expected_policy_digest=cost.execution_policy_digest,
        event_artifact_path=replay_path,
        expected_candidate_config_digest=candidate_config_digest,
        expected_evaluation_run_digest=evaluation_run_digest,
        expected_fold=fold,
        expected_seed=seed,
    )
    manifest_path = resolved_root / _MANIFEST_DIR / f"{artifact.digest}.json"
    artifacts = ExecutionPromotionArtifacts(
        root=resolved_root,
        replay_path=replay_path,
        replay_digest=artifact.digest,
        evidence_path=evidence_path,
        evidence_digest=evidence.digest,
        manifest_path=manifest_path,
        artifact=reloaded_artifact,
        evidence=reloaded_evidence,
    )
    _write_idempotent(
        manifest_path,
        _manifest_bytes(artifacts),
        field="execution promotion manifest",
    )
    return load_execution_promotion_artifacts(
        root=resolved_root,
        replay_digest=artifact.digest,
    )


def load_execution_promotion_artifacts(
    *,
    root: str | Path,
    replay_digest: str,
) -> ExecutionPromotionArtifacts:
    """Resolve a promotion root only through its content-addressed manifest."""

    require_sha256(replay_digest, field="replay_digest")
    resolved_root = Path(root)
    manifest_path = resolved_root / _MANIFEST_DIR / f"{replay_digest}.json"
    try:
        with open_regular_binary(
            manifest_path,
            field="execution promotion manifest",
        ) as handle:
            raw_bytes = handle.read()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("execution promotion manifest is missing or unsafe") from error
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution promotion manifest must be valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("execution promotion manifest must be an object")
    required = {
        "candidate_config_digest",
        "dataset_id",
        "digest",
        "evaluation_run_digest",
        "evidence_digest",
        "evidence_path",
        "execution_policy_digest",
        "fold",
        "replay_digest",
        "replay_path",
        "schema_version",
        "seed",
    }
    if set(raw) != required:
        raise ValueError("execution promotion manifest field closure mismatch")
    payload = {key: value for key, value in raw.items() if key != "digest"}
    if raw["schema_version"] != EXECUTION_PROMOTION_ARTIFACTS_SCHEMA:
        raise ValueError("unsupported execution promotion manifest schema")
    if not isinstance(raw["digest"], str) or raw["digest"] != content_digest(payload):
        raise ValueError("execution promotion manifest digest mismatch")
    if raw["replay_digest"] != replay_digest:
        raise ValueError("execution promotion manifest replay digest mismatch")
    evidence_digest = raw["evidence_digest"]
    if not isinstance(evidence_digest, str):
        raise ValueError("execution promotion evidence digest must be a string")
    require_sha256(evidence_digest, field="evidence_digest")
    replay_relative = _relative_path(raw["replay_path"], field="replay_path")
    evidence_relative = _relative_path(raw["evidence_path"], field="evidence_path")
    expected_replay_relative = PurePosixPath(
        _REPLAY_DIR, f"{replay_digest}.execution-replay.json"
    )
    expected_evidence_relative = PurePosixPath(
        _EVIDENCE_DIR, f"{evidence_digest}.execution-evidence.json"
    )
    if replay_relative != expected_replay_relative:
        raise ValueError("execution promotion manifest replay path mismatch")
    if evidence_relative != expected_evidence_relative:
        raise ValueError("execution promotion manifest evidence path mismatch")
    replay_path = resolved_root.joinpath(*replay_relative.parts)
    evidence_path = resolved_root.joinpath(*evidence_relative.parts)
    artifact = load_execution_event_artifact(replay_path)
    evidence = load_execution_evidence(evidence_path)
    artifacts = ExecutionPromotionArtifacts(
        root=resolved_root,
        replay_path=replay_path,
        replay_digest=replay_digest,
        evidence_path=evidence_path,
        evidence_digest=evidence_digest,
        manifest_path=manifest_path,
        artifact=artifact,
        evidence=evidence,
    )
    if _manifest_bytes(artifacts) != raw_bytes:
        raise ValueError("execution promotion manifest is not canonical")
    identity = artifact.replay_identity
    for actual, expected, field in (
        (artifact.dataset_id, raw["dataset_id"], "dataset"),
        (
            artifact.execution_policy_digest,
            raw["execution_policy_digest"],
            "execution policy",
        ),
        (
            identity.candidate_config_digest,
            raw["candidate_config_digest"],
            "candidate",
        ),
        (
            identity.evaluation_run_digest,
            raw["evaluation_run_digest"],
            "evaluation run",
        ),
        (identity.fold, raw["fold"], "fold"),
        (identity.seed, raw["seed"], "seed"),
    ):
        if actual != expected:
            raise ValueError(f"execution promotion manifest {field} mismatch")
    validate_execution_promotion(
        evidence,
        expected_policy_digest=artifact.execution_policy_digest,
        event_artifact_path=replay_path,
        expected_candidate_config_digest=identity.candidate_config_digest,
        expected_evaluation_run_digest=identity.evaluation_run_digest,
        expected_fold=identity.fold,
        expected_seed=identity.seed,
    )
    return artifacts


__all__ = [
    "EXECUTION_PROMOTION_ARTIFACTS_SCHEMA",
    "ExecutionPromotionArtifacts",
    "load_execution_promotion_artifacts",
    "write_execution_promotion_artifacts",
]
