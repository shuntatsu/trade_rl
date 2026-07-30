"""Fail-closed stage orchestration for shared-policy symbol-triplet training."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifest,
    load_checkpoint_manifest,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingCursor,
    SymbolTripletTrainingPlan,
    SymbolTripletTrainingStage,
    advance_symbol_triplet_training_cursor,
    current_symbol_triplet_training_stage,
)
from trade_rl.workflows.training_run import TrainingRunConfig

SYMBOL_TRIPLET_STAGE_CHECKPOINT_SCHEMA: Final = "symbol_triplet_stage_checkpoint_v1"
SYMBOL_TRIPLET_STAGE_REQUEST_SCHEMA: Final = "symbol_triplet_stage_request_v1"
SYMBOL_TRIPLET_STAGE_COMPLETION_SCHEMA: Final = "symbol_triplet_stage_completion_v1"


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _seed_tuple(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("training seeds must be non-empty")
    for seed in seeds:
        _non_negative_integer(seed, field="training seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    return seeds


def _json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(payload)


def _checkpoint_manifest(checkpoint_root: Path) -> CheckpointManifest:
    return load_checkpoint_manifest(checkpoint_root / CHECKPOINT_MANIFEST_NAME)


@dataclass(frozen=True, slots=True)
class SymbolTripletStageCheckpoint:
    """One immutable seed-scoped checkpoint reference between adjacent stages."""

    seed: int
    checkpoint_root: Path
    checkpoint_digest: str
    schema_version: str = SYMBOL_TRIPLET_STAGE_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_STAGE_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported symbol-triplet stage checkpoint schema")
        seed = _non_negative_integer(self.seed, field="checkpoint seed")
        checkpoint_root = Path(self.checkpoint_root)
        if not str(checkpoint_root):
            raise ValueError("checkpoint root must be non-empty")
        require_sha256(
            self.checkpoint_digest,
            field="stage_checkpoint.checkpoint_digest",
        )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "checkpoint_root", checkpoint_root)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_root": str(self.checkpoint_root),
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    def validate_manifest(self) -> CheckpointManifest:
        manifest = _checkpoint_manifest(self.checkpoint_root)
        if manifest.seed != self.seed:
            raise ValueError("stage checkpoint seed mismatch")
        if manifest.digest != self.checkpoint_digest:
            raise ValueError("stage checkpoint digest mismatch")
        return manifest


@dataclass(frozen=True, slots=True)
class SymbolTripletStageRequest:
    """Exact current-stage binding and validated transfer inputs."""

    plan_digest: str
    stage_id: str
    stage_index: int
    cycle_index: int
    train_split_slot: int
    source_slot_id: str
    source_triplet_id: str
    symbols: tuple[str, ...]
    slot_symbols: tuple[str, ...]
    training_seeds: tuple[int, ...]
    transfer_checkpoints: tuple[SymbolTripletStageCheckpoint, ...]
    schema_version: str = SYMBOL_TRIPLET_STAGE_REQUEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_STAGE_REQUEST_SCHEMA:
            raise ValueError("unsupported symbol-triplet stage request schema")
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-triplet stage request digest mismatch")
        require_sha256(self.plan_digest, field="stage_request.plan_digest")
        require_sha256(self.stage_id, field="stage_request.stage_id")
        require_sha256(self.source_slot_id, field="stage_request.source_slot_id")
        require_sha256(
            self.source_triplet_id,
            field="stage_request.source_triplet_id",
        )
        stage_index = _non_negative_integer(self.stage_index, field="stage_index")
        cycle_index = _non_negative_integer(self.cycle_index, field="cycle_index")
        train_split_slot = _non_negative_integer(
            self.train_split_slot,
            field="train_split_slot",
        )
        training_seeds = _seed_tuple(self.training_seeds)
        checkpoint_seeds = tuple(
            checkpoint.seed for checkpoint in self.transfer_checkpoints
        )
        if checkpoint_seeds and checkpoint_seeds != training_seeds:
            raise ValueError("transfer checkpoint seeds must match training seeds")
        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "train_split_slot", train_split_slot)
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "digest", expected_digest)

    @property
    def slot_bindings(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.slot_symbols, self.symbols, strict=True))

    def digest_payload(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "slot_symbols": self.slot_symbols,
            "source_slot_id": self.source_slot_id,
            "source_triplet_id": self.source_triplet_id,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "symbols": self.symbols,
            "train_split_slot": self.train_split_slot,
            "training_seeds": self.training_seeds,
            "transfer_checkpoints": tuple(
                checkpoint.to_json_dict() for checkpoint in self.transfer_checkpoints
            ),
        }


@dataclass(frozen=True, slots=True)
class SymbolTripletStageCompletion:
    """Validated checkpoint evidence for one completed training stage."""

    plan_digest: str
    stage_id: str
    stage_index: int
    training_seeds: tuple[int, ...]
    checkpoints: tuple[SymbolTripletStageCheckpoint, ...]
    schema_version: str = SYMBOL_TRIPLET_STAGE_COMPLETION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_STAGE_COMPLETION_SCHEMA:
            raise ValueError("unsupported symbol-triplet stage completion schema")
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-triplet stage completion digest mismatch")
        require_sha256(self.plan_digest, field="stage_completion.plan_digest")
        require_sha256(self.stage_id, field="stage_completion.stage_id")
        stage_index = _non_negative_integer(self.stage_index, field="stage_index")
        training_seeds = _seed_tuple(self.training_seeds)
        checkpoint_seeds = tuple(checkpoint.seed for checkpoint in self.checkpoints)
        if checkpoint_seeds != training_seeds:
            raise ValueError("completion checkpoint seeds must match training seeds")
        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "checkpoints": tuple(
                checkpoint.to_json_dict() for checkpoint in self.checkpoints
            ),
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "training_seeds": self.training_seeds,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    def validate_plan(self, plan: SymbolTripletTrainingPlan) -> None:
        if self.plan_digest != plan.digest:
            raise ValueError("stage completion belongs to a different training plan")
        if self.stage_index >= plan.stage_count:
            raise ValueError("stage completion index is outside the training plan")
        if self.stage_id != plan.stages[self.stage_index].stage_id:
            raise ValueError("stage completion stage identity mismatch")
        for checkpoint in self.checkpoints:
            checkpoint.validate_manifest()


def _request_from_stage(
    plan: SymbolTripletTrainingPlan,
    stage: SymbolTripletTrainingStage,
    *,
    training_seeds: tuple[int, ...],
    transfer_checkpoints: tuple[SymbolTripletStageCheckpoint, ...],
) -> SymbolTripletStageRequest:
    return SymbolTripletStageRequest(
        plan_digest=plan.digest,
        stage_id=stage.stage_id,
        stage_index=stage.stage_index,
        cycle_index=stage.cycle_index,
        train_split_slot=stage.train_split_slot,
        source_slot_id=stage.source_slot_id,
        source_triplet_id=stage.source_triplet_id,
        symbols=stage.symbols,
        slot_symbols=stage.slot_symbols,
        training_seeds=training_seeds,
        transfer_checkpoints=transfer_checkpoints,
    )


def _validate_previous_completion(
    plan: SymbolTripletTrainingPlan,
    stage: SymbolTripletTrainingStage,
    completion: SymbolTripletStageCompletion,
    *,
    training_seeds: tuple[int, ...],
) -> None:
    expected_previous_index = stage.stage_index - 1
    if (
        completion.plan_digest != plan.digest
        or completion.stage_index != expected_previous_index
        or completion.stage_id != plan.stages[expected_previous_index].stage_id
    ):
        raise ValueError("previous stage completion does not match the training cursor")
    if completion.training_seeds != training_seeds:
        raise ValueError("previous stage completion training seeds mismatch")
    completion.validate_plan(plan)


def build_symbol_triplet_stage_request(
    plan: SymbolTripletTrainingPlan,
    cursor: SymbolTripletTrainingCursor,
    *,
    training_seeds: tuple[int, ...] | list[int],
    previous_completion: SymbolTripletStageCompletion | None,
) -> SymbolTripletStageRequest | None:
    """Resolve the current stage and only the immediately preceding checkpoints."""

    cursor.validate_plan(plan)
    seeds = _seed_tuple(training_seeds)
    stage = current_symbol_triplet_training_stage(plan, cursor)
    if stage is None:
        if previous_completion is not None:
            previous_completion.validate_plan(plan)
            if (
                previous_completion.stage_index != plan.stage_count - 1
                or previous_completion.stage_id != plan.stages[-1].stage_id
                or previous_completion.training_seeds != seeds
            ):
                raise ValueError(
                    "previous stage completion does not match completed plan"
                )
        return None
    if stage.stage_index == 0:
        if previous_completion is not None:
            raise ValueError("previous stage completion is invalid for initial stage")
        return _request_from_stage(
            plan,
            stage,
            training_seeds=seeds,
            transfer_checkpoints=(),
        )
    if previous_completion is None:
        raise ValueError("previous stage completion is required")
    _validate_previous_completion(
        plan,
        stage,
        previous_completion,
        training_seeds=seeds,
    )
    return _request_from_stage(
        plan,
        stage,
        training_seeds=seeds,
        transfer_checkpoints=previous_completion.checkpoints,
    )


def training_config_for_symbol_triplet_stage(
    config: TrainingRunConfig,
    request: SymbolTripletStageRequest,
) -> TrainingRunConfig:
    """Bind a clean run configuration to one validated stage transfer mapping."""

    if tuple(config.training.seeds) != request.training_seeds:
        raise ValueError("training configuration seeds differ from stage request")
    if config.resume_checkpoints or config.transfer_checkpoints:
        raise ValueError("stage orchestration requires a transport-free base config")
    for checkpoint in request.transfer_checkpoints:
        checkpoint.validate_manifest()
    return replace(
        config,
        transfer_checkpoints=tuple(
            (checkpoint.seed, checkpoint.checkpoint_root)
            for checkpoint in request.transfer_checkpoints
        ),
    )


def _validate_request_for_cursor(
    plan: SymbolTripletTrainingPlan,
    cursor: SymbolTripletTrainingCursor,
    request: SymbolTripletStageRequest,
) -> SymbolTripletTrainingStage:
    cursor.validate_plan(plan)
    stage = current_symbol_triplet_training_stage(plan, cursor)
    if stage is None:
        raise ValueError("symbol-triplet training plan is already complete")
    if request.plan_digest != plan.digest:
        raise ValueError("stage request belongs to a different training plan")
    expected = (
        stage.stage_id,
        stage.stage_index,
        stage.cycle_index,
        stage.train_split_slot,
        stage.source_slot_id,
        stage.source_triplet_id,
        stage.symbols,
        stage.slot_symbols,
    )
    actual = (
        request.stage_id,
        request.stage_index,
        request.cycle_index,
        request.train_split_slot,
        request.source_slot_id,
        request.source_triplet_id,
        request.symbols,
        request.slot_symbols,
    )
    if actual != expected:
        raise ValueError("stage request does not match the training cursor")
    for checkpoint in request.transfer_checkpoints:
        checkpoint.validate_manifest()
    return stage


def _checkpoint_references(
    checkpoint_roots: dict[int, Path],
    *,
    training_seeds: tuple[int, ...],
) -> tuple[SymbolTripletStageCheckpoint, ...]:
    if set(checkpoint_roots) != set(training_seeds):
        raise ValueError("checkpoint seeds must exactly match training seeds")
    references: list[SymbolTripletStageCheckpoint] = []
    manifests: list[CheckpointManifest] = []
    for seed in training_seeds:
        checkpoint_root = Path(checkpoint_roots[seed])
        manifest = _checkpoint_manifest(checkpoint_root)
        if manifest.seed != seed:
            raise ValueError("checkpoint seed does not match its mapping key")
        references.append(
            SymbolTripletStageCheckpoint(
                seed=seed,
                checkpoint_root=checkpoint_root,
                checkpoint_digest=manifest.digest,
            )
        )
        manifests.append(manifest)
    algorithms = {manifest.algorithm for manifest in manifests}
    environment_digests = {manifest.environment_digest for manifest in manifests}
    training_config_digests = {
        manifest.training_config_digest for manifest in manifests
    }
    timestep_identities = {
        (manifest.requested_timestep, manifest.observed_timestep)
        for manifest in manifests
    }
    if len(algorithms) != 1:
        raise ValueError("stage checkpoints use different algorithms")
    if len(environment_digests) != 1:
        raise ValueError("stage checkpoints use different environments")
    if len(training_config_digests) != 1:
        raise ValueError("stage checkpoints use different training configurations")
    if len(timestep_identities) != 1:
        raise ValueError("stage checkpoints use different timestep identities")
    return tuple(references)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_completion_and_cursor(
    completion_path: Path,
    completion: SymbolTripletStageCompletion,
    cursor_path: Path,
    cursor: SymbolTripletTrainingCursor,
) -> None:
    if completion_path == cursor_path:
        raise ValueError("completion and cursor paths must differ")
    if completion_path.exists():
        raise FileExistsError(f"stage completion already exists: {completion_path}")
    completion_payload = canonical_json_bytes(completion.to_json_dict())
    cursor_payload = canonical_json_bytes(cursor.to_json_dict())
    _atomic_write(completion_path, completion_payload)
    try:
        _atomic_write(cursor_path, cursor_payload)
    except BaseException:
        completion_path.unlink(missing_ok=True)
        raise


def commit_symbol_triplet_stage_completion(
    plan: SymbolTripletTrainingPlan,
    cursor: SymbolTripletTrainingCursor,
    *,
    request: SymbolTripletStageRequest,
    checkpoint_roots: dict[int, Path],
    completion_path: str | Path,
    cursor_path: str | Path,
) -> tuple[SymbolTripletStageCompletion, SymbolTripletTrainingCursor]:
    """Validate all seed checkpoints, publish completion, then advance the cursor."""

    stage = _validate_request_for_cursor(plan, cursor, request)
    resolved_cursor_path = Path(cursor_path)
    if resolved_cursor_path.exists():
        from trade_rl.workflows.symbol_triplet_training_cursor import (
            load_symbol_triplet_training_cursor,
        )

        persisted_cursor = load_symbol_triplet_training_cursor(
            resolved_cursor_path,
            plan=plan,
        )
        if persisted_cursor != cursor:
            raise ValueError("persisted training cursor differs from requested cursor")
    checkpoints = _checkpoint_references(
        checkpoint_roots,
        training_seeds=request.training_seeds,
    )
    completion = SymbolTripletStageCompletion(
        plan_digest=plan.digest,
        stage_id=stage.stage_id,
        stage_index=stage.stage_index,
        training_seeds=request.training_seeds,
        checkpoints=checkpoints,
    )
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        cursor,
        completed_stage_id=stage.stage_id,
    )
    _write_completion_and_cursor(
        Path(completion_path),
        completion,
        resolved_cursor_path,
        advanced,
    )
    return completion, advanced


def load_symbol_triplet_stage_completion(
    path: str | Path,
    *,
    plan: SymbolTripletTrainingPlan,
) -> SymbolTripletStageCompletion:
    """Load completion evidence and revalidate every referenced checkpoint."""

    payload = _json_object(path, field="symbol-triplet stage completion")
    required = {
        "checkpoints",
        "digest",
        "plan_digest",
        "schema_version",
        "stage_id",
        "stage_index",
        "training_seeds",
    }
    if set(payload) != required:
        raise ValueError("symbol-triplet stage completion field closure mismatch")
    raw_checkpoints = payload["checkpoints"]
    raw_training_seeds = payload["training_seeds"]
    if not isinstance(raw_checkpoints, list):
        raise ValueError("stage completion checkpoints must be a list")
    if not isinstance(raw_training_seeds, list):
        raise ValueError("stage completion training seeds must be a list")
    checkpoint_fields = {
        "checkpoint_digest",
        "checkpoint_root",
        "schema_version",
        "seed",
    }
    checkpoints: list[SymbolTripletStageCheckpoint] = []
    for raw_checkpoint in raw_checkpoints:
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("stage completion checkpoint must be an object")
        if set(raw_checkpoint) != checkpoint_fields:
            raise ValueError("stage completion checkpoint field closure mismatch")
        checkpoints.append(
            SymbolTripletStageCheckpoint(
                seed=raw_checkpoint["seed"],
                checkpoint_root=Path(raw_checkpoint["checkpoint_root"]),
                checkpoint_digest=raw_checkpoint["checkpoint_digest"],
                schema_version=raw_checkpoint["schema_version"],
            )
        )
    completion = SymbolTripletStageCompletion(
        plan_digest=payload["plan_digest"],
        stage_id=payload["stage_id"],
        stage_index=payload["stage_index"],
        training_seeds=tuple(raw_training_seeds),
        checkpoints=tuple(checkpoints),
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )
    completion.validate_plan(plan)
    return completion


__all__ = [
    "SYMBOL_TRIPLET_STAGE_CHECKPOINT_SCHEMA",
    "SYMBOL_TRIPLET_STAGE_COMPLETION_SCHEMA",
    "SYMBOL_TRIPLET_STAGE_REQUEST_SCHEMA",
    "SymbolTripletStageCheckpoint",
    "SymbolTripletStageCompletion",
    "SymbolTripletStageRequest",
    "build_symbol_triplet_stage_request",
    "commit_symbol_triplet_stage_completion",
    "load_symbol_triplet_stage_completion",
    "training_config_for_symbol_triplet_stage",
]
