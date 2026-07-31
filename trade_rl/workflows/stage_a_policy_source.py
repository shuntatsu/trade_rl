"""Immutable checkpoint source bindings for Stage A policy evaluations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.rl.checkpointing import CheckpointManifest, load_checkpoint_manifest
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)

STAGE_A_POLICY_SOURCE_BINDING_SCHEMA: Final = "stage_a_policy_source_binding_v1"
STAGE_A_POLICY_SOURCE_REQUEST_INDEX_SCHEMA: Final = (
    "stage_a_policy_source_request_index_v1"
)
_BINDING_SUFFIX: Final = ".stage-a-policy-source.json"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    try:
        with open_regular_binary(path, field=field) as handle:
            return handle.read()
    except FileNotFoundError as error:
        raise ValueError(f"{field} is missing") from error
    except OSError as error:
        raise ValueError(f"{field} could not be read safely") from error


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
        existing = _read_regular_bytes(path, field=field)
        if existing != payload:
            raise ValueError(f"{field} already exists with different bytes") from None
        return path


def _relative_path(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a canonical relative path")
    return path


def _optional_relative_path(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _relative_path(value, field=field).as_posix()


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    resolved = _string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _path_relative_to_root(root: Path, path: Path, *, field: str) -> PurePosixPath:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{field} must be stored beneath the Stage A root") from error
    return _relative_path(relative.as_posix(), field=field)


def _reject_symlink_components(
    root: Path,
    relative: PurePosixPath,
    *,
    field: str,
) -> None:
    if root.is_symlink():
        raise ValueError(f"{field} root must not be a symlink")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} must not contain symlink components")


def _validate_request_against_plan(
    *,
    plan: StageAZeroShotEvaluationPlan,
    request: StageAEvaluationCellRequest,
) -> StageACandidate:
    if request.is_baseline:
        raise ValueError("Stage A baseline requests do not have policy sources")
    if request.plan_digest != plan.digest:
        raise ValueError("Stage A policy source plan digest mismatch")
    for field, actual, expected in (
        ("dataset", request.dataset_identity, plan.dataset_identity),
        ("feature", request.feature_identity, plan.feature_identity),
        ("execution", request.execution_identity, plan.execution_identity),
        ("evaluation", request.evaluation_identity, plan.evaluation_identity),
    ):
        if actual != expected:
            raise ValueError(f"Stage A policy source {field} identity mismatch")
    if request.seed not in plan.seeds:
        raise ValueError("Stage A policy source seed is not declared")
    if request.fold not in plan.folds:
        raise ValueError("Stage A policy source fold is not declared")
    if request.triplet_id not in plan.triplet_ids_for(request.split):
        raise ValueError("Stage A policy source triplet is not declared")
    candidate_id = request.candidate_id
    if candidate_id is None:
        raise ValueError("Stage A policy source candidate identity is missing")
    candidate = plan.candidate(candidate_id)
    expected_checkpoint = candidate.checkpoint_digest(request.seed)
    if request.checkpoint_digest != expected_checkpoint:
        raise ValueError("Stage A policy source checkpoint digest mismatch")
    return candidate


@dataclass(frozen=True, slots=True)
class StageAPolicySourceBinding:
    """One exact request-to-checkpoint identity binding."""

    plan_digest: str
    request_digest: str
    candidate_id: str
    seed: int
    checkpoint_digest: str
    candidate_config_digest: str
    checkpoint_policy_digest: str
    checkpoint_manifest_path: str
    serving_bundle_digest: str | None = None
    serving_bundle_path: str | None = None
    schema_version: str = STAGE_A_POLICY_SOURCE_BINDING_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_POLICY_SOURCE_BINDING_SCHEMA:
            raise ValueError("unsupported Stage A policy source binding schema")
        for field, value in (
            ("plan_digest", self.plan_digest),
            ("request_digest", self.request_digest),
            ("checkpoint_digest", self.checkpoint_digest),
            ("candidate_config_digest", self.candidate_config_digest),
            ("checkpoint_policy_digest", self.checkpoint_policy_digest),
        ):
            require_sha256(value, field=f"stage_a_policy_source.{field}")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_policy_source.candidate_id"
        )
        seed = _non_negative_int(self.seed, field="stage_a_policy_source.seed")
        checkpoint_path = _relative_path(
            self.checkpoint_manifest_path,
            field="stage_a_policy_source.checkpoint_manifest_path",
        ).as_posix()
        serving_path = _optional_relative_path(
            self.serving_bundle_path,
            field="stage_a_policy_source.serving_bundle_path",
        )
        serving_digest = _optional_sha256(
            self.serving_bundle_digest,
            field="stage_a_policy_source.serving_bundle_digest",
        )
        if (serving_path is None) != (serving_digest is None):
            raise ValueError("Stage A serving bundle source identity is incomplete")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "checkpoint_manifest_path", checkpoint_path)
        object.__setattr__(self, "serving_bundle_path", serving_path)
        object.__setattr__(self, "serving_bundle_digest", serving_digest)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A policy source binding digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_manifest_path": self.checkpoint_manifest_path,
            "checkpoint_policy_digest": self.checkpoint_policy_digest,
            "plan_digest": self.plan_digest,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "serving_bundle_digest": self.serving_bundle_digest,
            "serving_bundle_path": self.serving_bundle_path,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    @property
    def raw_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json_dict()) + b"\n"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> StageAPolicySourceBinding:
        required = {
            "candidate_config_digest",
            "candidate_id",
            "checkpoint_digest",
            "checkpoint_manifest_path",
            "checkpoint_policy_digest",
            "digest",
            "plan_digest",
            "request_digest",
            "schema_version",
            "seed",
            "serving_bundle_digest",
            "serving_bundle_path",
        }
        if set(value) != required:
            raise ValueError("Stage A policy source binding field closure mismatch")
        return cls(
            plan_digest=_string(value["plan_digest"], field="plan_digest"),
            request_digest=_string(value["request_digest"], field="request_digest"),
            candidate_id=_string(value["candidate_id"], field="candidate_id"),
            seed=_non_negative_int(value["seed"], field="seed"),
            checkpoint_digest=_string(
                value["checkpoint_digest"], field="checkpoint_digest"
            ),
            candidate_config_digest=_string(
                value["candidate_config_digest"], field="candidate_config_digest"
            ),
            checkpoint_policy_digest=_string(
                value["checkpoint_policy_digest"], field="checkpoint_policy_digest"
            ),
            checkpoint_manifest_path=_string(
                value["checkpoint_manifest_path"], field="checkpoint_manifest_path"
            ),
            serving_bundle_digest=_optional_sha256(
                value["serving_bundle_digest"], field="serving_bundle_digest"
            ),
            serving_bundle_path=_optional_relative_path(
                value["serving_bundle_path"], field="serving_bundle_path"
            ),
            schema_version=_string(value["schema_version"], field="schema_version"),
            digest=_string(value["digest"], field="digest"),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> StageAPolicySourceBinding:
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Stage A policy source binding must be valid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("Stage A policy source binding must be an object")
        binding = cls.from_mapping(cast(Mapping[str, object], raw))
        if payload != binding.raw_bytes:
            raise ValueError(
                "Stage A policy source binding must use canonical encoding"
            )
        return binding

    def _load_checkpoint(self, *, root: Path) -> CheckpointManifest:
        relative = _relative_path(
            self.checkpoint_manifest_path,
            field="stage_a_policy_source.checkpoint_manifest_path",
        )
        _reject_symlink_components(
            root,
            relative,
            field="Stage A checkpoint source",
        )
        manifest = load_checkpoint_manifest(root.joinpath(*relative.parts))
        if manifest.digest != self.checkpoint_digest:
            raise ValueError("Stage A checkpoint manifest digest mismatch")
        if manifest.seed != self.seed:
            raise ValueError("Stage A checkpoint seed mismatch")
        if manifest.training_config_digest != self.candidate_config_digest:
            raise ValueError("Stage A checkpoint training config digest mismatch")
        if manifest.policy_digest != self.checkpoint_policy_digest:
            raise ValueError("Stage A checkpoint policy digest mismatch")
        return manifest

    def validate(
        self,
        *,
        root: str | Path,
        plan: StageAZeroShotEvaluationPlan,
        request: StageAEvaluationCellRequest,
    ) -> CheckpointManifest:
        resolved_root = Path(root)
        candidate = _validate_request_against_plan(plan=plan, request=request)
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A policy source binding plan mismatch")
        if self.request_digest != request.digest:
            raise ValueError("Stage A policy source binding request mismatch")
        if self.candidate_id != candidate.candidate_id:
            raise ValueError("Stage A policy source binding candidate mismatch")
        if self.seed != request.seed:
            raise ValueError("Stage A policy source binding seed mismatch")
        if self.checkpoint_digest != request.checkpoint_digest:
            raise ValueError("Stage A policy source binding checkpoint mismatch")
        if self.candidate_config_digest != candidate.candidate_config_digest:
            raise ValueError("Stage A policy source binding config mismatch")
        if self.serving_bundle_path is not None:
            raise ValueError("Stage A serving bundle sources are not supported yet")
        return self._load_checkpoint(root=resolved_root)


class StageAPolicySourceStore:
    """Publish and reload immutable request-indexed policy sources."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _paths(self, binding: StageAPolicySourceBinding) -> tuple[Path, Path]:
        binding_path = (
            self.root
            / "bindings"
            / binding.request_digest
            / f"{binding.digest}{_BINDING_SUFFIX}"
        )
        index_path = self.root / "by-request" / f"{binding.request_digest}.json"
        return binding_path, index_path

    @staticmethod
    def _index_bytes(
        *,
        binding: StageAPolicySourceBinding,
    ) -> bytes:
        relative = PurePosixPath(
            "bindings",
            binding.request_digest,
            f"{binding.digest}{_BINDING_SUFFIX}",
        )
        payload: dict[str, object] = {
            "binding_digest": binding.digest,
            "binding_path": relative.as_posix(),
            "request_digest": binding.request_digest,
            "schema_version": STAGE_A_POLICY_SOURCE_REQUEST_INDEX_SCHEMA,
        }
        return (
            canonical_json_bytes({"digest": content_digest(payload), **payload}) + b"\n"
        )

    def publish(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        request: StageAEvaluationCellRequest,
        checkpoint_manifest_path: str | Path,
    ) -> StageAPolicySourceBinding:
        candidate = _validate_request_against_plan(plan=plan, request=request)
        relative = _path_relative_to_root(
            self.root,
            Path(checkpoint_manifest_path),
            field="Stage A checkpoint manifest path",
        )
        _reject_symlink_components(
            self.root,
            relative,
            field="Stage A checkpoint source",
        )
        manifest = load_checkpoint_manifest(self.root.joinpath(*relative.parts))
        if manifest.digest != request.checkpoint_digest:
            raise ValueError("Stage A checkpoint manifest digest mismatch")
        if manifest.seed != request.seed:
            raise ValueError("Stage A checkpoint seed mismatch")
        if manifest.training_config_digest != candidate.candidate_config_digest:
            raise ValueError("Stage A checkpoint training config digest mismatch")

        binding = StageAPolicySourceBinding(
            plan_digest=plan.digest,
            request_digest=request.digest,
            candidate_id=candidate.candidate_id,
            seed=request.seed,
            checkpoint_digest=manifest.digest,
            candidate_config_digest=manifest.training_config_digest,
            checkpoint_policy_digest=manifest.policy_digest,
            checkpoint_manifest_path=relative.as_posix(),
        )
        binding.validate(root=self.root, plan=plan, request=request)
        binding_path, index_path = self._paths(binding)
        binding_relative = _path_relative_to_root(
            self.root,
            binding_path,
            field="Stage A policy source binding path",
        )
        index_relative = _path_relative_to_root(
            self.root,
            index_path,
            field="Stage A policy source request index path",
        )
        _reject_symlink_components(
            self.root,
            binding_relative,
            field="Stage A policy source binding",
        )
        _reject_symlink_components(
            self.root,
            index_relative,
            field="Stage A policy source request index",
        )
        index_bytes = self._index_bytes(binding=binding)

        if index_path.exists() or index_path.is_symlink():
            existing_index = _read_regular_bytes(
                index_path,
                field="Stage A policy source request index",
            )
            if existing_index != index_bytes:
                raise ValueError("Stage A policy source request is already bound")
            loaded = self.load(request.digest)
            if loaded != binding:
                raise ValueError("Stage A policy source request is already bound")
            return loaded

        _write_idempotent(
            binding_path,
            binding.raw_bytes,
            field="Stage A policy source binding",
        )
        try:
            _write_idempotent(
                index_path,
                index_bytes,
                field="Stage A policy source request index",
            )
        except ValueError as error:
            raise ValueError(
                "Stage A policy source request is already bound"
            ) from error
        return self.load(request.digest)

    def load(self, request_digest: str) -> StageAPolicySourceBinding:
        require_sha256(request_digest, field="stage_a_policy_source_request_digest")
        index_relative = PurePosixPath(
            "by-request",
            f"{request_digest}.json",
        )
        _reject_symlink_components(
            self.root,
            index_relative,
            field="Stage A policy source request index",
        )
        index_path = self.root.joinpath(*index_relative.parts)
        index_bytes = _read_regular_bytes(
            index_path,
            field="Stage A policy source request index",
        )
        try:
            raw = json.loads(index_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Stage A policy source request index must be valid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("Stage A policy source request index must be an object")
        required = {
            "binding_digest",
            "binding_path",
            "digest",
            "request_digest",
            "schema_version",
        }
        if set(raw) != required:
            raise ValueError(
                "Stage A policy source request index field closure mismatch"
            )
        if raw["schema_version"] != STAGE_A_POLICY_SOURCE_REQUEST_INDEX_SCHEMA:
            raise ValueError("unsupported Stage A policy source request index schema")
        indexed_request = _string(raw["request_digest"], field="request_digest")
        if indexed_request != request_digest:
            raise ValueError("Stage A policy source request index identity mismatch")
        binding_digest = _string(raw["binding_digest"], field="binding_digest")
        require_sha256(binding_digest, field="stage_a_policy_source.binding_digest")
        relative = _relative_path(raw["binding_path"], field="binding_path")
        expected_relative = PurePosixPath(
            "bindings",
            request_digest,
            f"{binding_digest}{_BINDING_SUFFIX}",
        )
        if relative != expected_relative:
            raise ValueError("Stage A policy source binding path mismatch")
        payload = {key: value for key, value in raw.items() if key != "digest"}
        digest = _string(raw["digest"], field="digest")
        require_sha256(digest, field="stage_a_policy_source_request_index.digest")
        if digest != content_digest(payload):
            raise ValueError("Stage A policy source request index digest mismatch")
        if index_bytes != canonical_json_bytes(raw) + b"\n":
            raise ValueError(
                "Stage A policy source request index must use canonical encoding"
            )

        _reject_symlink_components(
            self.root,
            relative,
            field="Stage A policy source binding",
        )
        binding_bytes = _read_regular_bytes(
            self.root.joinpath(*relative.parts),
            field="Stage A policy source binding",
        )
        binding = StageAPolicySourceBinding.from_json_bytes(binding_bytes)
        if binding.digest != binding_digest:
            raise ValueError("Stage A policy source binding digest mismatch")
        if binding.request_digest != request_digest:
            raise ValueError("Stage A policy source binding request identity mismatch")
        binding._load_checkpoint(root=self.root)
        return binding


__all__ = [
    "STAGE_A_POLICY_SOURCE_BINDING_SCHEMA",
    "STAGE_A_POLICY_SOURCE_REQUEST_INDEX_SCHEMA",
    "StageAPolicySourceBinding",
    "StageAPolicySourceStore",
]
