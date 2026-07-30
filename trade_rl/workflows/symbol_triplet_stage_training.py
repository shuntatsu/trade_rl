"""Execute one validated symbol-triplet stage as an exploratory training run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.data import load_market_dataset_artifact
from trade_rl.domain.common import require_sha256
from trade_rl.rl.checkpointing import CheckpointManifest, checkpoint_manifests
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    SymbolTripletStageCompletion,
    SymbolTripletStageRequest,
    build_symbol_triplet_stage_request,
    commit_symbol_triplet_stage_completion,
    load_symbol_triplet_stage_completion,
    training_config_for_symbol_triplet_stage,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingCursor,
    SymbolTripletTrainingPlan,
    load_symbol_triplet_training_cursor,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    TrainingRunResult,
    execute_training_run,
    normalize_training_run_config,
)

SYMBOL_TRIPLET_STAGE_DATASET_BINDING_SCHEMA: Final = (
    "symbol_triplet_stage_dataset_binding_v1"
)


def _json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(payload)


def _resolved_path(path: str | Path) -> Path:
    return Path(path).resolve()


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    resolved = tuple(value)
    if not resolved or any(not isinstance(item, str) or not item for item in resolved):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must be unique")
    return resolved


@dataclass(frozen=True, slots=True)
class SymbolTripletStageDatasetBinding:
    """Immutable binding from one stage request to one generic-slot dataset."""

    plan_digest: str
    request_digest: str
    stage_id: str
    stage_index: int
    dataset_id: str
    dataset_path: Path
    selected_symbols: tuple[str, ...]
    slot_symbols: tuple[str, ...]
    schema_version: str = SYMBOL_TRIPLET_STAGE_DATASET_BINDING_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_STAGE_DATASET_BINDING_SCHEMA:
            raise ValueError("unsupported symbol-triplet stage dataset binding schema")
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-triplet stage dataset binding digest mismatch")
        require_sha256(self.plan_digest, field="dataset_binding.plan_digest")
        require_sha256(self.request_digest, field="dataset_binding.request_digest")
        require_sha256(self.stage_id, field="dataset_binding.stage_id")
        require_sha256(self.dataset_id, field="dataset_binding.dataset_id")
        if (
            isinstance(self.stage_index, bool)
            or not isinstance(self.stage_index, int)
            or self.stage_index < 0
        ):
            raise ValueError("dataset binding stage_index must be non-negative")
        selected_symbols = _string_tuple(
            self.selected_symbols,
            field="dataset binding selected_symbols",
        )
        slot_symbols = _string_tuple(
            self.slot_symbols,
            field="dataset binding slot_symbols",
        )
        if len(selected_symbols) != len(slot_symbols):
            raise ValueError("dataset binding symbol counts must match")
        object.__setattr__(self, "dataset_path", _resolved_path(self.dataset_path))
        object.__setattr__(self, "selected_symbols", selected_symbols)
        object.__setattr__(self, "slot_symbols", slot_symbols)
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_path": str(_resolved_path(self.dataset_path)),
            "plan_digest": self.plan_digest,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
            "selected_symbols": self.selected_symbols,
            "slot_symbols": self.slot_symbols,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    def validate_request(self, request: SymbolTripletStageRequest) -> None:
        if (
            self.plan_digest != request.plan_digest
            or self.request_digest != request.digest
            or self.stage_id != request.stage_id
            or self.stage_index != request.stage_index
        ):
            raise ValueError("dataset binding does not match the stage request")
        if self.selected_symbols != request.symbols:
            raise ValueError("dataset binding selected symbols mismatch")
        if self.slot_symbols != request.slot_symbols:
            raise ValueError("dataset binding slot symbols mismatch")

    def validate_dataset(self, dataset_path: str | Path) -> None:
        resolved = _resolved_path(dataset_path)
        if resolved != self.dataset_path:
            raise ValueError("dataset binding path mismatch")
        dataset = load_market_dataset_artifact(resolved)
        if dataset.dataset_id != self.dataset_id:
            raise ValueError("dataset identity differs from the stage binding")
        if dataset.symbols != self.slot_symbols:
            raise ValueError("dataset slot symbols differ from the stage binding")


@dataclass(frozen=True, slots=True)
class SymbolTripletStageTrainingResult:
    """Published exploratory training evidence and the advanced stage cursor."""

    request: SymbolTripletStageRequest
    completion: SymbolTripletStageCompletion
    cursor: SymbolTripletTrainingCursor
    training: TrainingRunResult
    dataset_binding: SymbolTripletStageDatasetBinding
    stage_config_path: Path


def build_symbol_triplet_stage_dataset_binding(
    request: SymbolTripletStageRequest,
    *,
    dataset_path: str | Path,
    selected_symbols: tuple[str, ...] | list[str],
) -> SymbolTripletStageDatasetBinding:
    """Bind concrete selected symbols to the stage's generic-slot dataset."""

    selected = _string_tuple(selected_symbols, field="selected symbols")
    if selected != request.symbols:
        raise ValueError("selected symbols do not match the stage request")
    resolved_dataset_path = _resolved_path(dataset_path)
    dataset = load_market_dataset_artifact(resolved_dataset_path)
    if dataset.symbols != request.slot_symbols:
        raise ValueError("dataset symbols do not match the stage slot symbols")
    return SymbolTripletStageDatasetBinding(
        plan_digest=request.plan_digest,
        request_digest=request.digest,
        stage_id=request.stage_id,
        stage_index=request.stage_index,
        dataset_id=dataset.dataset_id,
        dataset_path=resolved_dataset_path,
        selected_symbols=selected,
        slot_symbols=request.slot_symbols,
    )


def _write_immutable(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"{field} already exists with different content: {path}")
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_symbol_triplet_stage_dataset_binding(
    path: str | Path,
    binding: SymbolTripletStageDatasetBinding,
) -> Path:
    return _write_immutable(
        Path(path),
        canonical_json_bytes(binding.to_json_dict()),
        field="symbol-triplet stage dataset binding",
    )


def load_symbol_triplet_stage_dataset_binding(
    path: str | Path,
    *,
    request: SymbolTripletStageRequest,
    dataset_path: str | Path,
) -> SymbolTripletStageDatasetBinding:
    payload = _json_object(path, field="symbol-triplet stage dataset binding")
    required = {
        "dataset_id",
        "dataset_path",
        "digest",
        "plan_digest",
        "request_digest",
        "schema_version",
        "selected_symbols",
        "slot_symbols",
        "stage_id",
        "stage_index",
    }
    if set(payload) != required:
        raise ValueError("symbol-triplet stage dataset binding field closure mismatch")
    raw_dataset_path = payload["dataset_path"]
    if not isinstance(raw_dataset_path, str) or not raw_dataset_path:
        raise ValueError("dataset binding dataset_path must be a non-empty string")
    binding = SymbolTripletStageDatasetBinding(
        plan_digest=payload["plan_digest"],
        request_digest=payload["request_digest"],
        stage_id=payload["stage_id"],
        stage_index=payload["stage_index"],
        dataset_id=payload["dataset_id"],
        dataset_path=Path(raw_dataset_path),
        selected_symbols=_string_tuple(
            payload["selected_symbols"], field="dataset binding selected_symbols"
        ),
        slot_symbols=_string_tuple(
            payload["slot_symbols"], field="dataset binding slot_symbols"
        ),
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )
    binding.validate_request(request)
    binding.validate_dataset(dataset_path)
    return binding


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _training_config_mapping(config: TrainingRunConfig) -> dict[str, object]:
    environment = asdict(config.environment)
    environment.pop("reward", None)
    environment.pop("reward_config", None)
    execution = environment.pop("execution_cost")
    payload: dict[str, object] = {
        "action": asdict(config.action),
        "alpha_artifact": (
            None if config.alpha_artifact is None else str(config.alpha_artifact)
        ),
        "alpha_contract": asdict(config.alpha_contract),
        "environment": environment,
        "execution": execution,
        "exports": {
            "onnx": config.export_onnx,
            "structured_torchscript": config.export_structured_torchscript,
            "tolerance": config.export_tolerance,
            "torchscript": config.export_torchscript,
        },
        "factor_artifact": (
            None if config.factor_artifact is None else str(config.factor_artifact)
        ),
        "git_commit": config.git_commit,
        "git_dirty": config.git_dirty,
        "portfolio_risk": asdict(config.portfolio_risk),
        "resume_checkpoints": {
            str(seed): str(path) for seed, path in config.resume_checkpoints
        },
        "reward": asdict(config.reward),
        "risk": asdict(config.risk),
        "schema_version": config.schema_version,
        "training": asdict(config.training),
        "transfer_checkpoints": {
            str(seed): str(path) for seed, path in config.transfer_checkpoints
        },
        "trend": asdict(config.trend),
    }
    converted = _json_value(payload)
    if not isinstance(converted, dict):
        raise RuntimeError("training configuration serialization failed")
    return converted


def _write_stage_config(path: Path, config: TrainingRunConfig) -> Path:
    mapping = _training_config_mapping(config)
    round_tripped = TrainingRunConfig.from_mapping(mapping).resolve_artifact_paths(
        path.parent
    )
    if round_tripped != config:
        raise RuntimeError("stage training configuration did not round-trip exactly")
    return _write_immutable(
        path,
        canonical_json_bytes(mapping),
        field="symbol-triplet stage training config",
    )


def _required_integer(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"training ensemble {field} is invalid")
    return value


def _required_digest(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"training ensemble {field} is invalid")
    require_sha256(value, field=f"training_ensemble.{field}")
    return value


def _final_checkpoint_roots(
    training_path: Path,
    *,
    config: TrainingRunConfig,
) -> dict[int, Path]:
    ensemble = _json_object(training_path / "ensemble.json", field="training ensemble")
    expected_members = _required_integer(ensemble, "expected_members")
    if expected_members != len(config.training.seeds):
        raise RuntimeError("training ensemble member count differs from stage seeds")
    members = ensemble.get("members")
    if not isinstance(members, list) or len(members) != expected_members:
        raise RuntimeError("training ensemble member evidence is incomplete")
    member_seeds: list[int] = []
    for raw_member in members:
        if not isinstance(raw_member, dict):
            raise RuntimeError("training ensemble member evidence is invalid")
        raw_seed = raw_member.get("seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or raw_seed < 0:
            raise RuntimeError("training ensemble member seed is invalid")
        member_seeds.append(raw_seed)
    if tuple(member_seeds) != config.training.seeds:
        raise RuntimeError("training ensemble seed order differs from stage config")

    actual_timesteps = _required_integer(ensemble, "actual_timesteps")
    environment_digest = _required_digest(ensemble, "environment_digest")
    training_config_digest = _required_digest(ensemble, "training_config_digest")
    expected_training_digest = content_digest(config.training.digest_payload())
    if training_config_digest != expected_training_digest:
        raise RuntimeError("training ensemble configuration digest mismatch")

    roots: dict[int, Path] = {}
    for member_index, seed in enumerate(config.training.seeds):
        checkpoint_root = (
            training_path / "members" / f"member-{member_index:03d}" / "checkpoints"
        )
        manifests = checkpoint_manifests(checkpoint_root)
        eligible = tuple(
            manifest
            for manifest in manifests
            if manifest.seed == seed
            and manifest.algorithm == config.training.algorithm
            and manifest.environment_digest == environment_digest
            and manifest.training_config_digest == training_config_digest
            and manifest.observed_timestep == actual_timesteps
        )
        if not eligible:
            raise RuntimeError(f"seed {seed} has no validated final checkpoint")
        selected: CheckpointManifest = max(
            eligible,
            key=lambda manifest: (
                manifest.observed_timestep,
                manifest.requested_timestep,
                manifest.digest,
            ),
        )
        roots[seed] = selected.policy_path.parent
    return roots


def execute_symbol_triplet_stage_training(
    *,
    plan: SymbolTripletTrainingPlan,
    cursor_path: str | Path,
    previous_completion_path: str | Path | None,
    dataset_path: str | Path,
    dataset_binding_path: str | Path,
    base_config_path: str | Path,
    stage_config_path: str | Path,
    store_root: str | Path,
    run_id: str,
    completion_path: str | Path,
) -> SymbolTripletStageTrainingResult | None:
    """Run one stage and advance only after all published checkpoints validate."""

    resolved_cursor_path = Path(cursor_path)
    cursor = load_symbol_triplet_training_cursor(resolved_cursor_path, plan=plan)
    resolved_base_config_path = Path(base_config_path)
    base_config = normalize_training_run_config(
        TrainingRunConfig.from_json(resolved_base_config_path)
    )
    previous_completion = (
        None
        if previous_completion_path is None
        else load_symbol_triplet_stage_completion(
            previous_completion_path,
            plan=plan,
        )
    )
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=base_config.training.seeds,
        previous_completion=previous_completion,
    )
    if request is None:
        return None
    binding = load_symbol_triplet_stage_dataset_binding(
        dataset_binding_path,
        request=request,
        dataset_path=dataset_path,
    )
    stage_config = training_config_for_symbol_triplet_stage(base_config, request)
    resolved_stage_config_path = Path(stage_config_path)
    _write_stage_config(resolved_stage_config_path, stage_config)

    training = execute_training_run(
        config_path=resolved_stage_config_path,
        dataset_path=Path(dataset_path),
        store_root=Path(store_root),
        run_id=run_id,
    )
    if training.status != "published":
        raise RuntimeError("symbol-triplet stage training was not published")
    if training.run_kind != "research_exploratory":
        raise RuntimeError("symbol-triplet stage training must remain exploratory")
    if training.dataset_id != binding.dataset_id:
        raise RuntimeError("training result dataset identity mismatch")
    manifest = validate_training_run_directory(training.path)
    if manifest.run_kind != "research_exploratory":
        raise RuntimeError("published symbol-triplet stage is not exploratory")
    if manifest.dataset_id != binding.dataset_id:
        raise RuntimeError("published training dataset identity mismatch")

    checkpoint_roots = _final_checkpoint_roots(training.path, config=stage_config)
    completion, advanced = commit_symbol_triplet_stage_completion(
        plan,
        cursor,
        request=request,
        checkpoint_roots=checkpoint_roots,
        completion_path=completion_path,
        cursor_path=resolved_cursor_path,
    )
    return SymbolTripletStageTrainingResult(
        request=request,
        completion=completion,
        cursor=advanced,
        training=training,
        dataset_binding=binding,
        stage_config_path=resolved_stage_config_path,
    )


__all__ = [
    "SYMBOL_TRIPLET_STAGE_DATASET_BINDING_SCHEMA",
    "SymbolTripletStageDatasetBinding",
    "SymbolTripletStageTrainingResult",
    "build_symbol_triplet_stage_dataset_binding",
    "execute_symbol_triplet_stage_training",
    "load_symbol_triplet_stage_dataset_binding",
    "write_symbol_triplet_stage_dataset_binding",
]
