"""Atomic Stable-Baselines3 checkpoint artifacts and resume contracts."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import canonical_json_bytes, content_digest
from trade_rl.rl.training_telemetry import build_training_telemetry_callback

CHECKPOINT_MANIFEST_SCHEMA = "training_checkpoint_v1"
CHECKPOINT_MANIFEST_NAME = "checkpoint.json"
CHECKPOINT_POLICY_NAME = "policy.zip"


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    digest: str
    algorithm: str
    seed: int
    requested_timestep: int
    observed_timestep: int
    environment_digest: str
    training_config_digest: str
    policy_digest: str
    policy_path: Path
    schema_version: str = CHECKPOINT_MANIFEST_SCHEMA


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_policy_without_runtime_state(model: Any, target: str) -> None:
    """Save an SB3 policy while excluding process-local reconstruction helpers."""

    rollout_buffer_kwargs = getattr(model, "rollout_buffer_kwargs", None)
    if not isinstance(rollout_buffer_kwargs, dict):
        model.save(target)
        return
    runtime_reconstructor = rollout_buffer_kwargs.pop("sequence_reconstructor", None)
    try:
        model.save(target)
    finally:
        if runtime_reconstructor is not None:
            rollout_buffer_kwargs["sequence_reconstructor"] = runtime_reconstructor


def _same_checkpoint_identity(
    manifest: CheckpointManifest,
    *,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
) -> bool:
    return (
        manifest.requested_timestep == requested_timestep
        and manifest.observed_timestep == observed_timestep
        and manifest.algorithm == algorithm
        and manifest.seed == seed
        and manifest.environment_digest == environment_digest
        and manifest.training_config_digest == training_config_digest
    )


def _checkpoint_destination(
    checkpoint_root: Path,
    *,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
) -> tuple[Path, CheckpointManifest | None]:
    primary = checkpoint_root / f"step-{observed_timestep:012d}"
    if not primary.exists():
        return primary, None
    existing = load_checkpoint_manifest(primary / CHECKPOINT_MANIFEST_NAME)
    if _same_checkpoint_identity(
        existing,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
    ):
        return primary, existing
    if existing.observed_timestep != observed_timestep:
        raise ValueError("checkpoint destination already has conflicting identity")

    fallback = checkpoint_root / (
        f"step-{observed_timestep:012d}-requested-{requested_timestep:012d}"
    )
    if not fallback.exists():
        return fallback, None
    fallback_existing = load_checkpoint_manifest(fallback / CHECKPOINT_MANIFEST_NAME)
    if not _same_checkpoint_identity(
        fallback_existing,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
    ):
        raise ValueError("checkpoint destination already has conflicting identity")
    return fallback, fallback_existing


def publish_checkpoint(
    *,
    model: Any,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
) -> CheckpointManifest:
    """Publish one checkpoint atomically with requested and observed step identity."""

    if requested_timestep <= 0 or observed_timestep < requested_timestep:
        raise ValueError("checkpoint timestep identity is invalid")
    checkpoint_root = Path(checkpoint_root)
    destination, existing = _checkpoint_destination(
        checkpoint_root,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
    )
    if existing is not None:
        return existing
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_root / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        save_target = staging / CHECKPOINT_POLICY_NAME.removesuffix(".zip")
        save_policy_without_runtime_state(model, str(save_target))
        created = save_target.with_suffix(".zip")
        policy_path = staging / CHECKPOINT_POLICY_NAME
        if created != policy_path:
            created.replace(policy_path)
        if not policy_path.is_file():
            raise RuntimeError("checkpoint policy was not created")
        policy_digest = _file_digest(policy_path)
        payload = {
            "algorithm": algorithm,
            "environment_digest": environment_digest,
            "observed_timestep": observed_timestep,
            "policy_digest": policy_digest,
            "requested_timestep": requested_timestep,
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "seed": seed,
            "training_config_digest": training_config_digest,
        }
        manifest = CheckpointManifest(
            digest=content_digest(payload),
            algorithm=algorithm,
            seed=seed,
            requested_timestep=requested_timestep,
            observed_timestep=observed_timestep,
            environment_digest=environment_digest,
            training_config_digest=training_config_digest,
            policy_digest=policy_digest,
            policy_path=destination / CHECKPOINT_POLICY_NAME,
        )
        (staging / CHECKPOINT_MANIFEST_NAME).write_bytes(
            canonical_json_bytes(
                {
                    **asdict(manifest),
                    "policy_path": CHECKPOINT_POLICY_NAME,
                }
            )
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if checkpoint_root.is_dir() and not tuple(checkpoint_root.iterdir()):
            checkpoint_root.rmdir()
        raise


def _required_integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint {name} must be an integer")
    return value


def load_checkpoint_manifest(path: Path) -> CheckpointManifest:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint manifest must be an object")
    policy_file = raw.get("policy_path")
    if policy_file != CHECKPOINT_POLICY_NAME:
        raise ValueError("checkpoint policy file identity is invalid")
    manifest = CheckpointManifest(
        digest=str(raw.get("digest")),
        algorithm=str(raw.get("algorithm")),
        seed=_required_integer(raw, "seed"),
        requested_timestep=_required_integer(raw, "requested_timestep"),
        observed_timestep=_required_integer(raw, "observed_timestep"),
        environment_digest=str(raw.get("environment_digest")),
        training_config_digest=str(raw.get("training_config_digest")),
        policy_digest=str(raw.get("policy_digest")),
        policy_path=path.parent / CHECKPOINT_POLICY_NAME,
        schema_version=str(raw.get("schema_version")),
    )
    if not manifest.policy_path.is_file():
        raise FileNotFoundError(f"checkpoint policy is missing: {manifest.policy_path}")
    if _file_digest(manifest.policy_path) != manifest.policy_digest:
        raise ValueError("checkpoint policy digest mismatch")
    return manifest


def checkpoint_manifests(root: Path) -> tuple[CheckpointManifest, ...]:
    root = Path(root)
    if not root.is_dir():
        return ()
    return tuple(
        load_checkpoint_manifest(path)
        for path in sorted(root.glob(f"step-*/{CHECKPOINT_MANIFEST_NAME}"))
    )


def planned_checkpoint_steps(
    *,
    total_timesteps: int,
    interval_steps: int,
    max_checkpoints: int,
) -> tuple[int, ...]:
    """Select deterministic requested steps across the complete training horizon."""

    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, int)
        or total_timesteps <= 0
    ):
        raise ValueError("total_timesteps must be a positive integer")
    if (
        isinstance(interval_steps, bool)
        or not isinstance(interval_steps, int)
        or interval_steps < 0
    ):
        raise ValueError("interval_steps must be a non-negative integer")
    if (
        isinstance(max_checkpoints, bool)
        or not isinstance(max_checkpoints, int)
        or max_checkpoints <= 0
    ):
        raise ValueError("max_checkpoints must be a positive integer")
    if interval_steps == 0:
        return ()
    candidates = tuple(range(interval_steps, total_timesteps, interval_steps))
    if len(candidates) <= max_checkpoints:
        return candidates
    if max_checkpoints == 1:
        return (candidates[-1],)
    positions = tuple(
        round(index * (len(candidates) - 1) / (max_checkpoints - 1))
        for index in range(max_checkpoints)
    )
    return tuple(candidates[position] for position in positions)


def build_checkpoint_callback(
    *,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    interval_steps: int,
    max_checkpoints: int,
    total_timesteps: int,
    starting_timestep: int = 0,
    environment_digest: str,
    training_config_digest: str,
) -> Any:
    """Build full-horizon checkpoint and sampled Studio telemetry callbacks lazily."""

    if (
        isinstance(starting_timestep, bool)
        or not isinstance(starting_timestep, int)
        or starting_timestep < 0
        or starting_timestep > total_timesteps
    ):
        raise ValueError("starting_timestep must be within the training horizon")
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    checkpoint_root = Path(checkpoint_root)
    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    planned = tuple(
        step
        for step in planned_checkpoint_steps(
            total_timesteps=total_timesteps,
            interval_steps=interval_steps,
            max_checkpoints=max_checkpoints,
        )
        if step > starting_timestep
    )
    if not planned:
        return telemetry_callback

    class AtomicCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.cursor = 0

        def _on_step(self) -> bool:
            observed = int(self.model.num_timesteps)
            while self.cursor < len(planned) and observed >= planned[self.cursor]:
                requested = planned[self.cursor]
                publish_checkpoint(
                    model=self.model,
                    checkpoint_root=checkpoint_root,
                    algorithm=algorithm,
                    seed=seed,
                    requested_timestep=requested,
                    observed_timestep=observed,
                    environment_digest=environment_digest,
                    training_config_digest=training_config_digest,
                )
                self.cursor += 1
            return True

    return CallbackList([AtomicCheckpointCallback(), telemetry_callback])


__all__ = [
    "CHECKPOINT_MANIFEST_NAME",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_POLICY_NAME",
    "CheckpointManifest",
    "build_checkpoint_callback",
    "checkpoint_manifests",
    "load_checkpoint_manifest",
    "planned_checkpoint_steps",
    "publish_checkpoint",
    "save_policy_without_runtime_state",
]
