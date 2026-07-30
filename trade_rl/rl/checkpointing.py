"""Atomic, content-addressed intermediate policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.training_performance import TRAINING_RUNTIME_PATCHES_ATTRIBUTE
from trade_rl.rl.training_telemetry import build_training_telemetry_callback

CHECKPOINT_MANIFEST_SCHEMA = "policy_checkpoint_v1"
CHECKPOINT_MANIFEST_NAME = "checkpoint.json"
CHECKPOINT_POLICY_NAME = "policy.zip"


class SavablePolicy(Protocol):
    def save(self, path: str) -> None: ...


@contextmanager
def _open_regular_binary(path: Path, *, field: str) -> Iterator[BinaryIO]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _file_digest(path: Path, *, field: str = "checkpoint file") -> str:
    digest = hashlib.sha256()
    with _open_regular_binary(path, field=field) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_algorithm_identity(model: object) -> dict[str, object] | None:
    provider = getattr(model, "checkpoint_identity_payload", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TypeError("checkpoint_identity_payload must be callable")
    raw = provider()
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError("checkpoint algorithm identity must be a non-empty object")
    if any(not isinstance(key, str) or not key for key in raw):
        raise ValueError("checkpoint algorithm identity keys must be non-empty strings")
    payload = dict(raw)
    canonical_json_bytes(payload)
    return payload


def checkpoint_identity_payload_for_model(
    model: object,
) -> dict[str, object] | None:
    """Compose the actual policy architecture with algorithm-specific identity."""

    from trade_rl.rl.policy_identity import model_sb3_policy_identity

    policy_identity = model_sb3_policy_identity(model)
    algorithm_identity = _model_algorithm_identity(model)
    if policy_identity is None:
        return algorithm_identity
    payload: dict[str, object] = {
        "schema_version": "sb3_checkpoint_identity_v2",
        "policy": policy_identity,
        "algorithm": algorithm_identity,
    }
    canonical_json_bytes(payload)
    return payload


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
    algorithm_identity: dict[str, object] | None = None
    algorithm_identity_digest: str | None = None
    schema_version: str = CHECKPOINT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="checkpoint.digest")
        require_sha256(self.environment_digest, field="environment_digest")
        require_sha256(self.training_config_digest, field="training_config_digest")
        require_sha256(self.policy_digest, field="policy_digest")
        if not self.algorithm:
            raise ValueError("checkpoint algorithm must be non-empty")
        for name, value in (
            ("seed", self.seed),
            ("requested_timestep", self.requested_timestep),
            ("observed_timestep", self.observed_timestep),
        ):
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.observed_timestep < self.requested_timestep:
            raise ValueError("observed timestep cannot precede requested timestep")
        if self.schema_version != CHECKPOINT_MANIFEST_SCHEMA:
            raise ValueError("unsupported checkpoint manifest schema")
        if (self.algorithm_identity is None) != (
            self.algorithm_identity_digest is None
        ):
            raise ValueError("checkpoint algorithm identity is incomplete")
        if self.algorithm_identity is not None:
            if (
                not isinstance(self.algorithm_identity, dict)
                or not self.algorithm_identity
            ):
                raise ValueError(
                    "checkpoint algorithm identity must be a non-empty object"
                )
            if any(
                not isinstance(key, str) or not key for key in self.algorithm_identity
            ):
                raise ValueError(
                    "checkpoint algorithm identity keys must be non-empty strings"
                )
            canonical_json_bytes(self.algorithm_identity)
            identity_digest = self.algorithm_identity_digest
            if not isinstance(identity_digest, str):
                raise ValueError(
                    "checkpoint algorithm identity digest must be a string"
                )
            require_sha256(identity_digest, field="algorithm_identity_digest")
            if identity_digest != content_digest(self.algorithm_identity):
                raise ValueError("checkpoint algorithm identity digest mismatch")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("checkpoint manifest digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm": self.algorithm,
            "environment_digest": self.environment_digest,
            "observed_timestep": self.observed_timestep,
            "policy_digest": self.policy_digest,
            "policy_file": CHECKPOINT_POLICY_NAME,
            "requested_timestep": self.requested_timestep,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "training_config_digest": self.training_config_digest,
        }
        if self.algorithm_identity is not None:
            payload["algorithm_identity"] = self.algorithm_identity
            payload["algorithm_identity_digest"] = self.algorithm_identity_digest
        return payload


def validate_checkpoint_algorithm_identity(
    manifest: CheckpointManifest,
    expected_identity: dict[str, object] | None,
) -> None:
    """Fail closed when a checkpoint's optional algorithm identity differs."""

    if expected_identity is None:
        if manifest.algorithm_identity is not None:
            raise ValueError("checkpoint algorithm identity mismatch")
        return
    if not isinstance(expected_identity, dict) or not expected_identity:
        raise ValueError("expected algorithm identity must be a non-empty object")
    if any(not isinstance(key, str) or not key for key in expected_identity):
        raise ValueError("expected algorithm identity keys must be non-empty strings")
    canonical_json_bytes(expected_identity)
    if manifest.algorithm_identity is None:
        raise ValueError("checkpoint algorithm identity is missing")
    expected_digest = content_digest(expected_identity)
    if (
        manifest.algorithm_identity_digest != expected_digest
        or manifest.algorithm_identity != expected_identity
    ):
        raise ValueError("checkpoint algorithm identity mismatch")


def save_policy_without_runtime_state(model: SavablePolicy, target: str) -> None:
    """Save without serializing dataset-bound or temporary training runtime state."""

    missing = object()
    original_rollout_kwargs = getattr(model, "rollout_buffer_kwargs", missing)
    if (
        isinstance(original_rollout_kwargs, dict)
        and "sequence_reconstructor" in original_rollout_kwargs
    ):
        sanitized = {
            key: value
            for key, value in original_rollout_kwargs.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)

    raw_patches = getattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, missing)
    suspended: list[tuple[object, str, object]] = []
    if raw_patches is not missing:
        if not isinstance(raw_patches, tuple):
            raise TypeError("training runtime patch registry must be a tuple")
        try:
            for entry in reversed(raw_patches):
                if not isinstance(entry, tuple) or len(entry) != 4:
                    raise TypeError("training runtime patch entry is invalid")
                owner, name, had_local, local_value = entry
                if not isinstance(name, str) or not name:
                    raise TypeError("training runtime patch name is invalid")
                if not isinstance(had_local, bool):
                    raise TypeError("training runtime patch locality is invalid")
                namespace = getattr(owner, "__dict__", None)
                if not isinstance(namespace, dict) or name not in namespace:
                    raise RuntimeError("training runtime patch registry is stale")
                suspended.append((owner, name, namespace[name]))
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    delattr(owner, name)
            delattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)
        except BaseException:
            for owner, name, wrapper in reversed(suspended):
                setattr(owner, name, wrapper)
            raise

    try:
        model.save(target)
    finally:
        if raw_patches is not missing:
            for owner, name, wrapper in reversed(suspended):
                setattr(owner, name, wrapper)
            setattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, raw_patches)
        if original_rollout_kwargs is not missing:
            setattr(model, "rollout_buffer_kwargs", original_rollout_kwargs)


def _expected_algorithm_identity_digest(
    algorithm_identity: dict[str, object] | None,
) -> str | None:
    return None if algorithm_identity is None else content_digest(algorithm_identity)


def _same_checkpoint_identity(
    manifest: CheckpointManifest,
    *,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
    algorithm_identity: dict[str, object] | None,
) -> bool:
    return (
        manifest.requested_timestep == requested_timestep
        and manifest.observed_timestep == observed_timestep
        and manifest.algorithm == algorithm
        and manifest.seed == seed
        and manifest.environment_digest == environment_digest
        and manifest.training_config_digest == training_config_digest
        and manifest.algorithm_identity == algorithm_identity
        and manifest.algorithm_identity_digest
        == _expected_algorithm_identity_digest(algorithm_identity)
    )


def _same_checkpoint_run_identity(
    manifest: CheckpointManifest,
    *,
    algorithm: str,
    seed: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
    algorithm_identity: dict[str, object] | None,
) -> bool:
    return (
        manifest.observed_timestep == observed_timestep
        and manifest.algorithm == algorithm
        and manifest.seed == seed
        and manifest.environment_digest == environment_digest
        and manifest.training_config_digest == training_config_digest
        and manifest.algorithm_identity == algorithm_identity
        and manifest.algorithm_identity_digest
        == _expected_algorithm_identity_digest(algorithm_identity)
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
    algorithm_identity: dict[str, object] | None,
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
        algorithm_identity=algorithm_identity,
    ):
        return primary, existing
    if not _same_checkpoint_run_identity(
        existing,
        algorithm=algorithm,
        seed=seed,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    ):
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
        algorithm_identity=algorithm_identity,
    ):
        raise ValueError("checkpoint destination already has conflicting identity")
    return fallback, fallback_existing


def publish_checkpoint(
    *,
    model: SavablePolicy,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
) -> CheckpointManifest:
    """Publish one checkpoint atomically with full run and algorithm identity."""

    if requested_timestep <= 0 or observed_timestep < requested_timestep:
        raise ValueError("checkpoint timestep identity is invalid")
    checkpoint_root = Path(checkpoint_root)
    algorithm_identity = checkpoint_identity_payload_for_model(model)
    destination, existing = _checkpoint_destination(
        checkpoint_root,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    )
    if existing is not None:
        return existing
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_root / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        save_target = staging / "policy"
        save_policy_without_runtime_state(model, str(save_target))
        policy_path = save_target.with_suffix(".zip")
        if not policy_path.is_file():
            raise FileNotFoundError("checkpoint model save did not create policy.zip")
        policy_digest = _file_digest(policy_path)
        algorithm_identity_digest = _expected_algorithm_identity_digest(
            algorithm_identity
        )
        payload: dict[str, object] = {
            "algorithm": algorithm,
            "environment_digest": environment_digest,
            "observed_timestep": observed_timestep,
            "policy_digest": policy_digest,
            "policy_file": CHECKPOINT_POLICY_NAME,
            "requested_timestep": requested_timestep,
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "seed": seed,
            "training_config_digest": training_config_digest,
        }
        if algorithm_identity is not None:
            payload["algorithm_identity"] = algorithm_identity
            payload["algorithm_identity_digest"] = algorithm_identity_digest
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
            algorithm_identity=algorithm_identity,
            algorithm_identity_digest=algorithm_identity_digest,
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


def _optional_algorithm_identity(
    raw: dict[str, Any],
) -> tuple[dict[str, object] | None, str | None]:
    identity = raw.get("algorithm_identity")
    digest = raw.get("algorithm_identity_digest")
    if identity is None and digest is None:
        return None, None
    if not isinstance(identity, dict) or not identity:
        raise ValueError("checkpoint algorithm identity must be an object")
    if not isinstance(digest, str):
        raise ValueError("checkpoint algorithm identity digest must be a string")
    return dict(identity), digest


def load_checkpoint_manifest(path: Path) -> CheckpointManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint manifest is missing: {path}")
    with _open_regular_binary(path, field="checkpoint manifest") as handle:
        raw = json.loads(handle.read().decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint manifest must be an object")
    policy_file = raw.get("policy_path")
    if policy_file != CHECKPOINT_POLICY_NAME:
        raise ValueError("checkpoint policy file identity is invalid")
    algorithm_identity, algorithm_identity_digest = _optional_algorithm_identity(raw)
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
        algorithm_identity=algorithm_identity,
        algorithm_identity_digest=algorithm_identity_digest,
        schema_version=str(raw.get("schema_version")),
    )
    if not manifest.policy_path.exists():
        raise FileNotFoundError(f"checkpoint policy is missing: {manifest.policy_path}")
    if (
        _file_digest(manifest.policy_path, field="checkpoint policy")
        != manifest.policy_digest
    ):
        raise ValueError("checkpoint policy digest mismatch")
    return manifest


@contextmanager
def verified_checkpoint_policy_copy(
    manifest: CheckpointManifest,
) -> Iterator[Path]:
    """Yield a private immutable copy verified immediately before deserialization."""

    with tempfile.TemporaryDirectory(prefix="trade-rl-checkpoint-") as temporary:
        target = Path(temporary) / CHECKPOINT_POLICY_NAME
        with (
            _open_regular_binary(
                manifest.policy_path, field="checkpoint policy"
            ) as source,
            target.open("xb") as destination,
        ):
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if _file_digest(target) != manifest.policy_digest:
            raise ValueError("checkpoint policy changed during verified copy")
        yield target


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
    """Select deterministic requested steps across the full training horizon."""

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
    sequence_diagnostics_enabled: bool = False,
    sequence_diagnostics_interval: int = 1,
) -> Any:
    """Build full-horizon checkpoint and sampled Studio telemetry callbacks."""

    all_planned = planned_checkpoint_steps(
        total_timesteps=total_timesteps,
        interval_steps=interval_steps,
        max_checkpoints=max_checkpoints,
    )
    if (
        isinstance(starting_timestep, bool)
        or not isinstance(starting_timestep, int)
        or starting_timestep < 0
        or starting_timestep > total_timesteps
    ):
        raise ValueError("starting_timestep must be within the training horizon")
    planned = tuple(step for step in all_planned if step > starting_timestep)

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    checkpoint_root = Path(checkpoint_root)
    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    from trade_rl.rl.sequence_diagnostics import build_sequence_diagnostics_callback

    diagnostics_callback = build_sequence_diagnostics_callback(
        enabled=sequence_diagnostics_enabled,
        rollout_interval=sequence_diagnostics_interval,
    )
    passive_callbacks = [telemetry_callback]
    if diagnostics_callback is not None:
        passive_callbacks.append(diagnostics_callback)
    if not planned:
        return CallbackList(passive_callbacks)

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

    return CallbackList([AtomicCheckpointCallback(), *passive_callbacks])


__all__ = [
    "CHECKPOINT_MANIFEST_NAME",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_POLICY_NAME",
    "CheckpointManifest",
    "build_checkpoint_callback",
    "checkpoint_identity_payload_for_model",
    "checkpoint_manifests",
    "load_checkpoint_manifest",
    "planned_checkpoint_steps",
    "publish_checkpoint",
    "save_policy_without_runtime_state",
    "validate_checkpoint_algorithm_identity",
    "verified_checkpoint_policy_copy",
]
