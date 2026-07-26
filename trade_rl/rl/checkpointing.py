"""Atomic, content-addressed intermediate policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.training_telemetry import build_training_telemetry_callback

CHECKPOINT_MANIFEST_SCHEMA = "policy_checkpoint_v1"
CHECKPOINT_MANIFEST_NAME = "checkpoint.json"
CHECKPOINT_POLICY_NAME = "policy.zip"


class SavablePolicy(Protocol):
    def save(self, path: str) -> None: ...


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_algorithm_identity(model: SavablePolicy) -> dict[str, object] | None:
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
            require_sha256(
                self.algorithm_identity_digest,
                field="algorithm_identity_digest",
            )
            if self.algorithm_identity_digest != content_digest(
                self.algorithm_identity
            ):
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
    """Save without serializing dataset-bound rollout reconstruction objects."""

    missing = object()
    original = getattr(model, "rollout_buffer_kwargs", missing)
    if isinstance(original, dict) and "sequence_reconstructor" in original:
        sanitized = {
            key: value
            for key, value in original.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)
    try:
        model.save(target)
    finally:
        if original is not missing:
            setattr(model, "rollout_buffer_kwargs", original)


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
    """Save one model checkpoint into an atomically published step directory."""

    checkpoint_root = Path(checkpoint_root)
    destination = checkpoint_root / f"step-{observed_timestep:012d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=checkpoint_root))
    try:
        save_target = staging / "policy"
        save_policy_without_runtime_state(model, str(save_target))
        policy_path = save_target.with_suffix(".zip")
        if not policy_path.is_file():
            raise FileNotFoundError("checkpoint model save did not create policy.zip")
        policy_digest = _file_digest(policy_path)
        algorithm_identity = _model_algorithm_identity(model)
        algorithm_identity_digest = (
            None if algorithm_identity is None else content_digest(algorithm_identity)
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
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
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


def build_checkpoint_callback(
    *,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    interval_steps: int,
    max_checkpoints: int,
    environment_digest: str,
    training_config_digest: str,
) -> Any:
    """Build checkpoint and sampled Studio telemetry callbacks lazily."""

    if interval_steps < 0 or max_checkpoints <= 0:
        raise ValueError("checkpoint interval and maximum are invalid")
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    checkpoint_root = Path(checkpoint_root)
    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    if interval_steps == 0:
        return telemetry_callback

    class AtomicCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_timestep = interval_steps
            self.published = 0

        def _on_step(self) -> bool:
            observed = int(self.model.num_timesteps)
            if self.published >= max_checkpoints or observed < self.next_timestep:
                return True
            requested = self.next_timestep
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
            self.published += 1
            self.next_timestep = max(
                self.next_timestep + interval_steps,
                observed + interval_steps,
            )
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
    "publish_checkpoint",
    "save_policy_without_runtime_state",
    "validate_checkpoint_algorithm_identity",
]
