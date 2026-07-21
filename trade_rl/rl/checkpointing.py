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
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("checkpoint manifest digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
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
        payload = {
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
]
