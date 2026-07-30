"""Typed checkpoint loading for assembled Stable-Baselines3 models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_model_assembly import SB3PolicyAssembly
from trade_rl.rl.algorithm_configs import (
    AlgorithmConfig,
    CostCriticPPOConfig,
    LagrangianPPOConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
)
from trade_rl.rl.checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifest,
    checkpoint_identity_payload_for_model,
    load_checkpoint_manifest,
    validate_checkpoint_algorithm_identity,
)
from trade_rl.rl.policy_identity import (
    bind_sb3_policy_identity,
    validate_sb3_policy_architecture_compatibility,
)
from trade_rl.rl.training import ResidualTrainingConfig


@dataclass(frozen=True, slots=True)
class LoadedSB3Checkpoint:
    """Validated checkpoint model and its immutable manifest."""

    model: Any
    manifest: CheckpointManifest


def _environment_digest(identity: Mapping[str, object]) -> str:
    value = identity.get("environment_digest")
    if not isinstance(value, str) or not value:
        raise ValueError("training identity environment_digest must be non-empty")
    return value


def _checkpoint_algorithm_identity(
    model: object,
    algorithm_config: AlgorithmConfig,
) -> dict[str, object] | None:
    del algorithm_config
    return checkpoint_identity_payload_for_model(model)


def _checkpoint_loader(algorithm_config: AlgorithmConfig) -> object:
    import stable_baselines3

    if isinstance(algorithm_config, LagrangianPPOConfig):
        from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

        return LagrangianPPO
    if isinstance(algorithm_config, CostCriticPPOConfig):
        from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

        return CostCriticPPO
    if isinstance(algorithm_config, PPOConfig):
        return stable_baselines3.PPO
    if isinstance(algorithm_config, SACConfig):
        return stable_baselines3.SAC
    if isinstance(algorithm_config, TD3Config):
        return stable_baselines3.TD3
    try:
        from sb3_contrib import TQC
    except ImportError as error:
        raise RuntimeError(
            "TQC training requires the optional sb3-contrib package"
        ) from error
    return TQC


def _manifest_path(checkpoint_root: Path) -> Path:
    return (
        checkpoint_root / CHECKPOINT_MANIFEST_NAME
        if checkpoint_root.is_dir()
        else checkpoint_root
    )


def _validate_checkpoint_run_identity(
    manifest: CheckpointManifest,
    *,
    seed: int,
    config: ResidualTrainingConfig,
) -> None:
    if manifest.algorithm != config.algorithm:
        raise ValueError("checkpoint algorithm mismatch")
    if manifest.seed != seed:
        raise ValueError("checkpoint seed mismatch")
    expected_training_digest = content_digest(config.digest_payload())
    if manifest.training_config_digest != expected_training_digest:
        raise ValueError("checkpoint training configuration mismatch")


def _checkpoint_transfer_identity_parts(
    value: object,
    *,
    field: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a checkpoint identity object")
    payload = dict(value)
    required = {"algorithm", "policy", "schema_version"}
    if set(payload) != required:
        raise ValueError(f"{field} field closure mismatch")
    if payload.get("schema_version") != "sb3_checkpoint_identity_v2":
        raise ValueError(f"{field} schema mismatch")
    raw_policy = payload.get("policy")
    if not isinstance(raw_policy, Mapping) or not raw_policy:
        raise ValueError(f"{field} policy identity is missing")
    raw_algorithm = payload.get("algorithm")
    if raw_algorithm is None:
        algorithm = None
    elif isinstance(raw_algorithm, Mapping) and raw_algorithm:
        algorithm = dict(raw_algorithm)
    else:
        raise ValueError(f"{field} algorithm identity is invalid")
    return dict(raw_policy), algorithm


def _validate_checkpoint_transfer_identity(
    source: object,
    target: object,
) -> None:
    source_policy, source_algorithm = _checkpoint_transfer_identity_parts(
        source,
        field="checkpoint transfer source identity",
    )
    target_policy, target_algorithm = _checkpoint_transfer_identity_parts(
        target,
        field="checkpoint transfer target identity",
    )
    if source_algorithm != target_algorithm:
        raise ValueError("checkpoint transfer algorithm identity mismatch")
    validate_sb3_policy_architecture_compatibility(source_policy, target_policy)


def _bind_loaded_sequence_runtime(
    model: Any,
    *,
    policy: SB3PolicyAssembly,
    config: ResidualTrainingConfig,
) -> None:
    if config.observation_encoder != "hierarchical_sequence_v2":
        return
    reconstructor = policy.sequence_reconstructor
    if reconstructor is None:
        raise RuntimeError("sequence reconstructor was not resolved")
    rollout_buffer = getattr(model, "rollout_buffer", None)
    bind = getattr(rollout_buffer, "bind_sequence_reconstructor", None)
    if not callable(bind):
        raise ValueError("checkpoint rollout buffer cannot bind sequences")
    bind(
        reconstructor,
        sequence_transfer_mode=config.sequence_transfer_mode,
    )
    model.rollout_buffer_kwargs = {
        "sequence_reconstructor": reconstructor,
        "sequence_transfer_mode": config.sequence_transfer_mode,
    }


def _load_checkpoint_model(
    *,
    manifest: CheckpointManifest,
    environment: object,
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
) -> Any:
    loader: Any = _checkpoint_loader(algorithm_config)
    model: Any = loader.load(
        str(manifest.policy_path),
        env=environment,
        device=config.device,
    )
    if int(model.num_timesteps) != manifest.observed_timestep:
        raise ValueError("checkpoint timestep identity mismatch")
    return model


def load_sb3_checkpoint_model(
    *,
    checkpoint_root: Path,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    identity: Mapping[str, object],
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    fresh_model: Any,
) -> LoadedSB3Checkpoint:
    """Load one exact-environment checkpoint for ordinary resume."""

    manifest = load_checkpoint_manifest(_manifest_path(checkpoint_root))
    _validate_checkpoint_run_identity(manifest, seed=seed, config=config)
    if manifest.environment_digest != _environment_digest(identity):
        raise ValueError("checkpoint environment identity mismatch")
    expected_algorithm_identity = _checkpoint_algorithm_identity(
        fresh_model, algorithm_config
    )
    validate_checkpoint_algorithm_identity(manifest, expected_algorithm_identity)
    model = _load_checkpoint_model(
        manifest=manifest,
        environment=environment,
        config=config,
        algorithm_config=algorithm_config,
    )
    bind_sb3_policy_identity(model, policy)
    loaded_identity = _checkpoint_algorithm_identity(model, algorithm_config)
    validate_checkpoint_algorithm_identity(manifest, loaded_identity)
    _bind_loaded_sequence_runtime(model, policy=policy, config=config)
    return LoadedSB3Checkpoint(model=model, manifest=manifest)


def load_sb3_checkpoint_transfer_model(
    *,
    checkpoint_root: Path,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    identity: Mapping[str, object],
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    fresh_model: Any,
) -> LoadedSB3Checkpoint:
    """Load policy state into a different asset binding under strict compatibility."""

    manifest = load_checkpoint_manifest(_manifest_path(checkpoint_root))
    _validate_checkpoint_run_identity(manifest, seed=seed, config=config)
    target_environment_digest = _environment_digest(identity)
    if manifest.environment_digest == target_environment_digest:
        raise ValueError("checkpoint transfer requires a different environment")
    target_identity = _checkpoint_algorithm_identity(fresh_model, algorithm_config)
    _validate_checkpoint_transfer_identity(manifest.algorithm_identity, target_identity)
    model = _load_checkpoint_model(
        manifest=manifest,
        environment=environment,
        config=config,
        algorithm_config=algorithm_config,
    )
    bind_sb3_policy_identity(model, policy)
    loaded_identity = _checkpoint_algorithm_identity(model, algorithm_config)
    _validate_checkpoint_transfer_identity(manifest.algorithm_identity, loaded_identity)
    _bind_loaded_sequence_runtime(model, policy=policy, config=config)
    return LoadedSB3Checkpoint(model=model, manifest=manifest)


__all__ = [
    "LoadedSB3Checkpoint",
    "load_sb3_checkpoint_model",
    "load_sb3_checkpoint_transfer_model",
]
