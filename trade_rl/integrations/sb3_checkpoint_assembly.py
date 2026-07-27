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
    load_checkpoint_manifest,
    validate_checkpoint_algorithm_identity,
)
from trade_rl.rl.training import ResidualTrainingConfig


@dataclass(frozen=True, slots=True)
class LoadedSB3Checkpoint:
    """Validated checkpoint model and its immutable manifest."""

    model: object
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
    if not isinstance(algorithm_config, CostCriticPPOConfig):
        return None
    provider = getattr(model, "checkpoint_identity_payload", None)
    if not callable(provider):
        raise TypeError("checkpoint_identity_payload must be callable")
    value = provider()
    if not isinstance(value, dict) or not value:
        raise ValueError("checkpoint algorithm identity must be a non-empty object")
    return value


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


def load_sb3_checkpoint_model(
    *,
    checkpoint_root: Path,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    identity: Mapping[str, object],
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    fresh_model: object,
) -> LoadedSB3Checkpoint:
    """Load and validate one algorithm-matched SB3 checkpoint."""

    manifest_path = (
        checkpoint_root / CHECKPOINT_MANIFEST_NAME
        if checkpoint_root.is_dir()
        else checkpoint_root
    )
    manifest = load_checkpoint_manifest(manifest_path)
    if manifest.algorithm != config.algorithm:
        raise ValueError("checkpoint algorithm mismatch")
    if manifest.seed != seed:
        raise ValueError("checkpoint seed mismatch")
    if manifest.environment_digest != _environment_digest(identity):
        raise ValueError("checkpoint environment identity mismatch")
    expected_training_digest = content_digest(config.digest_payload())
    if manifest.training_config_digest != expected_training_digest:
        raise ValueError("checkpoint training configuration mismatch")
    expected_algorithm_identity = _checkpoint_algorithm_identity(
        fresh_model, algorithm_config
    )
    validate_checkpoint_algorithm_identity(manifest, expected_algorithm_identity)
    loader: Any = _checkpoint_loader(algorithm_config)
    model: Any = loader.load(
        str(manifest.policy_path),
        env=environment,
        device=config.device,
    )
    if int(model.num_timesteps) != manifest.observed_timestep:
        raise ValueError("checkpoint timestep identity mismatch")
    if isinstance(algorithm_config, CostCriticPPOConfig):
        loaded_identity = _checkpoint_algorithm_identity(model, algorithm_config)
        validate_checkpoint_algorithm_identity(manifest, loaded_identity)
    if config.sequence_encoder:
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
    return LoadedSB3Checkpoint(model=model, manifest=manifest)


__all__ = ["LoadedSB3Checkpoint", "load_sb3_checkpoint_model"]
