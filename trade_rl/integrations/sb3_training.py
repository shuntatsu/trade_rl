"""Stable-Baselines3 training adapter isolated from the RL core contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.algorithm_configs import (
    PPOConfig,
    SACConfig,
    TD3Config,
    build_algorithm_config,
)
from trade_rl.rl.replay import (
    load_replay_buffer_artifact,
    write_replay_buffer_artifact,
)
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    _environment_identity,
    _validate_training_environment,
)


def _build_training_environment(
    factory: Callable[[], Any], n_envs: int
) -> Any:
    if n_envs == 1:
        return factory()

    from stable_baselines3.common.vec_env import SubprocVecEnv

    return SubprocVecEnv([factory for _ in range(n_envs)])


class StableBaselines3Backend:
    """Train one policy with an optional SB3-family algorithm."""

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
    ) -> None:
        self.environment_factory = environment_factory
        self.verbose = verbose
        self.resume_replay_artifact = resume_replay_artifact

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        from stable_baselines3 import PPO, SAC, TD3

        probe = self.environment_factory()
        environment: Any | None = None
        try:
            identity = _environment_identity(probe)
            _validate_training_environment(identity, config)
            algorithm_config = build_algorithm_config(config)
            policy_kwargs: dict[str, Any] = {
                "net_arch": list(algorithm_config.policy_net_arch)
            }
            if isinstance(algorithm_config, PPOConfig):
                policy_kwargs["log_std_init"] = algorithm_config.log_std_init
            if config.asset_set_encoder:
                from trade_rl.rl.policies import AssetSetFeatureExtractor

                unwrapped: Any = getattr(probe, "unwrapped", probe)
                layout = getattr(unwrapped, "layout", None)
                active_column = getattr(unwrapped, "asset_active_column", None)
                if layout is None or not isinstance(active_column, int):
                    raise ValueError(
                        "asset-set training requires environment layout metadata"
                    )
                policy_kwargs.update(
                    {
                        "features_extractor_class": AssetSetFeatureExtractor,
                        "features_extractor_kwargs": {
                            "n_symbols": layout.n_symbols,
                            "per_symbol_width": layout.per_symbol_width,
                            "global_width": layout.global_width,
                            "active_column": active_column,
                            "asset_embedding_dim": config.asset_embedding_dim,
                            "global_embedding_dim": config.global_embedding_dim,
                        },
                    }
                )
            if config.n_envs == 1:
                environment = probe
                probe = None
            else:
                probe_to_close = probe
                probe = None
                probe_to_close.close()
                environment = _build_training_environment(
                    self.environment_factory, config.n_envs
                )
            common: dict[str, Any] = {
                "learning_rate": algorithm_config.learning_rate,
                "gamma": algorithm_config.gamma,
                "policy_kwargs": policy_kwargs,
                "seed": seed,
                "device": config.device,
                "verbose": self.verbose,
            }
            model: Any
            if isinstance(algorithm_config, PPOConfig):
                model = PPO(
                    config.policy,
                    environment,
                    n_steps=algorithm_config.n_steps,
                    batch_size=algorithm_config.batch_size,
                    n_epochs=algorithm_config.n_epochs,
                    gae_lambda=algorithm_config.gae_lambda,
                    clip_range=algorithm_config.clip_range,
                    normalize_advantage=algorithm_config.normalize_advantage,
                    ent_coef=algorithm_config.ent_coef,
                    vf_coef=algorithm_config.vf_coef,
                    max_grad_norm=algorithm_config.max_grad_norm,
                    target_kl=algorithm_config.target_kl,
                    use_sde=algorithm_config.use_sde,
                    sde_sample_freq=algorithm_config.sde_sample_freq,
                    **common,
                )
            else:
                off_policy: dict[str, Any] = {
                    "buffer_size": algorithm_config.buffer_size,
                    "learning_starts": algorithm_config.learning_starts,
                    "batch_size": algorithm_config.batch_size,
                    "train_freq": algorithm_config.train_freq,
                    "gradient_steps": algorithm_config.gradient_steps,
                    **common,
                }
                if isinstance(algorithm_config, SACConfig):
                    model = SAC(
                        config.policy,
                        environment,
                        use_sde=algorithm_config.use_sde,
                        sde_sample_freq=algorithm_config.sde_sample_freq,
                        **off_policy,
                    )
                elif isinstance(algorithm_config, TD3Config):
                    model = TD3(config.policy, environment, **off_policy)
                else:
                    try:
                        from sb3_contrib import TQC
                    except ImportError as error:
                        raise RuntimeError(
                            "TQC training requires the optional sb3-contrib package"
                        ) from error
                    model = TQC(
                        config.policy,
                        environment,
                        use_sde=algorithm_config.use_sde,
                        sde_sample_freq=algorithm_config.sde_sample_freq,
                        **off_policy,
                    )
            from trade_rl.rl.checkpointing import build_checkpoint_callback

            if self.resume_replay_artifact is not None:
                if config.algorithm == "ppo":
                    raise ValueError("PPO cannot resume from a replay buffer")
                resume_manifest, resume_path = load_replay_buffer_artifact(
                    self.resume_replay_artifact
                )
                if resume_manifest.algorithm != config.algorithm:
                    raise ValueError("replay buffer algorithm mismatch")
                if resume_manifest.environment_digest != identity["environment_digest"]:
                    raise ValueError("replay buffer environment identity mismatch")
                model.load_replay_buffer(str(resume_path))

            callback = build_checkpoint_callback(
                checkpoint_root=output_path.parent / "checkpoints",
                algorithm=config.algorithm,
                seed=seed,
                interval_steps=config.resolved_checkpoint_interval,
                max_checkpoints=config.max_checkpoints,
                environment_digest=str(identity["environment_digest"]),
                training_config_digest=content_digest(config.digest_payload()),
            )
            model.learn(total_timesteps=config.timesteps, callback=callback)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_target = output_path.with_suffix("")
            model.save(str(save_target))
            created = save_target.with_suffix(".zip")
            if created != output_path:
                created.replace(output_path)

            replay_buffer_path: Path | None = None
            replay_buffer_digest: str | None = None
            if config.algorithm != "ppo" and hasattr(model, "save_replay_buffer"):
                raw_replay = output_path.parent / ".replay-buffer.tmp.pkl"
                model.save_replay_buffer(str(raw_replay))
                replay_manifest = write_replay_buffer_artifact(
                    output_path.parent / "replay",
                    source=raw_replay,
                    algorithm=config.algorithm,
                    environment_digest=str(identity["environment_digest"]),
                    training_config_digest=content_digest(config.digest_payload()),
                    timesteps=int(model.num_timesteps),
                )
                raw_replay.unlink()
                replay_buffer_path = output_path.parent / "replay" / "replay-buffer.pkl"
                replay_buffer_digest = replay_manifest.artifact_digest

            return PolicyTrainingResult(
                checkpoint_path=output_path,
                actual_timesteps=int(model.num_timesteps),
                resolved_device=str(model.device),
                environment_digest=str(identity["environment_digest"]),
                initial_capital=float(identity["initial_capital"]),
                action_size=int(identity["action_size"]),
                action_names=tuple(identity["action_names"]),
                action_spec_digest=str(identity["action_spec_digest"]),
                observation_size=int(identity["observation_size"]),
                alpha_artifact_digest=identity["alpha_artifact_digest"],
                factor_artifact_digest=identity["factor_artifact_digest"],
                normalizer_digest=identity["normalizer_digest"],
                replay_buffer_path=replay_buffer_path,
                replay_buffer_digest=replay_buffer_digest,
            )
        finally:
            if probe is not None:
                probe.close()
            if environment is not None:
                environment.close()


class StableBaselines3PPOBackend(StableBaselines3Backend):
    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        if config.algorithm != "ppo":
            raise ValueError("StableBaselines3PPOBackend requires algorithm='ppo'")
        return super().train(seed=seed, config=config, output_path=output_path)


__all__ = ["StableBaselines3Backend", "StableBaselines3PPOBackend"]
