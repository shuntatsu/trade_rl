"""Stable-Baselines3 training adapter isolated from the RL core contracts."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.learning import (
    BehaviorCloningConfig,
    OracleTeacherConfig,
    StructuredTeacherObservationProvider,
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    load_teacher_artifact,
    oracle_target_path,
    write_teacher_artifact,
)
from trade_rl.rl.algorithm_configs import (
    CostCriticPPOConfig,
    LagrangianPPOConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    build_algorithm_config,
)
from trade_rl.rl.replay import (
    load_replay_buffer_artifact,
    write_replay_buffer_artifact,
)
from trade_rl.rl.rollout_memory import (
    estimate_index_backed_ppo_rollout_buffer_bytes,
    estimate_ppo_rollout_buffer_bytes,
)
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    _environment_identity,
    _validate_training_environment,
)


def _configure_torch_cuda_runtime(torch: Any, device: object) -> dict[str, object]:
    """Enable maintained CUDA fast paths without reducing training work."""

    requested = str(device).strip().lower()
    uses_cuda = requested == "auto" and bool(torch.cuda.is_available())
    if requested != "auto":
        try:
            uses_cuda = torch.device(device).type == "cuda"
        except (RuntimeError, TypeError, ValueError):
            uses_cuda = False
    if uses_cuda:
        # Ampere-and-newer GPUs accelerate float32 matrix products through TF32.
        # Parameters, optimizer state, losses, and checkpoints remain float32.
        torch.set_float32_matmul_precision("high")
        # Sequence windows and mini-batches use a small fixed set of shapes, so
        # the one-time cuDNN search is amortized over the full training run.
        # SB3 enables deterministic cuDNN while seeding a CUDA model; that
        # forces a pathologically slow algorithm for our dilated Conv1d stack.
        # Reapply this after model construction/load as well as before it.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    bf16_supported = bool(
        uses_cuda
        and callable(getattr(torch.cuda, "is_bf16_supported", None))
        and torch.cuda.is_bf16_supported()
    )
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sequence_encoder_autocast": ("bfloat16" if bf16_supported else "disabled"),
    }


def _teacher_cache_key(
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    environment_digest: str,
    action_spec_digest: str,
    teacher_config_digest: str,
) -> str:
    return content_digest(
        {
            "action_spec_digest": action_spec_digest,
            "dataset_id": dataset_id,
            "environment_digest": environment_digest,
            "teacher_config_digest": teacher_config_digest,
            "train_range": train_range,
        }
    )


@dataclass(frozen=True, slots=True)
class _TeacherIdentity:
    """Content identity for non-oracle causal behavior-cloning teachers."""

    digest: str


def _behavior_cloning_quality(
    *,
    initial_mse: float,
    final_mse: float,
    required_relative_improvement: float,
) -> tuple[float, bool]:
    """Return a fail-closed relative improvement decision for BC warm starts."""

    if (
        not np.isfinite(initial_mse)
        or not np.isfinite(final_mse)
        or initial_mse < 0.0
        or final_mse < 0.0
    ):
        raise ValueError("behavior cloning MSE values must be finite and non-negative")
    denominator = max(initial_mse, float(np.finfo(np.float64).eps))
    relative_improvement = (initial_mse - final_mse) / denominator
    return (
        relative_improvement,
        relative_improvement >= required_relative_improvement,
    )


_HEAVY_TRAINING_INFO_KEYS = (
    "hybrid_execution",
    "shadow_execution",
    "hybrid_liquidation",
    "shadow_liquidation",
)


def _compact_training_info(info: dict[str, object]) -> dict[str, object]:
    """Keep callback diagnostics without copying the environment's histories.

    ``DummyVecEnv`` deep-copies every info mapping.  Execution results reference
    ``BookState`` objects whose return histories grow for the whole episode, so
    exposing them to SB3 turns rollout collection into quadratic work.  The raw
    Gymnasium environment retains its rich diagnostic contract; only the
    training adapter replaces those objects with the small fields consumed by
    telemetry and callbacks.
    """

    compact = dict(info)
    execution = compact.get("hybrid_execution")
    book = getattr(execution, "book", None)
    weights = getattr(book, "weights", None)
    if weights is not None:
        compact["telemetry_weights_after"] = np.asarray(
            weights,
            dtype=np.float64,
        ).copy()
    if "telemetry_risk_reasons" not in compact:
        risk = compact.get("hybrid_risk")
        reasons = getattr(risk, "reasons", ())
        compact["telemetry_risk_reasons"] = tuple(
            str(getattr(item, "value", item)) for item in reasons if str(item)
        )
    for key in _HEAVY_TRAINING_INFO_KEYS:
        compact.pop(key, None)
    return compact


class _TrainingInfoFilter(gym.Wrapper):
    """Remove history-bearing diagnostics before SB3 copies vector infos."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            _compact_training_info(info),
        )


def _filtered_training_environment(factory: Callable[[], Any]) -> Any:
    return _TrainingInfoFilter(factory())


def _build_training_environment(
    factory: Callable[[], Any],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()

    if subprocesses:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(
            [factory for _ in range(n_envs)],
            start_method="spawn",
        )
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([factory for _ in range(n_envs)])


class StableBaselines3Backend:
    """Train one policy with an optional SB3-family algorithm."""

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
        resume_checkpoint_artifacts: Mapping[int, Path] | None = None,
    ) -> None:
        self.environment_factory = environment_factory
        self.verbose = verbose
        self.resume_replay_artifact = resume_replay_artifact
        self.resume_checkpoint_artifacts = dict(resume_checkpoint_artifacts or {})
        raw_teacher_cache = os.environ.get("TRADE_RL_TEACHER_CACHE_ROOT", "").strip()
        self.teacher_cache_root = (
            None if not raw_teacher_cache else Path(raw_teacher_cache).resolve()
        )
        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._trend_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], SupervisedPolicyDataset
        ] = {}

    def _oracle_targets(
        self,
        dataset: Any,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
    ) -> np.ndarray:
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("oracle dataset must expose dataset_id")
        start, stop = train_range
        key = (dataset_id, int(start), int(stop), teacher_config.digest)
        cached = self._oracle_target_cache.get(key)
        if cached is not None:
            return cached
        targets = np.asarray(
            oracle_target_path(dataset, train_range, teacher_config),
            dtype=np.float32,
        ).copy(order="C")
        targets.setflags(write=False)
        self._oracle_target_cache[key] = targets
        return targets

    def _trend_baseline_targets(
        self,
        dataset: Any,
        train_range: tuple[int, int],
        strategy: Any,
        *,
        teacher_digest: str,
    ) -> np.ndarray:
        """Return causal base-trend targets aligned with policy decisions."""

        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("trend teacher dataset must expose dataset_id")
        start, stop = train_range
        key = (dataset_id, int(start), int(stop), teacher_digest)
        cached = self._trend_target_cache.get(key)
        if cached is not None:
            return cached
        targets = np.stack(
            [
                np.asarray(strategy.targets(dataset, index).base, dtype=np.float32)
                for index in range(start, stop - 1)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if targets.ndim != 2 or len(targets) != stop - start - 1:
            raise RuntimeError("causal trend teacher target shape mismatch")
        if not np.isfinite(targets).all():
            raise RuntimeError("causal trend teacher targets are non-finite")
        targets.setflags(write=False)
        self._trend_target_cache[key] = targets
        return targets

    def _teacher_dataset(
        self,
        environment: Any,
        targets: np.ndarray,
        *,
        dataset_id: str,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig | _TeacherIdentity,
    ) -> SupervisedPolicyDataset:
        start, stop = train_range
        environment_digest = getattr(environment, "environment_digest", None)
        action_spec_digest = getattr(environment, "action_spec_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError("teacher environment must expose environment_digest")
        if not isinstance(action_spec_digest, str):
            raise ValueError("teacher environment must expose action_spec_digest")
        key = (
            dataset_id,
            int(start),
            int(stop),
            environment_digest,
            action_spec_digest,
            teacher_config.digest,
        )
        cached = self._teacher_dataset_cache.get(key)
        if cached is not None:
            return cached
        cache_path: Path | None = None
        if self.teacher_cache_root is not None:
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config.digest,
            )
            if cache_path.exists():
                _, teacher_dataset = load_teacher_artifact(
                    cache_path,
                    expected_dataset_id=dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                    expected_train_range=(start, stop),
                )
                if teacher_dataset.teacher_config_digest != teacher_config.digest:
                    raise ValueError("cached teacher configuration identity mismatch")
                self._teacher_dataset_cache[key] = teacher_dataset
                return teacher_dataset
        teacher_dataset = collect_teacher_rollout(
            environment,
            targets,
            dataset_id=dataset_id,
            train_range=(start, stop),
            teacher_config_digest=teacher_config.digest,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_path.name}.", dir=str(cache_path.parent)
                )
            )
            try:
                write_teacher_artifact(temporary, teacher_dataset)
                try:
                    temporary.replace(cache_path)
                except FileExistsError:
                    # A concurrent equivalent trainer won the content-addressed race.
                    pass
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        self._teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        import torch
        from stable_baselines3 import PPO, SAC, TD3

        torch_runtime = _configure_torch_cuda_runtime(torch, config.device)

        probe = self.environment_factory()
        environment: Any | None = None
        try:
            identity = _environment_identity(probe)
            _validate_training_environment(identity, config)
            algorithm_config = build_algorithm_config(config)
            canonical_action_probe_evidence = None
            if isinstance(algorithm_config, LagrangianPPOConfig):
                from trade_rl.rl.lagrangian_probe import (
                    run_canonical_action_feasibility_probe,
                )

                canonical_action_probe_evidence = (
                    run_canonical_action_feasibility_probe(
                        environment_factory=self.environment_factory,
                        schema=algorithm_config.lagrangian_schema,
                        episode_count=algorithm_config.probe_episodes,
                        max_steps_per_episode=(
                            algorithm_config.probe_max_steps_per_episode
                        ),
                    )
                )
            rollout_buffer_bytes: int | None = None
            if isinstance(algorithm_config, PPOConfig):
                estimator = (
                    estimate_index_backed_ppo_rollout_buffer_bytes
                    if config.sequence_encoder
                    else estimate_ppo_rollout_buffer_bytes
                )
                rollout_buffer_bytes = estimator(
                    probe.observation_space,
                    n_steps=algorithm_config.n_steps,
                    n_envs=config.n_envs,
                    action_dim=int(identity["action_size"]),
                )
                if isinstance(algorithm_config, CostCriticPPOConfig):
                    from trade_rl.integrations.cost_rollout_buffer import (
                        estimate_cost_rollout_storage_bytes,
                    )

                    rollout_buffer_bytes += estimate_cost_rollout_storage_bytes(
                        algorithm_config.n_steps,
                        config.n_envs,
                        len(algorithm_config.cost_schema.names),
                    )
                if rollout_buffer_bytes > config.max_rollout_buffer_bytes:
                    raise ValueError(
                        "estimated PPO rollout buffer exceeds max_rollout_buffer_bytes: "
                        f"{rollout_buffer_bytes} > {config.max_rollout_buffer_bytes}"
                    )
            policy_kwargs: dict[str, Any]
            sequence_metadata: dict[str, Any] | None = None
            sequence_reconstructor: Any | None = None
            uses_shared_asset_actor = False
            if config.sequence_encoder:
                from trade_rl.rl.policies import (
                    SequenceAssetFeatureExtractor,
                    SharedPerAssetActorCriticPolicy,
                )

                unwrapped: Any = getattr(probe, "unwrapped", probe)
                metadata = getattr(unwrapped, "sequence_layout_metadata", None)
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "sequence training requires environment sequence metadata"
                    )
                sequence_metadata = dict(metadata)
                action_names = tuple(str(name) for name in identity["action_names"])
                uses_shared_asset_actor = int(identity["action_size"]) == int(
                    sequence_metadata["n_symbols"]
                ) and all(name.startswith("target_weight:") for name in action_names)
                from trade_rl.integrations.compact_rollout_buffer import (
                    SequenceRolloutReconstructor,
                )

                dataset = getattr(unwrapped, "dataset", None)
                sequence_builder = getattr(
                    unwrapped, "sequence_observation_builder", None
                )
                if dataset is None or sequence_builder is None:
                    raise ValueError(
                        "sequence training requires dataset-bound reconstruction metadata"
                    )
                sequence_reconstructor = SequenceRolloutReconstructor(
                    dataset=dataset,
                    builder=sequence_builder,
                    normalizer=getattr(unwrapped, "sequence_normalizer", None),
                    expected_dataset_id=dataset.dataset_id,
                    expected_layout_digest=sequence_builder.layout_digest(dataset),
                    policy_plane=getattr(unwrapped, "sequence_policy_plane", None),
                )
                policy_kwargs = {
                    "net_arch": {
                        "pi": list(config.policy_net_arch),
                        "vf": list(config.value_net_arch),
                    },
                    "features_extractor_class": SequenceAssetFeatureExtractor,
                    "features_extractor_kwargs": {
                        **sequence_metadata,
                        "sequence_capacity": config.sequence_capacity,
                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_attention_heads,
                        "attention_layers": config.sequence_attention_layers,
                        "dropout": config.sequence_dropout,
                    },
                }
                if uses_shared_asset_actor:
                    policy_kwargs.update(
                        {
                            "shared_actor_n_symbols": int(
                                sequence_metadata["n_symbols"]
                            ),
                            "shared_actor_d_model": config.sequence_d_model,
                            "shared_actor_global_dim": 128,
                            "shared_actor_net_arch": tuple(config.policy_net_arch),
                        }
                    )
            elif isinstance(algorithm_config, PPOConfig):
                policy_kwargs = {
                    "net_arch": {
                        "pi": list(algorithm_config.policy_net_arch),
                        "vf": list(algorithm_config.value_net_arch),
                    }
                }
            else:
                policy_kwargs = {
                    "net_arch": {
                        "pi": list(algorithm_config.policy_net_arch),
                        "qf": list(algorithm_config.value_net_arch),
                    }
                }
            if isinstance(algorithm_config, PPOConfig):
                policy_kwargs["log_std_init"] = algorithm_config.log_std_init
            if config.asset_set_encoder:
                from trade_rl.rl.policies import AssetSetFeatureExtractor

                asset_unwrapped: Any = getattr(probe, "unwrapped", probe)
                layout = getattr(asset_unwrapped, "layout", None)
                active_column = getattr(asset_unwrapped, "asset_active_column", None)
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
                environment = _TrainingInfoFilter(probe)
                probe = None
            else:
                probe_to_close = probe
                probe = None
                probe_to_close.close()
                environment = _build_training_environment(
                    lambda: _filtered_training_environment(self.environment_factory),
                    config.n_envs,
                    subprocesses=False,
                )
            policy_identifier: Any = (
                SharedPerAssetActorCriticPolicy
                if uses_shared_asset_actor
                else config.policy
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
                rollout_kwargs: dict[str, Any] = {}
                if config.sequence_encoder:
                    from trade_rl.integrations.compact_rollout_buffer import (
                        IndexBackedDictRolloutBuffer,
                    )

                    if sequence_reconstructor is None:
                        raise RuntimeError(
                            "sequence rollout reconstructor was not resolved"
                        )
                    rollout_kwargs["rollout_buffer_class"] = (
                        IndexBackedDictRolloutBuffer
                    )
                    rollout_kwargs["rollout_buffer_kwargs"] = {
                        "sequence_reconstructor": sequence_reconstructor
                    }
                ppo_kwargs: dict[str, Any] = {
                    "n_steps": algorithm_config.n_steps,
                    "batch_size": algorithm_config.batch_size,
                    "n_epochs": algorithm_config.n_epochs,
                    "gae_lambda": algorithm_config.gae_lambda,
                    "clip_range": algorithm_config.clip_range,
                    "normalize_advantage": algorithm_config.normalize_advantage,
                    "ent_coef": algorithm_config.ent_coef,
                    "vf_coef": algorithm_config.vf_coef,
                    "max_grad_norm": algorithm_config.max_grad_norm,
                    "target_kl": algorithm_config.target_kl,
                    "use_sde": algorithm_config.use_sde,
                    "sde_sample_freq": algorithm_config.sde_sample_freq,
                    **rollout_kwargs,
                    **common,
                }
                if isinstance(algorithm_config, LagrangianPPOConfig):
                    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

                    model = LagrangianPPO(
                        policy_identifier,
                        environment,
                        cost_schema=algorithm_config.cost_schema,
                        cost_learning_rate=algorithm_config.cost_learning_rate,
                        cost_n_epochs=algorithm_config.cost_n_epochs,
                        cost_batch_size=algorithm_config.cost_batch_size,
                        cost_continuous_hidden_dims=(
                            algorithm_config.cost_continuous_hidden_dims
                        ),
                        cost_event_hidden_dims=algorithm_config.cost_event_hidden_dims,
                        cost_max_grad_norm=algorithm_config.cost_max_grad_norm,
                        lagrangian_schema=algorithm_config.lagrangian_schema,
                        canonical_action_probe_evidence=(
                            canonical_action_probe_evidence
                        ),
                        **ppo_kwargs,
                    )
                    model.canonical_action_probe_evidence = (
                        canonical_action_probe_evidence
                    )
                elif isinstance(algorithm_config, CostCriticPPOConfig):
                    from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

                    model = CostCriticPPO(
                        policy_identifier,
                        environment,
                        cost_schema=algorithm_config.cost_schema,
                        cost_learning_rate=algorithm_config.cost_learning_rate,
                        cost_n_epochs=algorithm_config.cost_n_epochs,
                        cost_batch_size=algorithm_config.cost_batch_size,
                        cost_continuous_hidden_dims=(
                            algorithm_config.cost_continuous_hidden_dims
                        ),
                        cost_event_hidden_dims=algorithm_config.cost_event_hidden_dims,
                        cost_max_grad_norm=algorithm_config.cost_max_grad_norm,
                        **ppo_kwargs,
                    )
                else:
                    model = PPO(policy_identifier, environment, **ppo_kwargs)
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
            resume_manifest = None
            resume_root = self.resume_checkpoint_artifacts.get(seed)
            if resume_root is not None:
                from trade_rl.rl.checkpointing import (
                    load_checkpoint_manifest,
                    validate_checkpoint_algorithm_identity,
                )

                manifest_path = Path(resume_root)
                if manifest_path.is_dir():
                    manifest_path = manifest_path / "checkpoint.json"
                resume_manifest = load_checkpoint_manifest(manifest_path)
                expected_training_digest = content_digest(config.digest_payload())
                if resume_manifest.algorithm != config.algorithm:
                    raise ValueError("checkpoint algorithm mismatch")
                if resume_manifest.seed != seed:
                    raise ValueError("checkpoint seed mismatch")
                if resume_manifest.environment_digest != identity["environment_digest"]:
                    raise ValueError("checkpoint environment identity mismatch")
                if resume_manifest.training_config_digest != expected_training_digest:
                    raise ValueError("checkpoint training configuration mismatch")
                expected_algorithm_identity = (
                    model.checkpoint_identity_payload()
                    if isinstance(algorithm_config, CostCriticPPOConfig)
                    else None
                )
                validate_checkpoint_algorithm_identity(
                    resume_manifest,
                    expected_algorithm_identity,
                )
                algorithm_class: Any
                if config.algorithm == "ppo":
                    algorithm_class = PPO
                elif config.algorithm == "cost_critic_ppo":
                    from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

                    algorithm_class = CostCriticPPO
                elif config.algorithm == "lagrangian_ppo":
                    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

                    algorithm_class = LagrangianPPO
                elif config.algorithm == "sac":
                    algorithm_class = SAC
                elif config.algorithm == "td3":
                    algorithm_class = TD3
                else:
                    from sb3_contrib import TQC

                    algorithm_class = TQC
                model = algorithm_class.load(
                    str(resume_manifest.policy_path),
                    env=environment,
                    device=config.device,
                )
                if int(model.num_timesteps) != resume_manifest.observed_timestep:
                    raise ValueError("checkpoint timestep identity mismatch")
                if isinstance(algorithm_config, CostCriticPPOConfig):
                    validate_checkpoint_algorithm_identity(
                        resume_manifest,
                        model.checkpoint_identity_payload(),
                    )
                if config.sequence_encoder:
                    if sequence_reconstructor is None:
                        raise RuntimeError("sequence reconstructor was not resolved")
                    rollout_buffer = getattr(model, "rollout_buffer", None)
                    binder = getattr(
                        rollout_buffer, "bind_sequence_reconstructor", None
                    )
                    if not callable(binder):
                        raise ValueError(
                            "checkpoint rollout buffer cannot bind sequences"
                        )
                    binder(sequence_reconstructor)
                    model.rollout_buffer_kwargs = {
                        "sequence_reconstructor": sequence_reconstructor
                    }

            # Stable-Baselines3 seeds CUDA during model construction/loading and
            # resets cuDNN to its deterministic, slow dilated-convolution path.
            # Capture and persist the effective post-construction runtime state.
            torch_runtime = _configure_torch_cuda_runtime(torch, config.device)

            parameter_count = sum(
                int(parameter.numel()) for parameter in model.policy.parameters()
            )
            if parameter_count > config.max_policy_parameters:
                raise ValueError(
                    "policy parameter count exceeds max_policy_parameters: "
                    f"{parameter_count} > {config.max_policy_parameters}"
                )
            declared_distribution = getattr(
                model.policy, "action_distribution_name", None
            )
            action_distribution = (
                declared_distribution
                if isinstance(declared_distribution, str) and declared_distribution
                else type(getattr(model.policy, "action_dist", None)).__name__
            )
            architecture_details: dict[str, object] = {
                "action_distribution": action_distribution,
                "actor_net_arch": config.policy_net_arch,
                "critic_net_arch": config.value_net_arch,
                "sequence_encoder": config.sequence_encoder,
            }
            if isinstance(algorithm_config, CostCriticPPOConfig):
                architecture_details["cost_critic"] = {
                    "architecture_digest": model.cost_critic.architecture_digest,
                    "continuous_hidden_dims": (
                        algorithm_config.cost_continuous_hidden_dims
                    ),
                    "cost_names": algorithm_config.cost_schema.names,
                    "cost_schema_digest": algorithm_config.cost_schema.digest,
                    "event_hidden_dims": algorithm_config.cost_event_hidden_dims,
                }
            if isinstance(algorithm_config, LagrangianPPOConfig):
                architecture_details["lagrangian"] = {
                    "actor_composition_mode": (algorithm_config.actor_composition_mode),
                    "completion_semantics": ("economic_time_limit_censored_shadow_v1"),
                    "probe_episodes": algorithm_config.probe_episodes,
                    "probe_max_steps_per_episode": (
                        algorithm_config.probe_max_steps_per_episode
                    ),
                    "schema": algorithm_config.lagrangian_schema.digest_payload(),
                    "schema_digest": algorithm_config.lagrangian_schema.digest,
                }
                if canonical_action_probe_evidence is None:
                    raise RuntimeError(
                        "canonical action probe evidence is unavailable"
                    )
                architecture_details["lagrangian_probe"] = {
                    "digest": canonical_action_probe_evidence.digest,
                    "payload": (
                        canonical_action_probe_evidence.digest_payload()
                    ),
                    "violated_costs": list(
                        canonical_action_probe_evidence.violated_costs
                    ),
                    "warning": canonical_action_probe_evidence.warning,
                }
            if config.sequence_encoder:
                if sequence_metadata is None:
                    raise RuntimeError("sequence metadata was not resolved")
                extractor = getattr(model.policy, "features_extractor", None)
                asset_encoder = getattr(extractor, "asset_encoder", None)
                timeframe_encoders = getattr(asset_encoder, "timeframe_encoders", None)
                if asset_encoder is None or timeframe_encoders is None:
                    raise ValueError(
                        "sequence policy does not expose its maintained timeframe encoders"
                    )
                architecture_details.update(
                    {
                        "actor_head": "shared_per_asset_v1",
                        "actor_parameter_sharing": "one_head_all_assets",
                        "actor_symbol_order": tuple(identity["action_names"]),
                        "encoder": "MultiTimeframeTCNEncoder",
                        "feature_counts": dict(sequence_metadata["feature_counts"]),
                        "window_lengths": dict(sequence_metadata["window_lengths"]),
                        "sequence_capacity": config.sequence_capacity,
                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_attention_heads,
                        "attention_layers": config.sequence_attention_layers,
                        "receptive_fields": {
                            timeframe: int(
                                timeframe_encoders[timeframe].receptive_field
                            )
                            for timeframe in ("15m", "1h", "4h", "1d")
                        },
                        "dilations": {
                            timeframe: tuple(
                                int(value)
                                for value in timeframe_encoders[timeframe].dilations
                            )
                            for timeframe in ("15m", "1h", "4h", "1d")
                        },
                        "encoder_widths": {
                            timeframe: tuple(
                                int(value)
                                for value in asset_encoder.architecture.encoder_widths[
                                    timeframe
                                ]
                            )
                            for timeframe in ("15m", "1h", "4h", "1d")
                        },
                    }
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            (output_path.parent / "model-architecture.json").write_bytes(
                canonical_json_bytes(
                    {
                        "architecture": architecture_details,
                        "environment_digest": identity["environment_digest"],
                        "observation_contract_digest": identity[
                            "observation_contract_digest"
                        ],
                        "observation_schema": identity["observation_schema"],
                        "parameter_count": parameter_count,
                        "policy": (
                            policy_identifier.__name__
                            if isinstance(policy_identifier, type)
                            else policy_identifier
                        ),
                        "rollout_buffer_bytes": rollout_buffer_bytes,
                        "torch_runtime": torch_runtime,
                        "rollout_buffer": (
                            "index_backed_dict"
                            if config.sequence_encoder
                            else "default"
                        ),
                        "vector_environment": (
                            "native" if config.n_envs == 1 else "dummy"
                        ),
                        "schema_version": "policy_architecture_v2",
                        "training_config_digest": content_digest(
                            config.digest_payload()
                        ),
                    }
                )
            )
            if config.behavior_cloning_epochs > 0 and resume_manifest is None:
                teacher_environment = self.environment_factory()
                try:
                    teacher_identity = _environment_identity(teacher_environment)
                    if (
                        teacher_identity["environment_digest"]
                        != identity["environment_digest"]
                    ):
                        raise ValueError("teacher environment identity mismatch")
                    unwrapped_teacher: Any = getattr(
                        teacher_environment, "unwrapped", teacher_environment
                    )
                    action_names = tuple(teacher_identity["action_names"])
                    if not action_names or not all(
                        name.startswith("target_weight:") for name in action_names
                    ):
                        raise ValueError(
                            "behavior cloning requires direct target-weight actions"
                        )
                    dataset = unwrapped_teacher.dataset
                    train_range = (
                        int(unwrapped_teacher.minimum_start_index),
                        int(dataset.n_bars),
                    )
                    teacher_kind = config.behavior_cloning_teacher
                    if teacher_kind == "oracle":
                        risk_config = unwrapped_teacher.pre_trade_risk.config
                        teacher_config: Any = OracleTeacherConfig(
                            execution_cost=unwrapped_teacher.config.execution_cost,
                            portfolio_risk=unwrapped_teacher.portfolio_risk.config,
                            max_gross=risk_config.max_gross,
                            max_abs_weight=risk_config.max_abs_weight,
                            entry_threshold=risk_config.entry_threshold,
                            exit_threshold=risk_config.exit_threshold,
                            no_trade_band=risk_config.no_trade_band,
                            reference_portfolio_value=(
                                unwrapped_teacher.initial_capital
                            ),
                            signal_delay_decisions=(
                                unwrapped_teacher.config.signal_delay_decisions
                            ),
                        )
                        targets = self._oracle_targets(
                            dataset, train_range, teacher_config
                        )
                    else:
                        trend_strategy = getattr(
                            unwrapped_teacher, "trend_strategy", None
                        )
                        if trend_strategy is None or not callable(
                            getattr(trend_strategy, "targets", None)
                        ):
                            raise ValueError(
                                "trend behavior cloning requires a trend strategy"
                            )
                        teacher_config = _TeacherIdentity(
                            digest=content_digest(
                                {
                                    "schema_version": (
                                        "causal_trend_baseline_teacher_v1"
                                    ),
                                    "signal_delay_decisions": (
                                        unwrapped_teacher.config.signal_delay_decisions
                                    ),
                                    "trend": trend_strategy.config,
                                }
                            )
                        )
                        targets = self._trend_baseline_targets(
                            dataset,
                            train_range,
                            trend_strategy,
                            teacher_digest=teacher_config.digest,
                        )
                    teacher_dataset = self._teacher_dataset(
                        teacher_environment,
                        targets,
                        dataset_id=dataset.dataset_id,
                        train_range=train_range,
                        teacher_config=teacher_config,
                    )
                    teacher_digest = write_teacher_artifact(
                        output_path.parent / "teacher",
                        teacher_dataset,
                    )
                    observation_provider = None
                    if isinstance(teacher_dataset.observations, Mapping):
                        sequence_builder = getattr(
                            unwrapped_teacher, "sequence_observation_builder", None
                        )
                        if sequence_builder is None:
                            raise ValueError(
                                "structured teacher requires a sequence observation builder"
                            )
                        observation_provider = StructuredTeacherObservationProvider(
                            dataset=dataset,
                            sequence_builder=sequence_builder,
                            observations=teacher_dataset.observations,
                            sequence_normalizer=getattr(
                                unwrapped_teacher, "sequence_normalizer", None
                            ),
                            policy_plane=getattr(
                                unwrapped_teacher, "sequence_policy_plane", None
                            ),
                        )
                    cloning = pretrain_policy(
                        model.policy,
                        teacher_dataset,
                        config=BehaviorCloningConfig(
                            epochs=config.behavior_cloning_epochs,
                            learning_rate=config.behavior_cloning_learning_rate,
                            batch_size=config.behavior_cloning_batch_size,
                            validation_fraction=(
                                config.behavior_cloning_validation_fraction
                            ),
                            early_stopping_patience=config.behavior_cloning_patience,
                            minimum_improvement=(
                                config.behavior_cloning_minimum_improvement
                            ),
                        ),
                        seed=seed,
                        observation_provider=observation_provider,
                    )
                    required_relative_improvement = (
                        config.behavior_cloning_required_relative_improvement
                    )
                    relative_improvement, quality_passed = _behavior_cloning_quality(
                        initial_mse=cloning.initial_mse,
                        final_mse=cloning.final_mse,
                        required_relative_improvement=(required_relative_improvement),
                    )
                    cloning_payload = {
                        "artifact_digest": teacher_digest,
                        "behavior_cloning_digest": cloning.digest,
                        "final_mse": cloning.final_mse,
                        "initial_mse": cloning.initial_mse,
                        "sample_count": cloning.sample_count,
                        "validation_mse": cloning.validation_mse,
                        "validation_sample_count": cloning.validation_sample_count,
                        "best_epoch": cloning.best_epoch,
                        "quality_passed": quality_passed,
                        "relative_improvement": relative_improvement,
                        "required_relative_improvement": (
                            required_relative_improvement
                        ),
                        "teacher_kind": teacher_kind,
                        "schema_version": "behavior_cloning_run_v3",
                    }
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    (output_path.parent / "behavior-cloning.json").write_bytes(
                        canonical_json_bytes(cloning_payload)
                    )
                    if not quality_passed:
                        raise RuntimeError(
                            "behavior cloning failed the required relative "
                            "improvement gate"
                        )
                finally:
                    teacher_environment.close()
            from trade_rl.rl.checkpointing import build_checkpoint_callback

            if self.resume_replay_artifact is not None:
                if config.algorithm in {
                    "ppo",
                    "cost_critic_ppo",
                    "lagrangian_ppo",
                }:
                    raise ValueError(
                        "PPO-family algorithms cannot resume from a replay buffer"
                    )
                replay_manifest, resume_path = load_replay_buffer_artifact(
                    self.resume_replay_artifact
                )
                if replay_manifest.algorithm != config.algorithm:
                    raise ValueError("replay buffer algorithm mismatch")
                if replay_manifest.environment_digest != identity["environment_digest"]:
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
            remaining_timesteps = config.timesteps
            if resume_manifest is not None:
                remaining_timesteps = max(
                    0, config.timesteps - resume_manifest.observed_timestep
                )
            if remaining_timesteps > 0:
                learn_kwargs: dict[str, object] = {
                    "total_timesteps": remaining_timesteps,
                    "callback": callback,
                }
                if resume_manifest is not None:
                    learn_kwargs["reset_num_timesteps"] = False
                model.learn(**learn_kwargs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if resume_manifest is not None:
                (output_path.parent / "resume.json").write_bytes(
                    canonical_json_bytes(
                        {
                            "checkpoint_digest": resume_manifest.digest,
                            "checkpoint_observed_timestep": (
                                resume_manifest.observed_timestep
                            ),
                            "remaining_timesteps": remaining_timesteps,
                            "schema_version": "training_resume_v1",
                        }
                    )
                )
            save_target = output_path.with_suffix("")
            from trade_rl.rl.checkpointing import save_policy_without_runtime_state

            save_policy_without_runtime_state(model, str(save_target))
            created = save_target.with_suffix(".zip")
            if created != output_path:
                created.replace(output_path)

            replay_buffer_path: Path | None = None
            replay_buffer_digest: str | None = None
            if config.algorithm not in {
                "ppo",
                "cost_critic_ppo",
                "lagrangian_ppo",
            } and hasattr(model, "save_replay_buffer"):
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
                observation_schema=str(identity["observation_schema"]),
                observation_contract_digest=identity["observation_contract_digest"],
                parameter_count=parameter_count,
                rollout_buffer_bytes=rollout_buffer_bytes,
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
