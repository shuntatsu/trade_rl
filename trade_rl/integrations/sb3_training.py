"""Stable-Baselines3 training adapter isolated from the RL core contracts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.integrations.sb3_behavior_cloning import (
    _behavior_cloning_gate_thresholds as _behavior_cloning_gate_thresholds,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _behavior_cloning_quality as _behavior_cloning_quality,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _enforce_behavior_cloning_gates as _enforce_behavior_cloning_gates,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _evaluate_hierarchical_behavior_cloning_gate as _evaluate_hierarchical_behavior_cloning_gate,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _hierarchical_behavior_cloning_config as _hierarchical_behavior_cloning_config,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _hierarchical_teacher_labels as _hierarchical_teacher_labels,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _required_hierarchical_config as _required_hierarchical_config,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _resolve_behavior_cloning_seed as _resolve_behavior_cloning_seed,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _restore_member_seed_after_behavior_cloning as _restore_member_seed_after_behavior_cloning,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _save_behavior_cloning_policy_candidate as _save_behavior_cloning_policy_candidate,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _teacher_cache_key as _teacher_cache_key,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _teacher_change_labels as _teacher_change_labels,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _TeacherIdentity as _TeacherIdentity,
)
from trade_rl.integrations.sb3_behavior_cloning import (
    _uses_hierarchical_actor_head as _uses_hierarchical_actor_head,
)
from trade_rl.integrations.sb3_checkpoint_assembly import (
    load_sb3_checkpoint_model,
    load_sb3_checkpoint_transfer_model,
)
from trade_rl.integrations.sb3_environment import (
    _HEAVY_TRAINING_INFO_KEYS as _HEAVY_TRAINING_INFO_KEYS,
)
from trade_rl.integrations.sb3_environment import (
    _build_parallel_sequence_training_environment as _build_parallel_sequence_training_environment,
)
from trade_rl.integrations.sb3_environment import (
    _build_training_environment as _build_training_environment,
)
from trade_rl.integrations.sb3_environment import (
    _compact_filtered_training_environment as _compact_filtered_training_environment,
)
from trade_rl.integrations.sb3_environment import (
    _compact_training_info as _compact_training_info,
)
from trade_rl.integrations.sb3_environment import (
    _effective_vector_environment_kind as _effective_vector_environment_kind,
)
from trade_rl.integrations.sb3_environment import (
    _filtered_environment_factory as _filtered_environment_factory,
)
from trade_rl.integrations.sb3_environment import (
    _filtered_training_environment as _filtered_training_environment,
)
from trade_rl.integrations.sb3_environment import (
    _reset_observation_for_export as _reset_observation_for_export,
)
from trade_rl.integrations.sb3_environment import (
    _TrainingInfoFilter as _TrainingInfoFilter,
)
from trade_rl.integrations.sb3_model_assembly import (
    build_sb3_model,
    resolve_sb3_policy_assembly,
)
from trade_rl.integrations.sb3_runtime import (
    _configure_sequence_runtime as _configure_sequence_runtime,
)
from trade_rl.integrations.sb3_runtime import (
    _configure_torch_cuda_runtime as _configure_torch_cuda_runtime,
)
from trade_rl.integrations.sb3_runtime import (
    _lagrangian_probe_worker_count as _lagrangian_probe_worker_count,
)
from trade_rl.integrations.sb3_runtime import (
    _oracle_accelerator_backend as _oracle_accelerator_backend,
)
from trade_rl.integrations.sb3_runtime import (
    _oracle_episode_sampling_config as _oracle_episode_sampling_config,
)
from trade_rl.integrations.sb3_runtime import (
    _oracle_solver_config as _oracle_solver_config,
)
from trade_rl.integrations.sb3_runtime import (
    _teacher_worker_count as _teacher_worker_count,
)
from trade_rl.integrations.sb3_runtime import (
    oracle_teacher_config_for_environment,
)
from trade_rl.integrations.sb3_teacher_pipeline import (
    _StableBaselines3TeacherPipeline,
)
from trade_rl.integrations.sb3_universal_pretraining import (
    _apply_universal_pretraining_if_configured,
)
from trade_rl.integrations.sb3_universal_pretraining import (
    _run_behavior_cloning_critic_warm_start_if_enabled as _run_critic_warm_start_helper,
)
from trade_rl.integrations.universal_critic_warm_start import (
    run_configured_critic_warm_start,
)
from trade_rl.learning import (
    BehaviorCloningConfig,
    BehaviorCloningGateEvaluation,
    BehaviorCloningHoldoutEvaluation,
    OracleTeacherConfig,
    StructuredTeacherObservationProvider,
    SupervisedPolicyDataset,
    write_learning_evaluation,
    write_teacher_artifact,
)
from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
from trade_rl.learning.episode_behavior_cloning import (
    BehaviorCloningSplit,
    align_behavior_cloning_validation,
)
from trade_rl.learning.episode_oracle_bc import (
    EpisodeBehaviorCloningHoldoutEvaluation,
    evaluate_episode_behavior_cloning_holdout,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.learning.episode_teacher_artifact import (
    EpisodeSupervisedPolicyDataset,
    write_episode_teacher_artifact,
)
from trade_rl.rl.algorithm_configs import (
    CostCriticPPOConfig,
    LagrangianPPOConfig,
    build_algorithm_config,
)
from trade_rl.rl.replay import (
    load_replay_buffer_artifact,
    verified_replay_buffer_copy,
    write_replay_buffer_artifact,
)
from trade_rl.rl.tensorboard_logging import build_tensorboard_metrics_callback
from trade_rl.rl.training import PolicyTrainingResult, ResidualTrainingConfig
from trade_rl.rl.training_environment_contract import (
    training_environment_identity,
    validate_training_environment,
)
from trade_rl.rl.training_performance import (
    TrainingPerformanceRecorder,
    activate_training_performance,
    write_training_performance_evidence,
)


def _run_behavior_cloning_critic_warm_start_if_enabled(**kwargs: Any) -> Any:
    return _run_critic_warm_start_helper(
        **kwargs, run_warm_start=run_configured_critic_warm_start
    )


def _resolved_vector_environment_kind(
    config: ResidualTrainingConfig,
    *,
    sequence_reconstructor: object | None,
) -> str:
    kind = _effective_vector_environment_kind(config)
    if kind == "subprocess_compact_sequence" and sequence_reconstructor is None:
        return "subprocess"
    return kind


def _publish_final_training_checkpoint(
    *,
    model: Any,
    output_root: Path,
    config: Any,
    seed: int,
    environment_digest: str,
    target_total_timesteps: int,
) -> Any:
    """Publish the exact completed policy as the retained Stage A checkpoint."""

    observed_timestep = getattr(model, "num_timesteps", None)
    if (
        isinstance(target_total_timesteps, bool)
        or not isinstance(target_total_timesteps, int)
        or target_total_timesteps <= 0
    ):
        raise ValueError("target_total_timesteps must be a positive integer")
    if (
        isinstance(observed_timestep, bool)
        or not isinstance(observed_timestep, int)
        or observed_timestep < target_total_timesteps
    ):
        raise RuntimeError("model has not reached the target training horizon")
    algorithm = getattr(config, "algorithm", None)
    digest_payload = getattr(config, "digest_payload", None)
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("training algorithm identity is unavailable")
    if not callable(digest_payload):
        raise TypeError("training config must expose digest_payload")
    from trade_rl.rl.checkpointing import publish_checkpoint

    return publish_checkpoint(
        model=model,
        checkpoint_root=Path(output_root) / "checkpoints",
        algorithm=algorithm,
        seed=seed,
        requested_timestep=target_total_timesteps,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=content_digest(digest_payload()),
    )


class StableBaselines3Backend(_StableBaselines3TeacherPipeline):
    """Train one policy with an optional SB3-family algorithm."""

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
        resume_checkpoint_artifacts: Mapping[int, Path] | None = None,
        transfer_checkpoint_artifacts: Mapping[int, Path] | None = None,
        universal_pretraining_hook: (Callable[..., Mapping[str, object]] | None) = None,
        structured_export_enabled: bool = False,
        structured_export_tolerance: float = 1e-5,
    ) -> None:
        self.environment_factory = environment_factory
        self.verbose = verbose
        self.resume_replay_artifact = resume_replay_artifact
        self.resume_checkpoint_artifacts = dict(resume_checkpoint_artifacts or {})
        self.transfer_checkpoint_artifacts = dict(transfer_checkpoint_artifacts or {})
        if universal_pretraining_hook is not None and not callable(
            universal_pretraining_hook
        ):
            raise TypeError("universal_pretraining_hook must be callable")
        self.universal_pretraining_hook = universal_pretraining_hook
        overlapping_seeds = (
            self.resume_checkpoint_artifacts.keys()
            & self.transfer_checkpoint_artifacts.keys()
        )
        if overlapping_seeds:
            raise ValueError(
                "checkpoint resume and transfer cannot target the same seed"
            )
        if (
            self.resume_replay_artifact is not None
            and self.transfer_checkpoint_artifacts
        ):
            raise ValueError(
                "replay resume cannot be combined with checkpoint transfer"
            )
        if not isinstance(structured_export_enabled, bool):
            raise ValueError("structured_export_enabled must be a boolean")
        if (
            not np.isfinite(structured_export_tolerance)
            or structured_export_tolerance <= 0.0
        ):
            raise ValueError("structured_export_tolerance must be finite and positive")
        self.structured_export_enabled = structured_export_enabled
        self.structured_export_tolerance = float(structured_export_tolerance)
        raw_teacher_cache = os.environ.get("TRADE_RL_TEACHER_CACHE_ROOT", "").strip()
        self.teacher_cache_root = (
            None if not raw_teacher_cache else Path(raw_teacher_cache).resolve()
        )
        self.reusable_artifact_index = (
            None
            if self.teacher_cache_root is None
            else ReusableArtifactIndex.from_environment(
                storage_root=self.teacher_cache_root
            )
        )
        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._oracle_episode_batch_cache: dict[
            tuple[str, int, int, str, str, str], EpisodeOracleBatch
        ] = {}
        self._trend_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], SupervisedPolicyDataset
        ] = {}
        self._episode_teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], EpisodeSupervisedPolicyDataset
        ] = {}

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        if self.structured_export_enabled and (
            config.observation_encoder != "hierarchical_sequence_v2"
        ):
            raise ValueError("structured export requires hierarchical_sequence_v2")

        probe: Any | None = None
        environment: Any | None = None
        try:
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
                        max_workers=_lagrangian_probe_worker_count(config.n_envs),
                    )
                )
            # A full-market environment is several GiB. Do not keep the
            # identity probe alive while the isolated canonical probe creates
            # its own environment.
            probe = self.environment_factory()
            identity = training_environment_identity(probe)
            validate_training_environment(identity, config)
            resume_root = self.resume_checkpoint_artifacts.get(seed)
            transfer_root = self.transfer_checkpoint_artifacts.get(seed)
            fresh_behavior_cloning = (
                config.behavior_cloning_epochs > 0
                and resume_root is None
                and transfer_root is None
            )
            behavior_cloning_seed = _resolve_behavior_cloning_seed(
                config,
                member_seed=seed,
            )
            prefetched_episode_batch: EpisodeOracleBatch | None = None
            prefetched_episode_teacher: EpisodeSupervisedPolicyDataset | None = None
            prefetched_oracle_config: OracleTeacherConfig | None = None
            if (
                fresh_behavior_cloning
                and self.universal_pretraining_hook is None
                and config.behavior_cloning_teacher == "oracle"
            ):
                unwrapped_probe: Any = getattr(probe, "unwrapped", probe)
                probe_action_names = tuple(identity["action_names"])
                if not probe_action_names or not all(
                    name.startswith("target_weight:") for name in probe_action_names
                ):
                    raise ValueError(
                        "behavior cloning requires direct target-weight actions"
                    )
                probe_dataset = unwrapped_probe.dataset
                probe_train_range = (
                    int(unwrapped_probe.minimum_start_index),
                    int(probe_dataset.n_bars),
                )
                prefetched_oracle_config = oracle_teacher_config_for_environment(
                    unwrapped_probe
                )
                sampling_config = _oracle_episode_sampling_config(
                    unwrapped_probe,
                    train_range=probe_train_range,
                    seed=behavior_cloning_seed,
                )
                oracle_solver_config = _oracle_solver_config()
                teacher_workers = _teacher_worker_count(
                    config.n_envs,
                    solver_config=oracle_solver_config,
                )
                prefetched_episode_batch = self._oracle_episode_batch(
                    unwrapped_probe,
                    probe_train_range,
                    prefetched_oracle_config,
                    sampling_config,
                    max_workers=teacher_workers,
                    solver_config=oracle_solver_config,
                )
                prefetched_episode_teacher = self._episode_teacher_dataset(
                    probe,
                    prefetched_episode_batch,
                    train_range=probe_train_range,
                    teacher_config=prefetched_oracle_config,
                    max_workers=teacher_workers,
                )
            import torch

            torch_runtime = _configure_torch_cuda_runtime(
                torch,
                config.device,
                config.cuda_runtime_mode,
            )
            policy = resolve_sb3_policy_assembly(
                probe=probe,
                identity=identity,
                config=config,
                algorithm_config=algorithm_config,
            )
            sequence_reconstructor = policy.sequence_reconstructor
            vector_environment_kind = _resolved_vector_environment_kind(
                config, sequence_reconstructor=sequence_reconstructor
            )
            full_observation_space = probe.observation_space

            def build_parallel_environment() -> Any:
                if vector_environment_kind == "subprocess_compact_sequence":
                    if sequence_reconstructor is None:
                        raise RuntimeError(
                            "parallel sequence environment requires a reconstructor"
                        )
                    if not isinstance(full_observation_space, gym.spaces.Dict):
                        raise RuntimeError(
                            "parallel sequence environment requires a Dict space"
                        )
                    return _build_parallel_sequence_training_environment(
                        self.environment_factory,
                        config.n_envs,
                        full_observation_space=full_observation_space,
                        reconstructor=sequence_reconstructor,
                    )
                return _build_training_environment(
                    _filtered_environment_factory(self.environment_factory),
                    config.n_envs,
                    subprocesses=vector_environment_kind == "subprocess",
                )

            if config.n_envs == 1:
                environment = _TrainingInfoFilter(probe)
                probe = None
            else:
                probe_to_close = probe
                probe = None
                probe_to_close.close()
                environment = build_parallel_environment()
            model = build_sb3_model(
                environment=environment,
                seed=(behavior_cloning_seed if fresh_behavior_cloning else seed),
                config=config,
                algorithm_config=algorithm_config,
                policy=policy,
                verbose=self.verbose,
                output_root=output_path,
                canonical_action_probe_evidence=canonical_action_probe_evidence,
            )
            resume_manifest = None
            transfer_manifest = None
            if resume_root is not None:
                loaded_checkpoint = load_sb3_checkpoint_model(
                    checkpoint_root=Path(resume_root),
                    environment=environment,
                    seed=seed,
                    config=config,
                    identity=identity,
                    algorithm_config=algorithm_config,
                    policy=policy,
                    fresh_model=model,
                )
                model = loaded_checkpoint.model
                resume_manifest = loaded_checkpoint.manifest
            elif transfer_root is not None:
                loaded_checkpoint = load_sb3_checkpoint_transfer_model(
                    checkpoint_root=Path(transfer_root),
                    environment=environment,
                    seed=seed,
                    config=config,
                    identity=identity,
                    algorithm_config=algorithm_config,
                    policy=policy,
                    fresh_model=model,
                )
                model = loaded_checkpoint.model
                transfer_manifest = loaded_checkpoint.manifest

            rollout_buffer_bytes = policy.rollout_buffer_bytes
            sequence_metadata = policy.sequence_metadata
            policy_identifier = policy.policy_identifier

            torch_runtime = _configure_torch_cuda_runtime(
                torch,
                config.device,
                config.cuda_runtime_mode,
            )
            sequence_runtime = _configure_sequence_runtime(torch, model, config)

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
                "observation_encoder": config.observation_encoder,
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
                    raise RuntimeError("canonical action probe evidence is unavailable")
                architecture_details["lagrangian_probe"] = {
                    "digest": canonical_action_probe_evidence.digest,
                    "payload": (canonical_action_probe_evidence.digest_payload()),
                    "workers": _lagrangian_probe_worker_count(config.n_envs),
                    "violated_costs": list(
                        canonical_action_probe_evidence.violated_costs
                    ),
                    "warning": canonical_action_probe_evidence.warning,
                }
            if config.observation_encoder == "hierarchical_sequence_v2":
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
                        "sequence_tcn_capacity": config.sequence_tcn_capacity,
                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_timeframe_attention_heads,
                        "attention_layers": config.sequence_timeframe_attention_layers,
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
                        "sequence_runtime": sequence_runtime,
                        "rollout_buffer": (
                            "index_backed_dict"
                            if (
                                config.observation_encoder == "hierarchical_sequence_v2"
                            )
                            else "default"
                        ),
                        "vector_environment": vector_environment_kind,
                        "schema_version": "policy_architecture_v2",
                        "training_config_digest": content_digest(
                            config.digest_payload()
                        ),
                    }
                )
            )
            if fresh_behavior_cloning != (
                config.behavior_cloning_epochs > 0
                and resume_manifest is None
                and transfer_manifest is None
            ):
                raise RuntimeError("behavior cloning resume state changed unexpectedly")
            environment_suspended_for_teacher = False
            if fresh_behavior_cloning and config.n_envs > 1:
                environment.close()
                environment = None
                environment_suspended_for_teacher = True
            if fresh_behavior_cloning and self.universal_pretraining_hook is None:
                teacher_environment = self.environment_factory()
                try:
                    teacher_identity = training_environment_identity(
                        teacher_environment
                    )
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
                    episode_batch: EpisodeOracleBatch | None = None
                    episode_split: BehaviorCloningSplit | None = None
                    teacher_dataset: SupervisedPolicyDataset
                    if teacher_kind == "oracle":
                        if (
                            prefetched_oracle_config is None
                            or prefetched_episode_batch is None
                            or prefetched_episode_teacher is None
                        ):
                            raise RuntimeError(
                                "prefetched Oracle teacher evidence is unavailable"
                            )
                        teacher_config: Any = prefetched_oracle_config
                        episode_batch = prefetched_episode_batch
                        targets = np.concatenate(episode_batch.targets, axis=0)
                        teacher_dataset = prefetched_episode_teacher
                        teacher_digest = write_episode_teacher_artifact(
                            output_path.parent / "teacher",
                            teacher_dataset,
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
                    teacher_change_labels = _teacher_change_labels(
                        teacher_dataset=teacher_dataset,
                        config=config,
                    )
                    hierarchical_labels = (
                        teacher_change_labels
                        if _uses_hierarchical_actor_head(model.policy)
                        else None
                    )
                    cloning_config = (
                        _hierarchical_behavior_cloning_config(config)
                        if hierarchical_labels is not None
                        else BehaviorCloningConfig(
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
                        )
                    )
                    if episode_batch is not None:
                        cloning_config, episode_split = (
                            align_behavior_cloning_validation(
                                cloning_config,
                                teacher_dataset,
                            )
                        )
                    behavior_cloning_progress_path = (
                        output_path.parent / "behavior-cloning-progress.json"
                    )
                    behavior_cloning_progress_state: dict[str, object] = {
                        "phase": "training",
                        "seed": seed,
                        "behavior_cloning_seed": behavior_cloning_seed,
                    }

                    def write_behavior_cloning_progress(
                        progress: Mapping[str, object],
                    ) -> None:
                        behavior_cloning_progress_state.update(progress)
                        atomic_write_bytes(
                            behavior_cloning_progress_path,
                            canonical_json_bytes(
                                {
                                    "schema_version": "behavior_cloning_progress_v1",
                                    "updated_at": datetime.now(UTC).isoformat(),
                                    **behavior_cloning_progress_state,
                                }
                            ),
                        )

                    cloning = pretrain_policy(
                        model.policy,
                        teacher_dataset,
                        config=cloning_config,
                        split=episode_split,
                        seed=behavior_cloning_seed,
                        observation_provider=observation_provider,
                        hierarchical_labels=hierarchical_labels,
                        progress_callback=write_behavior_cloning_progress,
                    )
                    write_behavior_cloning_progress(
                        {
                            "phase": "evaluating",
                            "epoch": cloning.best_epoch,
                            "total_epochs": cloning.config.epochs,
                            "best_epoch": cloning.best_epoch,
                            "validation_loss": (
                                cloning.validation_mse
                                if cloning.validation_hierarchical_losses is None
                                else cloning.validation_hierarchical_losses.weighted
                            ),
                            "gate_precision": None,
                            "gate_recall": None,
                            "activity_ratio": None,
                        }
                    )
                    required_relative_improvement = (
                        config.behavior_cloning_required_relative_improvement
                    )
                    relative_improvement, quality_passed = _behavior_cloning_quality(
                        initial_mse=cloning.initial_mse,
                        final_mse=cloning.final_mse,
                        required_relative_improvement=(required_relative_improvement),
                    )
                    oracle_audit_payload: dict[str, object] | None = None
                    holdout_evaluation: (
                        BehaviorCloningHoldoutEvaluation
                        | EpisodeBehaviorCloningHoldoutEvaluation
                        | None
                    ) = None
                    if teacher_kind == "oracle":
                        if episode_batch is None or episode_split is None:
                            raise RuntimeError(
                                "Oracle episode teacher evidence is unavailable"
                            )
                        (
                            oracle_audit_payload,
                            holdout_evaluation,
                        ) = evaluate_episode_behavior_cloning_holdout(
                            environment_factory=self.environment_factory,
                            model=model,
                            batch=episode_batch,
                            split=episode_split,
                            output_root=output_path.parent,
                            bootstrap_confidence_level=(
                                config.behavior_cloning_causal_holdout_confidence_level
                            ),
                            bootstrap_resamples=(
                                config.behavior_cloning_causal_holdout_bootstrap_resamples
                            ),
                        )
                    gate_evaluation: BehaviorCloningGateEvaluation | None = None
                    gate_evaluation_digest: str | None = None
                    if hierarchical_labels is not None:
                        gate_evaluation = _evaluate_hierarchical_behavior_cloning_gate(
                            cloning=cloning,
                            holdout=holdout_evaluation,
                            thresholds=_behavior_cloning_gate_thresholds(config),
                        )
                    elif teacher_kind == "oracle" and teacher_change_labels is not None:
                        gate_evaluation = evaluate_direct_behavior_cloning_gates(
                            initial_mse=cloning.initial_mse,
                            final_mse=cloning.final_mse,
                            teacher_change_support=(
                                teacher_change_labels.diagnostics.gate_positive_count
                            ),
                            holdout=holdout_evaluation,
                            thresholds=_behavior_cloning_gate_thresholds(config),
                        )
                    if gate_evaluation is not None:
                        gate_evaluation_digest = write_learning_evaluation(
                            output_path.parent / "behavior-cloning-gates.json",
                            gate_evaluation,
                        )
                        quality_passed = gate_evaluation.passed
                    (
                        behavior_cloning_policy_path,
                        behavior_cloning_policy_digest,
                    ) = _save_behavior_cloning_policy_candidate(
                        model,
                        output_dir=output_path.parent,
                    )
                    cloning_payload = {
                        "artifact_digest": teacher_digest,
                        "behavior_cloning_seed": behavior_cloning_seed,
                        "member_seed": seed,
                        "behavior_cloning_digest": cloning.digest,
                        "behavior_cloning_policy_digest": (
                            behavior_cloning_policy_digest
                        ),
                        "behavior_cloning_policy_file": (
                            behavior_cloning_policy_path.name
                        ),
                        "behavior_cloning_gate_digest": gate_evaluation_digest,
                        "behavior_cloning_gates": (
                            None
                            if gate_evaluation is None
                            else gate_evaluation.to_dict()
                        ),
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
                        "episode_sampling": (
                            None
                            if episode_batch is None
                            else {
                                "batch_digest": episode_batch.digest,
                                "decision_count": episode_batch.decision_count,
                                "episode_count": episode_batch.episode_count,
                                "sampling_config_digest": (
                                    episode_batch.sampling_config_digest
                                ),
                            }
                        ),
                        "oracle_reproduction": oracle_audit_payload,
                        "schema_version": "behavior_cloning_run_v6",
                    }
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    (output_path.parent / "behavior-cloning.json").write_bytes(
                        canonical_json_bytes(cloning_payload)
                    )
                    write_behavior_cloning_progress(
                        {
                            "phase": "passed" if quality_passed else "failed",
                            "epoch": cloning.best_epoch,
                            "total_epochs": cloning.config.epochs,
                            "best_epoch": cloning.best_epoch,
                            "validation_loss": (
                                cloning.validation_mse
                                if cloning.validation_hierarchical_losses is None
                                else cloning.validation_hierarchical_losses.weighted
                            ),
                            "gate_precision": (
                                None
                                if cloning.validation_hierarchical_metrics is None
                                else cloning.validation_hierarchical_metrics.gate_precision
                            ),
                            "gate_recall": (
                                None
                                if cloning.validation_hierarchical_metrics is None
                                else cloning.validation_hierarchical_metrics.gate_recall
                            ),
                            "activity_ratio": (
                                None
                                if cloning.validation_hierarchical_metrics is None
                                else cloning.validation_hierarchical_metrics.activity_ratio
                            ),
                        }
                    )
                    if gate_evaluation is not None:
                        _enforce_behavior_cloning_gates(gate_evaluation)
                    elif not quality_passed:
                        raise RuntimeError(
                            "behavior cloning failed the required MSE improvement gate"
                        )
                    _run_behavior_cloning_critic_warm_start_if_enabled(
                        policy=model.policy,
                        teacher_environment=teacher_environment,
                        teacher_dataset=teacher_dataset,
                        episode_batch=episode_batch,
                        episode_split=episode_split,
                        config=config,
                        observation_provider=observation_provider,
                        behavior_cloning_seed=behavior_cloning_seed,
                        output_root=output_path.parent,
                    )
                finally:
                    teacher_environment.close()
            if fresh_behavior_cloning and self.universal_pretraining_hook is not None:
                _apply_universal_pretraining_if_configured(
                    hook=self.universal_pretraining_hook,
                    policy=model.policy,
                    config=config,
                    behavior_cloning_seed=behavior_cloning_seed,
                    member_seed=seed,
                    output_root=output_path.parent,
                )
            if environment_suspended_for_teacher:
                environment = build_parallel_environment()
                model.set_env(environment)
            if fresh_behavior_cloning:
                _restore_member_seed_after_behavior_cloning(
                    model,
                    behavior_cloning_seed=behavior_cloning_seed,
                    member_seed=seed,
                )
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
                with verified_replay_buffer_copy(
                    replay_manifest,
                    resume_path,
                ) as verified_resume_path:
                    model.load_replay_buffer(str(verified_resume_path))

            remaining_timesteps = config.timesteps
            starting_timestep = 0
            target_total_timesteps = config.timesteps
            if resume_manifest is not None:
                starting_timestep = resume_manifest.observed_timestep
                remaining_timesteps = max(0, config.timesteps - starting_timestep)
            elif transfer_manifest is not None:
                starting_timestep = transfer_manifest.observed_timestep
                target_total_timesteps = starting_timestep + config.timesteps
            checkpoint_callback = build_checkpoint_callback(
                checkpoint_root=output_path.parent / "checkpoints",
                algorithm=config.algorithm,
                seed=seed,
                interval_steps=(
                    0
                    if config.max_checkpoints == 1
                    else config.resolved_checkpoint_interval
                ),
                max_checkpoints=max(1, config.max_checkpoints - 1),
                total_timesteps=target_total_timesteps,
                starting_timestep=starting_timestep,
                environment_digest=str(identity["environment_digest"]),
                training_config_digest=content_digest(config.digest_payload()),
                sequence_diagnostics_enabled=config.tensorboard_enabled,
                sequence_diagnostics_interval=config.tensorboard_log_interval,
            )
            metrics_callback = build_tensorboard_metrics_callback(
                enabled=config.tensorboard_enabled,
                log_interval=config.tensorboard_log_interval,
            )
            callback: object = checkpoint_callback
            if metrics_callback is not None:
                from stable_baselines3.common.callbacks import CallbackList

                callback = CallbackList([checkpoint_callback, metrics_callback])
            if config.tensorboard_enabled:
                model.tensorboard_log = str(output_path.parent / "tensorboard")
            if remaining_timesteps > 0:
                learn_kwargs: dict[str, object] = {
                    "total_timesteps": remaining_timesteps,
                    "callback": callback,
                }
                if config.tensorboard_enabled:
                    learn_kwargs["tb_log_name"] = f"seed-{seed}-{config.algorithm}"
                if resume_manifest is not None or transfer_manifest is not None:
                    learn_kwargs["reset_num_timesteps"] = False
                performance = TrainingPerformanceRecorder()
                performance.start(torch_module=torch, device=model.device)
                learn_start_timestep = int(model.num_timesteps)
                with (
                    activate_training_performance(performance),
                    performance.instrument_model(model),
                ):
                    model.learn(**learn_kwargs)
                performance_evidence = performance.finish(
                    torch_module=torch,
                    device=model.device,
                    requested_environment_steps=remaining_timesteps,
                    observed_environment_steps=(
                        int(model.num_timesteps) - learn_start_timestep
                    ),
                )
                write_training_performance_evidence(
                    output_path.parent / "training-performance.json",
                    performance_evidence,
                )
            _publish_final_training_checkpoint(
                model=model,
                output_root=output_path.parent,
                config=config,
                seed=seed,
                environment_digest=str(identity["environment_digest"]),
                target_total_timesteps=target_total_timesteps,
            )
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
            elif transfer_manifest is not None:
                (output_path.parent / "transfer.json").write_bytes(
                    canonical_json_bytes(
                        {
                            "checkpoint_digest": transfer_manifest.digest,
                            "checkpoint_observed_timestep": (
                                transfer_manifest.observed_timestep
                            ),
                            "requested_additional_timesteps": config.timesteps,
                            "schema_version": "training_transfer_v1",
                            "source_environment_digest": (
                                transfer_manifest.environment_digest
                            ),
                            "target_environment_digest": str(
                                identity["environment_digest"]
                            ),
                            "target_total_timesteps": target_total_timesteps,
                        }
                    )
                )
            save_target = output_path.with_suffix("")
            from trade_rl.rl.checkpointing import save_policy_without_runtime_state

            save_policy_without_runtime_state(model, str(save_target))
            created = save_target.with_suffix(".zip")
            if created != output_path:
                created.replace(output_path)

            structured_manifest_path: Path | None = None
            structured_manifest_digest: str | None = None
            structured_model_path: Path | None = None
            structured_model_digest: str | None = None
            architecture_digest: str | None = None
            if config.observation_encoder == "hierarchical_sequence_v2":
                from trade_rl.rl.policy_identity import model_sb3_policy_identity

                policy_identity = model_sb3_policy_identity(model)
                raw_architecture_digest = (
                    None
                    if policy_identity is None
                    else policy_identity.get("policy_architecture_digest")
                )
                if not isinstance(raw_architecture_digest, str):
                    raise RuntimeError(
                        "hierarchical model architecture identity is unavailable"
                    )
                architecture_digest = raw_architecture_digest
            if self.structured_export_enabled:
                from trade_rl.rl.structured_export import (
                    STRUCTURED_EXPORT_MANIFEST_NAME,
                    export_structured_policy_actor,
                )

                example_observation = _reset_observation_for_export(
                    environment,
                    seed=seed,
                )
                structured_manifest = export_structured_policy_actor(
                    model=model,
                    output_dir=output_path.parent,
                    example_observation=example_observation,
                    action_size=int(identity["action_size"]),
                    tolerance=self.structured_export_tolerance,
                )
                structured_manifest_path = (
                    output_path.parent / STRUCTURED_EXPORT_MANIFEST_NAME
                )
                structured_manifest_digest = structured_manifest.digest
                structured_model_path = (
                    output_path.parent / structured_manifest.model_path
                )
                structured_model_digest = structured_manifest.model_digest
                if structured_manifest.architecture_digest != architecture_digest:
                    raise RuntimeError(
                        "structured export architecture differs from the bound model"
                    )

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
                structured_export_manifest_path=structured_manifest_path,
                structured_export_manifest_digest=structured_manifest_digest,
                structured_export_model_path=structured_model_path,
                structured_export_model_digest=structured_model_digest,
                architecture_digest=architecture_digest,
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
