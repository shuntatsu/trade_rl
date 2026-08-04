"""Stable-Baselines3 training adapter isolated from the RL core contracts."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.contracts import ArtifactKind
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.integrations.sb3_checkpoint_assembly import (
    load_sb3_checkpoint_model,
    load_sb3_checkpoint_transfer_model,
)
from trade_rl.integrations.sb3_model_assembly import (
    build_sb3_model,
    resolve_sb3_policy_assembly,
)
from trade_rl.learning import (
    BehaviorCloningConfig,
    BehaviorCloningGateEvaluation,
    BehaviorCloningGateThresholds,
    BehaviorCloningHoldoutEvaluation,
    OracleTeacherConfig,
    StructuredTeacherObservationProvider,
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    evaluate_behavior_cloning_gates,
    load_teacher_artifact,
    oracle_target_path,
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
    oracle_episode_sampling_config,
    resolve_episode_initial_weights,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeSamplingConfig,
    build_episode_oracle_batch,
)
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    collect_episode_teacher_rollout_parallel,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    build_hierarchical_teacher_labels,
)
from trade_rl.learning.oracle_bellman_contracts import (
    CompileMode,
    OracleSolverConfig,
    SolverSelection,
)
from trade_rl.learning.teacher_cache import (
    teacher_cache_identity,
    teacher_cache_identity_v2,
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
from trade_rl.rl.tensorboard_logging import (
    build_tensorboard_metrics_callback,
)
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    _environment_identity,
    _validate_training_environment,
)
from trade_rl.rl.training_modes import CudaRuntimeMode
from trade_rl.rl.training_performance import (
    TrainingPerformanceRecorder,
    activate_training_performance,
    write_training_performance_evidence,
)


def _lagrangian_probe_worker_count(n_envs: int) -> int:
    raw = os.environ.get("TRADE_RL_LAGRANGIAN_PROBE_WORKERS", "1").strip()
    try:
        configured = int(raw)
    except ValueError as error:
        raise ValueError(
            "TRADE_RL_LAGRANGIAN_PROBE_WORKERS must be an integer"
        ) from error
    if configured <= 0:
        raise ValueError("TRADE_RL_LAGRANGIAN_PROBE_WORKERS must be positive")
    return min(n_envs, configured)


def _oracle_solver_config() -> OracleSolverConfig:
    """Parse the explicit Oracle backend/resource contract before generation."""

    selection_name = "TRADE_RL_ORACLE_SOLVER"
    raw_selection = os.environ.get(selection_name, "numpy").strip()
    if raw_selection not in {"numpy", "cuda", "cuda_or_numpy"}:
        raise ValueError(
            f"{selection_name} must be one of numpy, cuda, or cuda_or_numpy"
        )

    def positive_integer(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    batch_size = positive_integer("TRADE_RL_ORACLE_EPISODE_BATCH_SIZE", 8)
    block_name = "TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE"
    raw_block = os.environ.get(block_name, "").strip()
    block_size: int | None = None
    if raw_block:
        try:
            block_size = int(raw_block)
        except ValueError as error:
            raise ValueError(f"{block_name} must be an integer when set") from error
        if block_size <= 0:
            raise ValueError(f"{block_name} must be positive when set")

    memory_name = "TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION"
    raw_memory = os.environ.get(memory_name, "0.65").strip()
    try:
        memory_fraction = float(raw_memory)
    except ValueError as error:
        raise ValueError(f"{memory_name} must be numeric") from error
    if not np.isfinite(memory_fraction) or not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"{memory_name} must be within (0, 1]")

    compile_name = "TRADE_RL_ORACLE_COMPILE_MODE"
    raw_compile = os.environ.get(compile_name, "disabled").strip()
    if raw_compile not in {"disabled", "reduce_overhead"}:
        raise ValueError(f"{compile_name} must be disabled or reduce_overhead")
    chunk_name = "TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE"
    compile_chunk_size = positive_integer(chunk_name, 16)
    if compile_chunk_size not in {8, 16, 32, 64}:
        raise ValueError(f"{chunk_name} must be one of 8, 16, 32, or 64")

    return OracleSolverConfig(
        selection=cast(SolverSelection, raw_selection),
        episode_batch_size=batch_size,
        target_state_block_size=block_size,
        cuda_memory_fraction=memory_fraction,
        compile_mode=cast(CompileMode, raw_compile),
        compile_chunk_size=compile_chunk_size,
    )


def _teacher_worker_count(
    n_envs: int,
    *,
    solver_config: OracleSolverConfig | None = None,
) -> int:
    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "").strip()
    try:
        if raw:
            configured = int(raw)
        else:
            configured = n_envs if solver_config is None else 1
    except ValueError as error:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be an integer") from error
    if configured <= 0:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be positive")
    if solver_config is not None and solver_config.selection != "numpy":
        if configured != 1:
            raise ValueError("CUDA Oracle solving requires TRADE_RL_TEACHER_WORKERS=1")
    return min(n_envs, configured)


def _oracle_episode_sampling_config(
    environment: Any,
    *,
    train_range: tuple[int, int],
    seed: int,
) -> OracleEpisodeSamplingConfig:
    return oracle_episode_sampling_config(
        environment,
        train_range=train_range,
        seed=seed,
    )


def _configure_torch_cuda_runtime(
    torch: Any,
    device: object,
    mode: CudaRuntimeMode | str,
) -> dict[str, object]:
    """Apply one explicit CUDA speed/reproducibility contract."""

    resolved_mode = CudaRuntimeMode(mode)
    requested = str(device).strip().lower()
    uses_cuda = requested == "auto" and bool(torch.cuda.is_available())
    if requested != "auto":
        try:
            uses_cuda = torch.device(device).type == "cuda"
        except (RuntimeError, TypeError, ValueError):
            uses_cuda = False

    deterministic = resolved_mode is CudaRuntimeMode.DETERMINISTIC
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if uses_cuda and deterministic:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif uses_cuda:
        # Performance mode intentionally permits nondeterministic kernel selection.
        # Parameters, optimizer state, losses, and checkpoints remain float32.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    bf16_supported = bool(
        uses_cuda
        and callable(getattr(torch.cuda, "is_bf16_supported", None))
        and torch.cuda.is_bf16_supported()
    )
    return {
        "mode": str(resolved_mode),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sequence_encoder_autocast": ("bfloat16" if bf16_supported else "disabled"),
    }


def _configure_sequence_runtime(
    torch: Any,
    model: Any,
    config: ResidualTrainingConfig,
) -> dict[str, object]:
    # Apply the identity-bound sequence runtime after construction or load.

    compile_enabled = bool(config.sequence_compile)
    compile_target: str | None = None
    if compile_enabled:
        resolved_device = torch.device(model.device)
        if resolved_device.type != "cuda":
            raise RuntimeError("sequence_compile requires a resolved CUDA device")
        extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
        compile_module = getattr(extractor, "compile", None)
        if not callable(compile_module):
            raise RuntimeError(
                "sequence feature extractor does not support in-place compile"
            )
        compile_module(
            mode=config.sequence_compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        compile_target = type(extractor).__name__
    return {
        "compile_enabled": compile_enabled,
        "compile_mode": config.sequence_compile_mode,
        "compile_target": compile_target,
        "fullgraph": False,
        "dynamic": False,
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "sequence_transfer_mode": config.sequence_transfer_mode,
        "torch_version": str(torch.__version__),
        "schema_version": "sequence_runtime_v2",
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


def _required_hierarchical_config(config: object, name: str) -> int | float:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(
            f"hierarchical BC requires explicit training_run_config_v4 field {name}"
        )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"hierarchical BC training_run_config_v4 field {name} must be numeric"
        )
    if not np.isfinite(float(value)):
        raise ValueError(
            f"hierarchical BC training_run_config_v4 field {name} must be finite"
        )
    return value


def _teacher_change_labels(
    *,
    teacher_dataset: SupervisedPolicyDataset,
    config: object,
) -> HierarchicalTeacherLabels | None:
    observations = teacher_dataset.observations
    if not isinstance(observations, Mapping):
        return None
    missing = {"active", "current_weights"} - set(observations)
    if missing:
        raise ValueError(
            "structured BC teacher observations are missing "
            + ", ".join(sorted(missing))
        )
    change_threshold = _required_hierarchical_config(
        config, "behavior_cloning_gate_change_threshold"
    )
    return build_hierarchical_teacher_labels(
        teacher_targets=np.asarray(teacher_dataset.actions),
        current_weights=np.asarray(observations["current_weights"]),
        active_mask=np.asarray(observations["active"]) > 0.5,
        change_threshold=float(change_threshold),
        source_teacher_digest=teacher_dataset.action_digest,
    )


def _uses_hierarchical_actor_head(policy: object) -> bool:
    actor_head = getattr(policy, "shared_actor_head", None)
    return actor_head in {None, "hierarchical_gate_target_v1"} and callable(
        getattr(policy, "hierarchical_actor_outputs", None)
    )


def _hierarchical_teacher_labels(
    *,
    policy: object,
    teacher_dataset: SupervisedPolicyDataset,
    config: object,
) -> HierarchicalTeacherLabels | None:
    if not _uses_hierarchical_actor_head(policy):
        return None
    labels = _teacher_change_labels(
        teacher_dataset=teacher_dataset,
        config=config,
    )
    if labels is None:
        raise ValueError("hierarchical BC requires structured teacher observations")
    return labels


def _hierarchical_behavior_cloning_config(
    config: ResidualTrainingConfig,
) -> BehaviorCloningConfig:
    return BehaviorCloningConfig(
        epochs=config.behavior_cloning_epochs,
        learning_rate=config.behavior_cloning_learning_rate,
        batch_size=config.behavior_cloning_batch_size,
        validation_fraction=config.behavior_cloning_validation_fraction,
        early_stopping_patience=config.behavior_cloning_patience,
        minimum_improvement=config.behavior_cloning_minimum_improvement,
        gate_loss_weight=float(
            _required_hierarchical_config(config, "behavior_cloning_gate_loss_weight")
        ),
        target_loss_weight=float(
            _required_hierarchical_config(config, "behavior_cloning_target_loss_weight")
        ),
        composed_loss_weight=float(
            _required_hierarchical_config(
                config, "behavior_cloning_composed_loss_weight"
            )
        ),
        max_positive_class_weight=float(
            _required_hierarchical_config(
                config, "behavior_cloning_max_positive_class_weight"
            )
        ),
        gate_prediction_threshold=float(
            _required_hierarchical_config(
                config, "behavior_cloning_gate_prediction_threshold"
            )
        ),
    )


def _behavior_cloning_gate_thresholds(
    config: ResidualTrainingConfig,
) -> BehaviorCloningGateThresholds:
    return BehaviorCloningGateThresholds(
        minimum_composed_loss_relative_improvement=(
            config.behavior_cloning_required_relative_improvement
        ),
        minimum_gate_precision=float(
            _required_hierarchical_config(config, "behavior_cloning_min_gate_precision")
        ),
        minimum_gate_recall=float(
            _required_hierarchical_config(config, "behavior_cloning_min_gate_recall")
        ),
        maximum_active_target_rmse=float(
            _required_hierarchical_config(
                config, "behavior_cloning_max_active_target_rmse"
            )
        ),
        minimum_activity_ratio=float(
            _required_hierarchical_config(config, "behavior_cloning_min_activity_ratio")
        ),
        maximum_activity_ratio=float(
            _required_hierarchical_config(config, "behavior_cloning_max_activity_ratio")
        ),
        minimum_teacher_positive_support=1,
        minimum_causal_holdout_trades=int(
            _required_hierarchical_config(
                config, "behavior_cloning_min_causal_holdout_trades"
            )
        ),
        maximum_causal_holdout_regret=float(
            _required_hierarchical_config(
                config, "behavior_cloning_max_causal_holdout_regret"
            )
        ),
        minimum_causal_holdout_episodes=1,
        maximum_causal_holdout_regret_upper_bound=float(
            _required_hierarchical_config(
                config, "behavior_cloning_max_causal_holdout_regret"
            )
        ),
    )


def _evaluate_hierarchical_behavior_cloning_gate(
    *,
    cloning: object,
    holdout: Any,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
    initial_losses = getattr(cloning, "initial_hierarchical_losses", None)
    final_losses = getattr(cloning, "final_hierarchical_losses", None)
    validation_metrics = getattr(cloning, "validation_hierarchical_metrics", None)
    if validation_metrics is None:
        validation_metrics = getattr(cloning, "final_hierarchical_metrics", None)
    return evaluate_behavior_cloning_gates(
        initial_composed_loss=(
            None if initial_losses is None else float(initial_losses.composed)
        ),
        final_composed_loss=(
            None if final_losses is None else float(final_losses.composed)
        ),
        reconstruction_metrics=validation_metrics,
        holdout=holdout,
        thresholds=thresholds,
    )


def _enforce_behavior_cloning_gates(
    evaluation: BehaviorCloningGateEvaluation,
) -> None:
    evaluation.require_passed()


def _build_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [factory for _ in range(n_envs)]
    if subprocesses:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def _effective_vector_environment_kind(config: ResidualTrainingConfig) -> str:
    if config.n_envs == 1:
        return "direct"
    if config.vector_environment_mode != "subprocess":
        return "in_process"
    if config.observation_encoder == "hierarchical_sequence_v2":
        return "subprocess_compact_sequence"
    return "subprocess"


def _compact_filtered_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
) -> gym.Env[Any, Any]:
    from trade_rl.rl.sequence_observations import (
        sequence_policy_plane_materialization,
    )

    with sequence_policy_plane_materialization(False):
        environment = factory()
    unwrapped: Any = getattr(environment, "unwrapped", environment)
    setter = getattr(unwrapped, "set_compact_sequence_training_observations", None)
    if not callable(setter):
        environment.close()
        raise TypeError(
            "parallel sequence worker does not support compact observations"
        )
    setter(True)
    return _TrainingInfoFilter(environment)


def _build_parallel_sequence_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    full_observation_space: gym.spaces.Dict,
    reconstructor: Any,
) -> Any:
    from trade_rl.integrations.parallel_sequence_env import ParallelSequenceVecEnv

    workers = _build_training_environment(
        partial(_compact_filtered_training_environment, factory),
        n_envs,
        subprocesses=True,
    )
    try:
        return ParallelSequenceVecEnv(
            workers,
            full_observation_space=full_observation_space,
            reconstructor=reconstructor,
        )
    except BaseException:
        workers.close()
        raise


def _reset_observation_for_export(
    environment: object, *, seed: int
) -> Mapping[str, np.ndarray]:
    reset = getattr(environment, "reset", None)
    if not callable(reset):
        raise TypeError("structured export environment does not support reset")
    try:
        raw = reset(seed=seed)
    except TypeError:
        raw = reset()
    observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
    if not isinstance(observation, Mapping):
        raise ValueError("structured export requires a mapping observation")
    return {key: np.asarray(value) for key, value in observation.items()}


class StableBaselines3Backend:
    """Train one policy with an optional SB3-family algorithm."""

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
        resume_checkpoint_artifacts: Mapping[int, Path] | None = None,
        transfer_checkpoint_artifacts: Mapping[int, Path] | None = None,
        structured_export_enabled: bool = False,
        structured_export_tolerance: float = 1e-5,
    ) -> None:
        self.environment_factory = environment_factory
        self.verbose = verbose
        self.resume_replay_artifact = resume_replay_artifact
        self.resume_checkpoint_artifacts = dict(resume_checkpoint_artifacts or {})
        self.transfer_checkpoint_artifacts = dict(transfer_checkpoint_artifacts or {})
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

    def _oracle_episode_batch(
        self,
        environment: Any,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
        sampling_config: OracleEpisodeSamplingConfig,
        max_workers: int = 1,
        solver_config: OracleSolverConfig | None = None,
    ) -> EpisodeOracleBatch:
        resolved_solver_config = solver_config or OracleSolverConfig()
        dataset = environment.dataset
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("Oracle episode dataset must expose dataset_id")
        start, stop = train_range
        key = (
            dataset_id,
            int(start),
            int(stop),
            teacher_config.digest,
            sampling_config.digest,
            resolved_solver_config.digest,
        )
        cached = self._oracle_episode_batch_cache.get(key)
        if cached is not None:
            return cached
        batch = build_episode_oracle_batch(
            dataset,
            minimum_start_index=start,
            sampling_config=sampling_config,
            teacher_config=teacher_config,
            initial_weight_provider=lambda mode, index: resolve_episode_initial_weights(
                environment,
                mode,
                index,
            ),
            max_workers=max_workers,
            solver_config=resolved_solver_config,
        )
        self._oracle_episode_batch_cache[key] = batch
        return batch

    def _episode_teacher_dataset(
        self,
        environment: Any,
        batch: EpisodeOracleBatch,
        *,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
        max_workers: int = 1,
    ) -> EpisodeSupervisedPolicyDataset:
        start, stop = train_range
        environment_digest = getattr(environment, "environment_digest", None)
        action_spec_digest = getattr(environment, "action_spec_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError(
                "episode teacher environment must expose environment_digest"
            )
        if not isinstance(action_spec_digest, str):
            raise ValueError(
                "episode teacher environment must expose action_spec_digest"
            )
        artifact_schema = (
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1
            if batch.solver_provenance is None
            else EPISODE_TEACHER_ARTIFACT_SCHEMA
        )
        teacher_identity = content_digest(
            {
                "episode_batch_digest": batch.digest,
                "schema_version": artifact_schema,
                "teacher_config_digest": teacher_config.digest,
            }
        )
        key = (
            batch.dataset_id,
            int(start),
            int(stop),
            environment_digest,
            action_spec_digest,
            teacher_identity,
        )
        cached = self._episode_teacher_dataset_cache.get(key)
        if cached is not None:
            return cached
        cache_path: Path | None = None
        shard_root: Path | None = None
        if self.teacher_cache_root is not None:
            cache_identity = (
                teacher_cache_identity(
                    dataset_id=batch.dataset_id,
                    train_range=(start, stop),
                    environment_digest=environment_digest,
                    action_spec_digest=action_spec_digest,
                    teacher_config_digest=teacher_identity,
                )
                if batch.solver_provenance is None
                else teacher_cache_identity_v2(
                    dataset_id=batch.dataset_id,
                    train_range=(start, stop),
                    environment_digest=environment_digest,
                    action_spec_digest=action_spec_digest,
                    teacher_config_digest=teacher_identity,
                    solver_provenance=batch.solver_provenance,
                )
            )
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=batch.dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_identity,
            )
            if self.reusable_artifact_index is not None:
                indexed = self.reusable_artifact_index.resolve(
                    ArtifactKind.ORACLE_TEACHER,
                    cache_identity,
                )
                if indexed is not None:
                    cache_path = indexed
            shard_root = cache_path.with_name(f".{cache_path.name}.episodes")
            if cache_path.exists():
                manifest, teacher_dataset = load_episode_teacher_artifact(
                    cache_path,
                    expected_dataset_id=batch.dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                )
                if teacher_dataset.teacher_config_digest != teacher_identity:
                    raise ValueError("cached episode teacher identity mismatch")
                if self.reusable_artifact_index is not None:
                    self.reusable_artifact_index.register_directory(
                        artifact_digest=manifest.artifact_digest,
                        artifact_kind=ArtifactKind.ORACLE_TEACHER,
                        schema_version=manifest.schema_version,
                        dataset_id=batch.dataset_id,
                        cache_key=cache_identity,
                        metadata={
                            "episode_count": manifest.episode_count,
                            "sample_count": manifest.sample_count,
                            **(
                                {}
                                if manifest.solver_provenance is None
                                else {
                                    "solver_provenance": manifest.solver_provenance.serialized_payload()
                                }
                            ),
                        },
                        location=cache_path,
                    )
                self._episode_teacher_dataset_cache[key] = teacher_dataset
                return teacher_dataset
        if max_workers == 1:
            teacher_dataset = collect_episode_teacher_rollout(
                environment,
                batch,
                teacher_config_digest=teacher_identity,
            )
        else:
            teacher_dataset = collect_episode_teacher_rollout_parallel(
                self.environment_factory,
                batch,
                teacher_config_digest=teacher_identity,
                max_workers=max_workers,
                shard_root=shard_root,
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_path.name}.", dir=str(cache_path.parent)
                )
            )
            try:
                write_episode_teacher_artifact(temporary, teacher_dataset)
                try:
                    temporary.replace(cache_path)
                except FileExistsError:
                    pass
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            if shard_root is not None and shard_root.exists():
                shutil.rmtree(shard_root)
            if self.reusable_artifact_index is not None:
                manifest, _ = load_episode_teacher_artifact(
                    cache_path,
                    expected_dataset_id=batch.dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                )
                self.reusable_artifact_index.register_directory(
                    artifact_digest=manifest.artifact_digest,
                    artifact_kind=ArtifactKind.ORACLE_TEACHER,
                    schema_version=manifest.schema_version,
                    dataset_id=batch.dataset_id,
                    cache_key=cache_identity,
                    metadata={
                        "episode_count": manifest.episode_count,
                        "sample_count": manifest.sample_count,
                    },
                    location=cache_path,
                )
        self._episode_teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset

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
            cache_identity = teacher_cache_identity(
                dataset_id=dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config.digest,
            )
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config.digest,
            )
            if self.reusable_artifact_index is not None:
                indexed = self.reusable_artifact_index.resolve(
                    ArtifactKind.ORACLE_TEACHER,
                    cache_identity,
                )
                if indexed is not None:
                    cache_path = indexed
            if cache_path.exists():
                manifest, teacher_dataset = load_teacher_artifact(
                    cache_path,
                    expected_dataset_id=dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                    expected_train_range=(start, stop),
                )
                if teacher_dataset.teacher_config_digest != teacher_config.digest:
                    raise ValueError("cached teacher configuration identity mismatch")
                if self.reusable_artifact_index is not None:
                    self.reusable_artifact_index.register_directory(
                        artifact_digest=manifest.artifact_digest,
                        artifact_kind=ArtifactKind.ORACLE_TEACHER,
                        schema_version=manifest.schema_version,
                        dataset_id=dataset_id,
                        cache_key=cache_identity,
                        metadata={"sample_count": manifest.sample_count},
                        location=cache_path,
                    )
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
            if self.reusable_artifact_index is not None:
                manifest, _ = load_teacher_artifact(
                    cache_path,
                    expected_dataset_id=dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                    expected_train_range=(start, stop),
                )
                self.reusable_artifact_index.register_directory(
                    artifact_digest=manifest.artifact_digest,
                    artifact_kind=ArtifactKind.ORACLE_TEACHER,
                    schema_version=manifest.schema_version,
                    dataset_id=dataset_id,
                    cache_key=cache_identity,
                    metadata={"sample_count": manifest.sample_count},
                    location=cache_path,
                )
        self._teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset

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
            # A full-market environment is several GiB.  Do not keep the
            # identity probe alive while the isolated canonical probe creates
            # its own environment, otherwise the 15.5 GiB training cgroup can
            # contain two full environments at once and be OOM killed.
            probe = self.environment_factory()
            identity = _environment_identity(probe)
            _validate_training_environment(identity, config)
            resume_root = self.resume_checkpoint_artifacts.get(seed)
            transfer_root = self.transfer_checkpoint_artifacts.get(seed)
            fresh_behavior_cloning = (
                config.behavior_cloning_epochs > 0
                and resume_root is None
                and transfer_root is None
            )
            prefetched_episode_batch: EpisodeOracleBatch | None = None
            prefetched_episode_teacher: EpisodeSupervisedPolicyDataset | None = None
            prefetched_oracle_config: OracleTeacherConfig | None = None
            if fresh_behavior_cloning and config.behavior_cloning_teacher == "oracle":
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
                risk_config = unwrapped_probe.pre_trade_risk.config
                prefetched_oracle_config = OracleTeacherConfig(
                    execution_cost=unwrapped_probe.config.execution_cost,
                    portfolio_risk=unwrapped_probe.portfolio_risk.config,
                    max_gross=risk_config.max_gross,
                    max_abs_weight=risk_config.max_abs_weight,
                    entry_threshold=risk_config.entry_threshold,
                    exit_threshold=risk_config.exit_threshold,
                    no_trade_band=risk_config.no_trade_band,
                    reference_portfolio_value=unwrapped_probe.initial_capital,
                    signal_delay_decisions=(
                        unwrapped_probe.config.signal_delay_decisions
                    ),
                )
                sampling_config = _oracle_episode_sampling_config(
                    unwrapped_probe,
                    train_range=probe_train_range,
                    seed=seed,
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
            # The CPU-only canonical probe may fork on Linux.  Initialize the
            # CUDA runtime only after its workers and the forked Oracle teacher
            # workers have joined so no child inherits a live CUDA context.
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
            vector_environment_kind = _effective_vector_environment_kind(config)
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
                    lambda: _filtered_training_environment(self.environment_factory),
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
                seed=seed,
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

            # Stable-Baselines3 seeds CUDA during model construction/loading and
            # resets cuDNN to its deterministic, slow dilated-convolution path.
            # Capture and persist the effective post-construction runtime state.
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
                # The compact PPO vector environment keeps one full market view in
                # each subprocess.  It is idle during behavior cloning, while the
                # Oracle rollout creates its own bounded worker environments.  Do
                # not retain both groups: on the 15-symbol full-history dataset the
                # otherwise-idle PPO workers alone consume most of a 24 GiB cgroup.
                environment.close()
                environment = None
                environment_suspended_for_teacher = True
            if fresh_behavior_cloning:
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
                        seed=seed,
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
                    cloning_payload = {
                        "artifact_digest": teacher_digest,
                        "behavior_cloning_digest": cloning.digest,
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
                finally:
                    teacher_environment.close()
            if environment_suspended_for_teacher:
                environment = build_parallel_environment()
                model.set_env(environment)
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
                interval_steps=config.resolved_checkpoint_interval,
                max_checkpoints=config.max_checkpoints,
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
