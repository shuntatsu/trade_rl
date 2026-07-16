"""Residual-policy ensemble training orchestration and backend isolation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.policies import PolicyEnsembleManifest, PolicyMember
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA


def _require_inactive_default(
    field_name: str,
    value: object,
    default: object,
    *,
    context: str,
) -> None:
    if value != default:
        raise ValueError(
            f"{field_name} is inactive for {context}; leave it at its default value"
        )


def _require_inactive_defaults(
    fields: tuple[tuple[str, object, object], ...],
    *,
    context: str,
) -> None:
    for field_name, value, default in fields:
        _require_inactive_default(field_name, value, default, context=context)


def gamma_from_half_life(*, decision_hours: float, half_life_hours: float) -> float:
    """Convert a real-time discount half-life to a per-decision gamma."""

    if not math.isfinite(decision_hours) or decision_hours <= 0.0:
        raise ValueError("decision_hours must be finite and positive")
    if not math.isfinite(half_life_hours) or half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be finite and positive")
    gamma = math.exp(math.log(0.5) * decision_hours / half_life_hours)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("resolved gamma must be within (0, 1]")
    return gamma


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    timesteps: int
    gamma: float
    seeds: tuple[int, ...]
    learning_rate: float = 3e-4
    n_steps: int = 2_048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MlpPolicy"
    device: str = "auto"
    decision_hours: float | None = None
    discount_half_life_hours: float | None = None
    log_std_init: float = -0.5
    target_kl: float | None = 0.02
    use_sde: bool = False
    sde_sample_freq: int = -1
    policy_net_arch: tuple[int, ...] = (128, 128)
    value_net_arch: tuple[int, ...] = (128, 128)
    sequence_encoder: bool = False
    sequence_d_model: int = 320
    sequence_attention_heads: int = 8
    sequence_attention_layers: int = 2
    sequence_dropout: float = 0.05
    max_policy_parameters: int = 12_000_000
    max_rollout_buffer_bytes: int = 805_306_368
    asset_set_encoder: bool = True
    asset_embedding_dim: int = 64
    global_embedding_dim: int = 64
    algorithm: str = "ppo"
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    checkpoint_interval_steps: int | None = None
    max_checkpoints: int = 5
    n_envs: int = 1
    behavior_cloning_epochs: int = 0
    behavior_cloning_learning_rate: float = 1e-3
    behavior_cloning_batch_size: int = 256
    behavior_cloning_validation_fraction: float = 0.0
    behavior_cloning_patience: int = 3
    behavior_cloning_minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        for integer_field_name, integer_value in (
            ("timesteps", self.timesteps),
            ("n_steps", self.n_steps),
            ("n_envs", self.n_envs),
            ("batch_size", self.batch_size),
            ("n_epochs", self.n_epochs),
            ("buffer_size", self.buffer_size),
            ("train_freq", self.train_freq),
            ("gradient_steps", self.gradient_steps),
            ("behavior_cloning_batch_size", self.behavior_cloning_batch_size),
            ("behavior_cloning_patience", self.behavior_cloning_patience),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"{integer_field_name} must be a positive integer")
        if (
            isinstance(self.behavior_cloning_epochs, bool)
            or not isinstance(self.behavior_cloning_epochs, int)
            or self.behavior_cloning_epochs < 0
        ):
            raise ValueError("behavior_cloning_epochs must be non-negative")
        if (
            not math.isfinite(self.behavior_cloning_learning_rate)
            or self.behavior_cloning_learning_rate <= 0.0
        ):
            raise ValueError("behavior_cloning_learning_rate must be positive")
        if (
            not math.isfinite(self.behavior_cloning_validation_fraction)
            or not 0.0 <= self.behavior_cloning_validation_fraction < 0.5
        ):
            raise ValueError(
                "behavior_cloning_validation_fraction must be within [0, 0.5)"
            )
        if (
            not math.isfinite(self.behavior_cloning_minimum_improvement)
            or self.behavior_cloning_minimum_improvement < 0.0
        ):
            raise ValueError(
                "behavior_cloning_minimum_improvement must be non-negative"
            )
        if self.checkpoint_interval_steps is not None and (
            isinstance(self.checkpoint_interval_steps, bool)
            or not isinstance(self.checkpoint_interval_steps, int)
            or self.checkpoint_interval_steps < 0
        ):
            raise ValueError("checkpoint_interval_steps must be non-negative")
        if (
            isinstance(self.max_checkpoints, bool)
            or not isinstance(self.max_checkpoints, int)
            or self.max_checkpoints <= 0
        ):
            raise ValueError("max_checkpoints must be a positive integer")
        if (
            self.algorithm.lower() == "ppo"
            and (self.n_steps * self.n_envs) % self.batch_size != 0
        ):
            raise ValueError("batch_size must divide the complete PPO rollout")
        algorithm = self.algorithm.lower()
        if algorithm not in {"ppo", "sac", "td3", "tqc"}:
            raise ValueError("algorithm must be one of ppo, sac, td3, or tqc")
        object.__setattr__(self, "algorithm", algorithm)
        if self.behavior_cloning_epochs > 0 and algorithm != "ppo":
            raise ValueError("behavior cloning warm start currently requires PPO")
        if (
            isinstance(self.learning_starts, bool)
            or not isinstance(self.learning_starts, int)
            or self.learning_starts < 0
        ):
            raise ValueError("learning_starts must be a non-negative integer")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.gae_lambda) or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be within (0, 1]")
        if not math.isfinite(self.clip_range) or self.clip_range <= 0.0:
            raise ValueError("clip_range must be finite and positive")
        for coefficient_field_name, coefficient_value in (
            ("ent_coef", self.ent_coef),
            ("vf_coef", self.vf_coef),
        ):
            if not math.isfinite(coefficient_value) or coefficient_value < 0.0:
                raise ValueError(
                    f"{coefficient_field_name} must be finite and non-negative"
                )
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not isinstance(self.normalize_advantage, bool):
            raise ValueError("normalize_advantage must be a boolean")
        require_non_empty(self.policy, field="policy")
        require_non_empty(self.device, field="device")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.decision_hours is not None and (
            not math.isfinite(self.decision_hours) or self.decision_hours <= 0.0
        ):
            raise ValueError("decision_hours must be finite and positive")
        if self.discount_half_life_hours is not None and (
            not math.isfinite(self.discount_half_life_hours)
            or self.discount_half_life_hours <= 0.0
        ):
            raise ValueError("discount_half_life_hours must be finite and positive")
        if self.discount_half_life_hours is not None and self.decision_hours is None:
            raise ValueError("discount_half_life_hours requires decision_hours")
        if (
            self.decision_hours is not None
            and self.discount_half_life_hours is not None
        ):
            expected_gamma = gamma_from_half_life(
                decision_hours=self.decision_hours,
                half_life_hours=self.discount_half_life_hours,
            )
            if not math.isclose(self.gamma, expected_gamma, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "gamma does not match the configured real-time half-life"
                )
        if not math.isfinite(self.log_std_init):
            raise ValueError("log_std_init must be finite")
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0.0
        ):
            raise ValueError("target_kl must be finite and positive")
        if not isinstance(self.use_sde, bool):
            raise ValueError("use_sde must be a boolean")
        if (
            isinstance(self.sde_sample_freq, bool)
            or not isinstance(self.sde_sample_freq, int)
            or self.sde_sample_freq == 0
            or self.sde_sample_freq < -1
        ):
            raise ValueError("sde_sample_freq must be -1 or a positive integer")
        for field_name, architecture in (
            ("policy_net_arch", self.policy_net_arch),
            ("value_net_arch", self.value_net_arch),
        ):
            if not architecture or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in architecture
            ):
                raise ValueError(f"{field_name} must contain positive integers")
        if not isinstance(self.sequence_encoder, bool):
            raise ValueError("sequence_encoder must be a boolean")
        if self.sequence_encoder and self.policy != "MultiInputPolicy":
            raise ValueError("sequence_encoder requires MultiInputPolicy")
        if self.sequence_encoder and self.algorithm != "ppo":
            raise ValueError("sequence_encoder currently requires PPO")
        for field_name, value in (
            ("sequence_d_model", self.sequence_d_model),
            ("sequence_attention_heads", self.sequence_attention_heads),
            ("sequence_attention_layers", self.sequence_attention_layers),
            ("max_policy_parameters", self.max_policy_parameters),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.sequence_d_model % self.sequence_attention_heads != 0:
            raise ValueError(
                "sequence_d_model must divide evenly across attention heads"
            )
        if (
            not math.isfinite(self.sequence_dropout)
            or not 0.0 <= self.sequence_dropout <= 0.05
        ):
            raise ValueError("sequence_dropout must be within [0, 0.05]")
        if self.sequence_encoder and self.asset_set_encoder:
            raise ValueError(
                "sequence_encoder and asset_set_encoder are mutually exclusive"
            )
        if (
            isinstance(self.max_rollout_buffer_bytes, bool)
            or not isinstance(self.max_rollout_buffer_bytes, int)
            or self.max_rollout_buffer_bytes <= 0
        ):
            raise ValueError("max_rollout_buffer_bytes must be a positive integer")
        if not isinstance(self.asset_set_encoder, bool):
            raise ValueError("asset_set_encoder must be a boolean")
        for field_name, value in (
            ("asset_embedding_dim", self.asset_embedding_dim),
            ("global_embedding_dim", self.global_embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if algorithm == "ppo":
            _require_inactive_defaults(
                (
                    ("buffer_size", self.buffer_size, 100_000),
                    ("learning_starts", self.learning_starts, 10_000),
                    ("train_freq", self.train_freq, 1),
                    ("gradient_steps", self.gradient_steps, 1),
                ),
                context="PPO",
            )
        else:
            _require_inactive_defaults(
                (
                    ("n_steps", self.n_steps, 2_048),
                    ("n_epochs", self.n_epochs, 10),
                    ("gae_lambda", self.gae_lambda, 0.95),
                    ("clip_range", self.clip_range, 0.2),
                    ("normalize_advantage", self.normalize_advantage, True),
                    ("ent_coef", self.ent_coef, 0.0),
                    ("vf_coef", self.vf_coef, 0.5),
                    ("max_grad_norm", self.max_grad_norm, 0.5),
                    ("log_std_init", self.log_std_init, -0.5),
                    ("target_kl", self.target_kl, 0.02),
                    (
                        "max_rollout_buffer_bytes",
                        self.max_rollout_buffer_bytes,
                        805_306_368,
                    ),
                ),
                context=algorithm.upper(),
            )

        if algorithm == "td3":
            if self.use_sde or self.sde_sample_freq != -1:
                raise ValueError("TD3 does not support SDE settings")

        if not self.sequence_encoder:
            _require_inactive_defaults(
                (
                    ("sequence_d_model", self.sequence_d_model, 320),
                    ("sequence_attention_heads", self.sequence_attention_heads, 8),
                    ("sequence_attention_layers", self.sequence_attention_layers, 2),
                    ("sequence_dropout", self.sequence_dropout, 0.05),
                ),
                context="sequence_encoder=False",
            )

        if not self.asset_set_encoder:
            _require_inactive_defaults(
                (
                    ("asset_embedding_dim", self.asset_embedding_dim, 64),
                    ("global_embedding_dim", self.global_embedding_dim, 64),
                ),
                context="asset_set_encoder=False",
            )

        if self.behavior_cloning_epochs == 0:
            _require_inactive_defaults(
                (
                    (
                        "behavior_cloning_learning_rate",
                        self.behavior_cloning_learning_rate,
                        1e-3,
                    ),
                    (
                        "behavior_cloning_batch_size",
                        self.behavior_cloning_batch_size,
                        256,
                    ),
                    (
                        "behavior_cloning_validation_fraction",
                        self.behavior_cloning_validation_fraction,
                        0.0,
                    ),
                    ("behavior_cloning_patience", self.behavior_cloning_patience, 3),
                    (
                        "behavior_cloning_minimum_improvement",
                        self.behavior_cloning_minimum_improvement,
                        0.0,
                    ),
                ),
                context="behavior cloning disabled",
            )

    @property
    def rounded_timesteps(self) -> int:
        if self.algorithm == "ppo":
            rollout_size = self.n_steps * self.n_envs
            return math.ceil(self.timesteps / rollout_size) * rollout_size
        return self.timesteps

    @property
    def resolved_checkpoint_interval(self) -> int:
        if self.checkpoint_interval_steps is not None:
            return self.checkpoint_interval_steps
        return max(1, math.ceil(self.timesteps / self.max_checkpoints))

    def digest_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "asset_embedding_dim": self.asset_embedding_dim,
            "asset_set_encoder": self.asset_set_encoder,
            "batch_size": self.batch_size,
            "behavior_cloning_batch_size": self.behavior_cloning_batch_size,
            "behavior_cloning_epochs": self.behavior_cloning_epochs,
            "behavior_cloning_learning_rate": self.behavior_cloning_learning_rate,
            "behavior_cloning_validation_fraction": self.behavior_cloning_validation_fraction,
            "behavior_cloning_patience": self.behavior_cloning_patience,
            "behavior_cloning_minimum_improvement": self.behavior_cloning_minimum_improvement,
            "buffer_size": self.buffer_size,
            "global_embedding_dim": self.global_embedding_dim,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "clip_range": self.clip_range,
            "decision_hours": self.decision_hours,
            "device": self.device,
            "discount_half_life_hours": self.discount_half_life_hours,
            "ent_coef": self.ent_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "gradient_steps": self.gradient_steps,
            "learning_rate": self.learning_rate,
            "learning_starts": self.learning_starts,
            "log_std_init": self.log_std_init,
            "max_checkpoints": self.max_checkpoints,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "n_envs": self.n_envs,
            "n_steps": self.n_steps,
            "normalize_advantage": self.normalize_advantage,
            "policy": self.policy,
            "policy_net_arch": self.policy_net_arch,
            "value_net_arch": self.value_net_arch,
            "sequence_encoder": self.sequence_encoder,
            "sequence_d_model": self.sequence_d_model,
            "sequence_attention_heads": self.sequence_attention_heads,
            "sequence_attention_layers": self.sequence_attention_layers,
            "sequence_dropout": self.sequence_dropout,
            "max_policy_parameters": self.max_policy_parameters,
            "max_rollout_buffer_bytes": self.max_rollout_buffer_bytes,
            "sde_sample_freq": self.sde_sample_freq,
            "seeds": self.seeds,
            "target_kl": self.target_kl,
            "timesteps": self.timesteps,
            "train_freq": self.train_freq,
            "use_sde": self.use_sde,
            "vf_coef": self.vf_coef,
        }


@dataclass(frozen=True, slots=True)
class PolicyTrainingResult:
    """One backend run with complete environment and model-shape identity."""

    checkpoint_path: Path
    actual_timesteps: int
    resolved_device: str
    environment_digest: str
    initial_capital: float
    action_size: int = 2
    action_names: tuple[str, ...] = ()
    action_spec_digest: str | None = None
    observation_size: int | None = None
    observation_schema: str = OBSERVATION_SCHEMA
    observation_contract_digest: str | None = None
    parameter_count: int | None = None
    rollout_buffer_bytes: int | None = None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    normalizer_digest: str | None = None
    replay_buffer_path: Path | None = None
    replay_buffer_digest: str | None = None

    def __post_init__(self) -> None:
        if self.actual_timesteps <= 0:
            raise ValueError("actual_timesteps must be positive")
        require_non_empty(self.resolved_device, field="resolved_device")
        require_sha256(self.environment_digest, field="environment_digest")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, int)
            or self.action_size <= 0
        ):
            raise ValueError("action_size must be a positive integer")
        if len(self.action_names) != self.action_size:
            raise ValueError("action_names must match action_size")
        if len(set(self.action_names)) != len(self.action_names) or any(
            not name for name in self.action_names
        ):
            raise ValueError("action_names must be unique and non-empty")
        if self.action_spec_digest is None:
            raise ValueError("action_spec_digest is required")
        require_sha256(self.action_spec_digest, field="action_spec_digest")
        require_non_empty(self.observation_schema, field="observation_schema")
        if self.observation_contract_digest is not None:
            require_sha256(
                self.observation_contract_digest, field="observation_contract_digest"
            )
        if self.parameter_count is not None and (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count <= 0
        ):
            raise ValueError("parameter_count must be a positive integer")
        if self.rollout_buffer_bytes is not None and (
            isinstance(self.rollout_buffer_bytes, bool)
            or not isinstance(self.rollout_buffer_bytes, int)
            or self.rollout_buffer_bytes <= 0
        ):
            raise ValueError("rollout_buffer_bytes must be a positive integer")
        if self.observation_size is not None and (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, int)
            or self.observation_size <= 0
        ):
            raise ValueError("observation_size must be a positive integer")
        for field_name, value in (
            ("alpha_artifact_digest", self.alpha_artifact_digest),
            ("factor_artifact_digest", self.factor_artifact_digest),
            ("normalizer_digest", self.normalizer_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)
        if (
            self.replay_buffer_path is not None
            and self.replay_buffer_path.suffix != ".pkl"
        ):
            raise ValueError("replay_buffer_path must use a .pkl suffix")
        if self.replay_buffer_digest is not None:
            require_sha256(self.replay_buffer_digest, field="replay_buffer_digest")
        if (self.replay_buffer_path is None) != (self.replay_buffer_digest is None):
            raise ValueError("replay buffer path and digest must be provided together")


class PolicyTrainingBackend(Protocol):
    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult: ...


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"training backend did not create checkpoint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_normalizer_digest(unwrapped: Any) -> str | None:
    flat = getattr(getattr(unwrapped, "normalizer", None), "digest", None)
    sequence = getattr(getattr(unwrapped, "sequence_normalizer", None), "digest", None)
    if flat is None and sequence is None:
        return None
    if sequence is None:
        return str(flat)
    if flat is None:
        return str(sequence)
    require_sha256(str(flat), field="normalizer_digest")
    require_sha256(str(sequence), field="sequence_normalizer_digest")
    return content_digest(
        {
            "flat": flat,
            "schema_version": "policy_normalizer_bundle_v1",
            "sequence": sequence,
        }
    )


def _environment_identity(environment: Any) -> dict[str, Any]:
    unwrapped: Any = getattr(environment, "unwrapped", environment)
    environment_digest = getattr(unwrapped, "environment_digest", None)
    initial_capital = getattr(unwrapped, "initial_capital", None)
    if not isinstance(environment_digest, str):
        raise ValueError("training environment must expose environment_digest")
    require_sha256(environment_digest, field="environment_digest")
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, int | float)
        or not math.isfinite(float(initial_capital))
        or float(initial_capital) <= 0.0
    ):
        raise ValueError("training environment must expose positive initial_capital")
    action_space = getattr(environment, "action_space", None)
    observation_space = getattr(environment, "observation_space", None)
    action_shape = getattr(action_space, "shape", None)
    observation_shape = getattr(observation_space, "shape", None)
    if not action_shape or len(action_shape) != 1 or action_shape[0] <= 0:
        raise ValueError("training environment must expose a flat action space")
    if observation_shape and len(observation_shape) == 1 and observation_shape[0] > 0:
        observation_size = int(observation_shape[0])
    else:
        component_spaces = getattr(observation_space, "spaces", None)
        if not isinstance(component_spaces, dict) or not component_spaces:
            raise ValueError(
                "training environment must expose a flat or structured observation space"
            )
        observation_size = 0
        for component in component_spaces.values():
            shape = getattr(component, "shape", None)
            if not shape or any(int(width) <= 0 for width in shape):
                raise ValueError("structured observation component has invalid shape")
            component_size = 1
            for width in shape:
                component_size *= int(width)
            observation_size += component_size
    observation_schema = getattr(unwrapped, "observation_schema", OBSERVATION_SCHEMA)
    observation_contract_digest = getattr(
        unwrapped, "observation_contract_digest", None
    )
    if not isinstance(observation_schema, str) or not observation_schema:
        raise ValueError("training environment must expose observation_schema")
    if observation_contract_digest is not None:
        require_sha256(observation_contract_digest, field="observation_contract_digest")
    return {
        "environment_digest": environment_digest,
        "initial_capital": float(initial_capital),
        "action_size": int(action_shape[0]),
        "action_names": tuple(getattr(unwrapped, "action_names", ())),
        "action_spec_digest": getattr(unwrapped, "action_spec_digest", None),
        "observation_size": observation_size,
        "observation_schema": observation_schema,
        "observation_contract_digest": observation_contract_digest,
        "decision_hours": getattr(unwrapped, "decision_hours", None),
        "alpha_artifact_digest": getattr(unwrapped, "alpha_artifact_digest", None),
        "factor_artifact_digest": getattr(unwrapped, "factor_artifact_digest", None),
        "normalizer_digest": _combined_normalizer_digest(unwrapped),
    }


def _validate_training_environment(
    identity: dict[str, Any],
    config: ResidualTrainingConfig,
) -> None:
    action_size = int(identity["action_size"])
    action_names = identity["action_names"]
    action_spec_digest = identity["action_spec_digest"]
    if not isinstance(action_names, tuple) or len(action_names) != action_size:
        raise ValueError("training environment must expose exact action_names")
    if not isinstance(action_spec_digest, str):
        raise ValueError("training environment must expose action_spec_digest")
    require_sha256(action_spec_digest, field="action_spec_digest")
    environment_decision_hours = identity["decision_hours"]
    if config.decision_hours is not None:
        if not isinstance(environment_decision_hours, int | float):
            raise ValueError("training environment must expose decision_hours")
        if not math.isclose(
            float(environment_decision_hours),
            config.decision_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("training decision_hours do not match the environment")


def train_residual_ensemble(
    *,
    dataset: DatasetManifest,
    environment_dataset_id: str,
    config: ResidualTrainingConfig,
    backend: PolicyTrainingBackend,
    output_dir: Path,
    created_at: datetime,
) -> PolicyEnsembleManifest:
    require_sha256(environment_dataset_id, field="environment_dataset_id")
    require_aware_datetime(created_at, field="created_at")
    if dataset.dataset_id != environment_dataset_id:
        raise ValueError("dataset identity mismatch between manifest and environment")
    output_dir.mkdir(parents=True, exist_ok=True)

    members: list[PolicyMember] = []
    results: list[PolicyTrainingResult] = []
    for member_index, seed in enumerate(config.seeds):
        checkpoint = output_dir / f"member-{member_index:03d}" / "policy.zip"
        result = backend.train(seed=seed, config=config, output_path=checkpoint)
        resolved_path = Path(result.checkpoint_path)
        if resolved_path.resolve() != checkpoint.resolve():
            raise ValueError("training backend returned an unexpected checkpoint path")
        results.append(result)
        members.append(
            PolicyMember(seed=seed, checkpoint_digest=_file_digest(resolved_path))
        )

    consistency_fields = (
        "actual_timesteps",
        "resolved_device",
        "environment_digest",
        "initial_capital",
        "action_size",
        "action_names",
        "action_spec_digest",
        "observation_size",
        "observation_schema",
        "alpha_artifact_digest",
        "factor_artifact_digest",
        "normalizer_digest",
    )
    values: dict[str, Any] = {}
    for field_name in consistency_fields:
        observed = {getattr(result, field_name) for result in results}
        if len(observed) != 1:
            raise ValueError(f"ensemble members reported inconsistent {field_name}")
        values[field_name] = observed.pop()

    training_config_digest = content_digest(config.digest_payload())
    digest_payload = {
        "action_schema": ACTION_SCHEMA,
        "action_names": values["action_names"],
        "action_size": values["action_size"],
        "action_spec_digest": values["action_spec_digest"],
        "actual_timesteps": values["actual_timesteps"],
        "alpha_artifact_digest": values["alpha_artifact_digest"],
        "created_at": created_at,
        "dataset_id": dataset.dataset_id,
        "environment_digest": values["environment_digest"],
        "factor_artifact_digest": values["factor_artifact_digest"],
        "initial_capital": values["initial_capital"],
        "members": tuple(
            {"checkpoint_digest": member.checkpoint_digest, "seed": member.seed}
            for member in members
        ),
        "normalizer_digest": values["normalizer_digest"],
        "observation_schema": values["observation_schema"],
        "observation_size": values["observation_size"],
        "requested_timesteps": config.timesteps,
        "resolved_device": values["resolved_device"],
        "schema_version": "policy_ensemble_v4",
        "training_config_digest": training_config_digest,
    }
    return PolicyEnsembleManifest(
        digest=content_digest(digest_payload),
        dataset_id=dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        observation_schema=str(values["observation_schema"]),
        training_config_digest=training_config_digest,
        environment_digest=str(values["environment_digest"]),
        initial_capital=float(values["initial_capital"]),
        requested_timesteps=config.timesteps,
        actual_timesteps=int(values["actual_timesteps"]),
        resolved_device=str(values["resolved_device"]),
        expected_members=len(config.seeds),
        members=tuple(members),
        created_at=created_at,
        action_size=int(values["action_size"]),
        action_names=tuple(values["action_names"]),
        action_spec_digest=str(values["action_spec_digest"]),
        observation_size=(
            None
            if values["observation_size"] is None
            else int(values["observation_size"])
        ),
        alpha_artifact_digest=values["alpha_artifact_digest"],
        factor_artifact_digest=values["factor_artifact_digest"],
        normalizer_digest=values["normalizer_digest"],
    )


__all__ = [
    "PolicyTrainingBackend",
    "PolicyTrainingResult",
    "ResidualTrainingConfig",
    "gamma_from_half_life",
    "train_residual_ensemble",
]
