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
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.rl.training_modes import CudaRuntimeMode, ObservationEncoder


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
    cuda_runtime_mode: CudaRuntimeMode | str = CudaRuntimeMode.PERFORMANCE
    decision_hours: float | None = None
    discount_half_life_hours: float | None = None
    log_std_init: float = -0.5
    target_kl: float | None = 0.02
    use_sde: bool = False
    sde_sample_freq: int = -1
    policy_net_arch: tuple[int, ...] = (128, 128)
    value_net_arch: tuple[int, ...] = (128, 128)
    observation_encoder: ObservationEncoder | str = ObservationEncoder.ASSET_SET
    policy_actor_head: str | None = None
    hierarchical_gate_temperature: float = 1.0
    sequence_tcn_capacity: str = "standard"
    sequence_d_model: int = 320
    sequence_timeframe_attention_heads: int = 8
    sequence_timeframe_attention_layers: int = 2
    sequence_timeframe_ffn_multiplier: int = 3
    sequence_timeframe_gate_bias: float = -2.0
    sequence_asset_attention_heads: int = 8
    sequence_asset_attention_layers: int = 2
    sequence_asset_ffn_multiplier: int = 3
    sequence_asset_gate_bias: float = -2.0
    sequence_dropout: float = 0.05
    sequence_compile: bool = False
    sequence_compile_mode: str = "reduce-overhead"
    sequence_transfer_mode: str = "synchronous"
    max_policy_parameters: int = 12_000_000
    max_rollout_buffer_bytes: int = 805_306_368
    asset_embedding_dim: int = 64
    global_embedding_dim: int = 64
    algorithm: str = "ppo"
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    cost_learning_rate: float = 3e-4
    cost_n_epochs: int = 1
    cost_batch_size: int | None = None
    cost_continuous_hidden_dims: tuple[int, ...] = (128, 64)
    cost_event_hidden_dims: tuple[int, ...] = (128, 64)
    cost_max_grad_norm: float = 0.5
    cost_continuous_gae_lambda: float = 0.95
    cost_event_gae_lambda: float = 0.95
    cost_value_loss_coefficient: float = 1.0
    cost_auxiliary_event_loss_coefficient: float = 0.0
    cost_architecture_variant: str = "family_separated_v1"
    checkpoint_interval_steps: int | None = None
    max_checkpoints: int = 5
    n_envs: int = 1
    vector_environment_mode: str = "auto"
    behavior_cloning_epochs: int = 0
    behavior_cloning_learning_rate: float = 1e-3
    behavior_cloning_batch_size: int = 256
    behavior_cloning_validation_fraction: float = 0.0
    behavior_cloning_patience: int = 3
    behavior_cloning_minimum_improvement: float = 0.0
    behavior_cloning_teacher: str = "oracle"
    behavior_cloning_seed: int | None = None
    behavior_cloning_required_relative_improvement: float = 0.0
    behavior_cloning_gate_loss_weight: float = 1.0
    behavior_cloning_target_loss_weight: float = 1.0
    behavior_cloning_composed_loss_weight: float = 1.0
    behavior_cloning_gate_change_threshold: float = 0.05
    behavior_cloning_gate_prediction_threshold: float = 0.5
    behavior_cloning_max_positive_class_weight: float = 20.0
    behavior_cloning_min_gate_precision: float = 0.0
    behavior_cloning_min_gate_recall: float = 0.0
    behavior_cloning_max_active_target_rmse: float = 1.0
    behavior_cloning_min_activity_ratio: float = 0.0
    behavior_cloning_max_activity_ratio: float = 1.0
    behavior_cloning_min_causal_holdout_trades: int = 0
    behavior_cloning_max_causal_holdout_regret: float = 0.0
    behavior_cloning_causal_holdout_bootstrap_resamples: int = 2_000
    behavior_cloning_causal_holdout_confidence_level: float = 0.95
    lagrangian_budgets: tuple[float, ...] = ()
    lagrangian_dual_learning_rates: tuple[float, ...] = ()
    lagrangian_ema_betas: tuple[float, ...] = ()
    lagrangian_initial_multipliers: tuple[float, ...] = ()
    lagrangian_max_multipliers: tuple[float, ...] = ()
    lagrangian_warmup_rollouts: tuple[int, ...] = ()
    lagrangian_update_interval_rollouts: tuple[int, ...] = ()
    lagrangian_minimum_completed_episodes: tuple[int, ...] = ()
    lagrangian_probe_episodes: int = 0
    lagrangian_probe_max_steps_per_episode: int = 0
    learning_rate_schedule: str = "constant"
    learning_rate_final_ratio: float = 0.1
    tensorboard_enabled: bool = False
    tensorboard_log_interval: int = 1

    def __post_init__(self) -> None:
        for integer_field_name, integer_value in (
            ("timesteps", self.timesteps),
            ("n_steps", self.n_steps),
            ("n_envs", self.n_envs),
            ("batch_size", self.batch_size),
            ("n_epochs", self.n_epochs),
            ("tensorboard_log_interval", self.tensorboard_log_interval),
            ("buffer_size", self.buffer_size),
            ("train_freq", self.train_freq),
            ("gradient_steps", self.gradient_steps),
            ("cost_n_epochs", self.cost_n_epochs),
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
        if self.behavior_cloning_teacher not in {"oracle", "trend_baseline"}:
            raise ValueError(
                "behavior_cloning_teacher must be oracle or trend_baseline"
            )
        if self.behavior_cloning_seed is not None and (
            isinstance(self.behavior_cloning_seed, bool)
            or not isinstance(self.behavior_cloning_seed, int)
            or self.behavior_cloning_seed < 0
        ):
            raise ValueError(
                "behavior_cloning_seed must be a non-negative integer or null"
            )
        if (
            not math.isfinite(self.behavior_cloning_required_relative_improvement)
            or not 0.0 <= self.behavior_cloning_required_relative_improvement < 1.0
        ):
            raise ValueError(
                "behavior_cloning_required_relative_improvement must be within [0, 1)"
            )
        hierarchical_loss_weights = (
            self.behavior_cloning_gate_loss_weight,
            self.behavior_cloning_target_loss_weight,
            self.behavior_cloning_composed_loss_weight,
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in hierarchical_loss_weights
        ):
            raise ValueError(
                "hierarchical behavior cloning loss weights must be finite and non-negative"
            )
        if sum(hierarchical_loss_weights) <= 0.0:
            raise ValueError(
                "at least one hierarchical behavior cloning loss weight must be positive"
            )
        if (
            not math.isfinite(self.behavior_cloning_gate_change_threshold)
            or not 0.0 < self.behavior_cloning_gate_change_threshold <= 1.0
        ):
            raise ValueError(
                "behavior_cloning_gate_change_threshold must be within (0, 1]"
            )
        if (
            not math.isfinite(self.behavior_cloning_gate_prediction_threshold)
            or not 0.0 < self.behavior_cloning_gate_prediction_threshold < 1.0
        ):
            raise ValueError(
                "behavior_cloning_gate_prediction_threshold must be within (0, 1)"
            )
        if (
            not math.isfinite(self.behavior_cloning_max_positive_class_weight)
            or self.behavior_cloning_max_positive_class_weight < 1.0
        ):
            raise ValueError(
                "behavior_cloning_max_positive_class_weight must be at least one"
            )
        for field_name, value in (
            (
                "behavior_cloning_min_gate_precision",
                self.behavior_cloning_min_gate_precision,
            ),
            ("behavior_cloning_min_gate_recall", self.behavior_cloning_min_gate_recall),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
        if (
            not math.isfinite(self.behavior_cloning_max_active_target_rmse)
            or self.behavior_cloning_max_active_target_rmse < 0.0
        ):
            raise ValueError(
                "behavior_cloning_max_active_target_rmse must be finite and non-negative"
            )
        for field_name, value in (
            (
                "behavior_cloning_min_activity_ratio",
                self.behavior_cloning_min_activity_ratio,
            ),
            (
                "behavior_cloning_max_activity_ratio",
                self.behavior_cloning_max_activity_ratio,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            self.behavior_cloning_min_activity_ratio
            > self.behavior_cloning_max_activity_ratio
        ):
            raise ValueError("behavior cloning activity ratio bounds must be ordered")
        if (
            isinstance(self.behavior_cloning_min_causal_holdout_trades, bool)
            or not isinstance(self.behavior_cloning_min_causal_holdout_trades, int)
            or self.behavior_cloning_min_causal_holdout_trades < 0
        ):
            raise ValueError(
                "behavior_cloning_min_causal_holdout_trades must be a non-negative integer"
            )
        if (
            not math.isfinite(self.behavior_cloning_max_causal_holdout_regret)
            or self.behavior_cloning_max_causal_holdout_regret < 0.0
        ):
            raise ValueError(
                "behavior_cloning_max_causal_holdout_regret must be finite and non-negative"
            )
        if (
            isinstance(self.behavior_cloning_causal_holdout_bootstrap_resamples, bool)
            or not isinstance(
                self.behavior_cloning_causal_holdout_bootstrap_resamples, int
            )
            or self.behavior_cloning_causal_holdout_bootstrap_resamples < 1_000
        ):
            raise ValueError(
                "behavior_cloning_causal_holdout_bootstrap_resamples must be at least 1000"
            )
        if (
            not math.isfinite(self.behavior_cloning_causal_holdout_confidence_level)
            or not 0.5 < self.behavior_cloning_causal_holdout_confidence_level < 1.0
        ):
            raise ValueError(
                "behavior_cloning_causal_holdout_confidence_level must be within (0.5, 1)"
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
        algorithm = self.algorithm.lower()
        if algorithm not in {
            "ppo",
            "cost_critic_ppo",
            "lagrangian_ppo",
            "sac",
            "td3",
            "tqc",
        }:
            raise ValueError(
                "algorithm must be one of ppo, cost_critic_ppo, lagrangian_ppo, "
                "sac, td3, or tqc"
            )
        object.__setattr__(self, "algorithm", algorithm)
        ppo_like = algorithm in {"ppo", "cost_critic_ppo", "lagrangian_ppo"}
        rollout_size = self.n_steps * self.n_envs
        if ppo_like and rollout_size % self.batch_size != 0:
            raise ValueError("batch_size must divide the complete PPO rollout")
        if self.behavior_cloning_epochs > 0 and not ppo_like:
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
        if self.learning_rate_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "learning_rate_schedule must be constant, linear, or cosine"
            )
        if (
            not math.isfinite(self.learning_rate_final_ratio)
            or not 0.0 < self.learning_rate_final_ratio <= 1.0
        ):
            raise ValueError("learning_rate_final_ratio must be within (0, 1]")
        if not isinstance(self.tensorboard_enabled, bool):
            raise ValueError("tensorboard_enabled must be a boolean")
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
        if self.vector_environment_mode not in {
            "auto",
            "in_process",
            "subprocess",
        }:
            raise ValueError(
                "vector_environment_mode must be auto, in_process, or subprocess"
            )
        try:
            encoder = ObservationEncoder(str(self.observation_encoder).strip().lower())
        except ValueError as error:
            raise ValueError(
                "observation_encoder must be flat_mlp, asset_set, or "
                "hierarchical_sequence_v2"
            ) from error
        object.__setattr__(self, "observation_encoder", encoder)
        sequence_active = encoder is ObservationEncoder.HIERARCHICAL_SEQUENCE_V2
        actor_head = self.policy_actor_head
        if actor_head is None:
            actor_head = (
                "hierarchical_gate_target_v1"
                if sequence_active
                else "standard_continuous_v1"
            )
        if not isinstance(actor_head, str):
            raise ValueError("policy_actor_head must be a string")
        sequence_actor_heads = {
            "hierarchical_gate_target_v1",
            "shared_target_v1",
        }
        if sequence_active:
            if actor_head not in sequence_actor_heads:
                raise ValueError(
                    "policy_actor_head must be hierarchical_gate_target_v1 or "
                    "shared_target_v1 for observation_encoder="
                    f"{encoder}"
                )
        elif actor_head != "standard_continuous_v1":
            raise ValueError(
                "policy_actor_head must be standard_continuous_v1 for "
                f"observation_encoder={encoder}"
            )
        object.__setattr__(self, "policy_actor_head", actor_head)
        if (
            not math.isfinite(self.hierarchical_gate_temperature)
            or self.hierarchical_gate_temperature <= 0.0
        ):
            raise ValueError(
                "hierarchical_gate_temperature must be finite and positive"
            )
        if (
            actor_head == "shared_target_v1"
            and self.hierarchical_gate_temperature != 1.0
        ):
            raise ValueError(
                "hierarchical_gate_temperature is inactive for "
                "policy_actor_head=shared_target_v1"
            )
        if not sequence_active and self.hierarchical_gate_temperature != 1.0:
            raise ValueError(
                "hierarchical_gate_temperature is inactive for non-sequence actors"
            )
        try:
            cuda_runtime_mode = CudaRuntimeMode(
                str(self.cuda_runtime_mode).strip().lower()
            )
        except ValueError as error:
            raise ValueError(
                "cuda_runtime_mode must be deterministic or performance"
            ) from error
        object.__setattr__(self, "cuda_runtime_mode", cuda_runtime_mode)
        if not isinstance(self.sequence_compile, bool):
            raise ValueError("sequence_compile must be a boolean")
        if self.sequence_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError(
                "sequence_compile_mode must be default, reduce-overhead, or "
                "max-autotune"
            )
        if self.sequence_transfer_mode not in {
            "synchronous",
            "pinned_non_blocking",
        }:
            raise ValueError(
                "sequence_transfer_mode must be synchronous or pinned_non_blocking"
            )
        if (
            not self.sequence_compile
            and self.sequence_compile_mode != "reduce-overhead"
        ):
            raise ValueError(
                "sequence_compile_mode is inactive when sequence_compile is false"
            )
        if self.sequence_tcn_capacity not in {"standard", "compact"}:
            raise ValueError("sequence_tcn_capacity must be standard or compact")
        if sequence_active and self.policy != "MultiInputPolicy":
            raise ValueError("hierarchical_sequence_v2 requires MultiInputPolicy")
        if sequence_active and not ppo_like:
            raise ValueError(
                "hierarchical_sequence_v2 currently requires a PPO-family algorithm"
            )
        for field_name, value in (
            ("sequence_d_model", self.sequence_d_model),
            (
                "sequence_timeframe_attention_heads",
                self.sequence_timeframe_attention_heads,
            ),
            (
                "sequence_timeframe_attention_layers",
                self.sequence_timeframe_attention_layers,
            ),
            (
                "sequence_timeframe_ffn_multiplier",
                self.sequence_timeframe_ffn_multiplier,
            ),
            (
                "sequence_asset_attention_heads",
                self.sequence_asset_attention_heads,
            ),
            (
                "sequence_asset_attention_layers",
                self.sequence_asset_attention_layers,
            ),
            (
                "sequence_asset_ffn_multiplier",
                self.sequence_asset_ffn_multiplier,
            ),
            ("max_policy_parameters", self.max_policy_parameters),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name, gate_value in (
            (
                "sequence_timeframe_gate_bias",
                self.sequence_timeframe_gate_bias,
            ),
            ("sequence_asset_gate_bias", self.sequence_asset_gate_bias),
        ):
            if not math.isfinite(gate_value):
                raise ValueError(f"{field_name} must be finite")
        for field_name, heads in (
            (
                "sequence_timeframe_attention_heads",
                self.sequence_timeframe_attention_heads,
            ),
            (
                "sequence_asset_attention_heads",
                self.sequence_asset_attention_heads,
            ),
        ):
            if self.sequence_d_model % heads != 0:
                raise ValueError(
                    f"sequence_d_model must divide evenly across {field_name}"
                )
        if (
            not math.isfinite(self.sequence_dropout)
            or not 0.0 <= self.sequence_dropout <= 0.05
        ):
            raise ValueError("sequence_dropout must be within [0, 0.05]")
        if (
            isinstance(self.max_rollout_buffer_bytes, bool)
            or not isinstance(self.max_rollout_buffer_bytes, int)
            or self.max_rollout_buffer_bytes <= 0
        ):
            raise ValueError("max_rollout_buffer_bytes must be a positive integer")
        for field_name, value in (
            ("asset_embedding_dim", self.asset_embedding_dim),
            ("global_embedding_dim", self.global_embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if ppo_like:
            _require_inactive_defaults(
                (
                    ("buffer_size", self.buffer_size, 100_000),
                    ("learning_starts", self.learning_starts, 10_000),
                    ("train_freq", self.train_freq, 1),
                    ("gradient_steps", self.gradient_steps, 1),
                ),
                context=algorithm.upper(),
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

        if algorithm in {"cost_critic_ppo", "lagrangian_ppo"}:
            if (
                not math.isfinite(self.cost_learning_rate)
                or self.cost_learning_rate <= 0.0
            ):
                raise ValueError("cost_learning_rate must be finite and positive")
            if self.cost_batch_size is not None and (
                isinstance(self.cost_batch_size, bool)
                or not isinstance(self.cost_batch_size, int)
                or self.cost_batch_size <= 0
                or self.cost_batch_size > rollout_size
            ):
                raise ValueError(
                    "cost_batch_size must be null or within the complete PPO rollout"
                )
            for field_name, architecture in (
                ("cost_continuous_hidden_dims", self.cost_continuous_hidden_dims),
                ("cost_event_hidden_dims", self.cost_event_hidden_dims),
            ):
                if not architecture or any(
                    isinstance(width, bool) or not isinstance(width, int) or width <= 0
                    for width in architecture
                ):
                    raise ValueError(f"{field_name} must contain positive integers")
            if (
                not math.isfinite(self.cost_max_grad_norm)
                or self.cost_max_grad_norm <= 0.0
            ):
                raise ValueError("cost_max_grad_norm must be finite and positive")
            for field_name, lambda_value in (
                ("cost_continuous_gae_lambda", self.cost_continuous_gae_lambda),
                ("cost_event_gae_lambda", self.cost_event_gae_lambda),
            ):
                if not math.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
                    raise ValueError(f"{field_name} must be within [0, 1]")
            for field_name, cost_coefficient in (
                ("cost_value_loss_coefficient", self.cost_value_loss_coefficient),
                (
                    "cost_auxiliary_event_loss_coefficient",
                    self.cost_auxiliary_event_loss_coefficient,
                ),
            ):
                if not math.isfinite(cost_coefficient) or cost_coefficient < 0.0:
                    raise ValueError(f"{field_name} must be finite and non-negative")
            if self.cost_architecture_variant != "family_separated_v1":
                raise ValueError(
                    "cost_architecture_variant must be family_separated_v1"
                )
        else:
            _require_inactive_defaults(
                (
                    ("cost_learning_rate", self.cost_learning_rate, 3e-4),
                    ("cost_n_epochs", self.cost_n_epochs, 1),
                    ("cost_batch_size", self.cost_batch_size, None),
                    (
                        "cost_continuous_hidden_dims",
                        self.cost_continuous_hidden_dims,
                        (128, 64),
                    ),
                    (
                        "cost_event_hidden_dims",
                        self.cost_event_hidden_dims,
                        (128, 64),
                    ),
                    ("cost_max_grad_norm", self.cost_max_grad_norm, 0.5),
                    (
                        "cost_continuous_gae_lambda",
                        self.cost_continuous_gae_lambda,
                        0.95,
                    ),
                    ("cost_event_gae_lambda", self.cost_event_gae_lambda, 0.95),
                    (
                        "cost_value_loss_coefficient",
                        self.cost_value_loss_coefficient,
                        1.0,
                    ),
                    (
                        "cost_auxiliary_event_loss_coefficient",
                        self.cost_auxiliary_event_loss_coefficient,
                        0.0,
                    ),
                    (
                        "cost_architecture_variant",
                        self.cost_architecture_variant,
                        "family_separated_v1",
                    ),
                ),
                context=f"algorithm={algorithm}",
            )

        lagrangian_fields = (
            ("lagrangian_budgets", self.lagrangian_budgets, ()),
            (
                "lagrangian_dual_learning_rates",
                self.lagrangian_dual_learning_rates,
                (),
            ),
            ("lagrangian_ema_betas", self.lagrangian_ema_betas, ()),
            (
                "lagrangian_initial_multipliers",
                self.lagrangian_initial_multipliers,
                (),
            ),
            ("lagrangian_max_multipliers", self.lagrangian_max_multipliers, ()),
            ("lagrangian_warmup_rollouts", self.lagrangian_warmup_rollouts, ()),
            (
                "lagrangian_update_interval_rollouts",
                self.lagrangian_update_interval_rollouts,
                (),
            ),
            (
                "lagrangian_minimum_completed_episodes",
                self.lagrangian_minimum_completed_episodes,
                (),
            ),
            ("lagrangian_probe_episodes", self.lagrangian_probe_episodes, 0),
            (
                "lagrangian_probe_max_steps_per_episode",
                self.lagrangian_probe_max_steps_per_episode,
                0,
            ),
        )
        if algorithm == "lagrangian_ppo":
            expected_count = len(canonical_cost_learning_schema().names)
            for field_name, values, _default in lagrangian_fields[:8]:
                if not isinstance(values, tuple) or len(values) != expected_count:
                    raise ValueError(
                        f"{field_name} must contain exactly {expected_count} values"
                    )
            for field_name, value in (
                ("lagrangian_probe_episodes", self.lagrangian_probe_episodes),
                (
                    "lagrangian_probe_max_steps_per_episode",
                    self.lagrangian_probe_max_steps_per_episode,
                ),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{field_name} must be a positive integer")
            from trade_rl.rl.lagrangian import canonical_lagrangian_schema

            canonical_lagrangian_schema(
                names=canonical_cost_learning_schema().names,
                budgets=self.lagrangian_budgets,
                dual_learning_rates=self.lagrangian_dual_learning_rates,
                ema_betas=self.lagrangian_ema_betas,
                initial_multipliers=self.lagrangian_initial_multipliers,
                max_multipliers=self.lagrangian_max_multipliers,
                warmup_rollouts=self.lagrangian_warmup_rollouts,
                update_interval_rollouts=(self.lagrangian_update_interval_rollouts),
                minimum_completed_episodes=(self.lagrangian_minimum_completed_episodes),
            )
        else:
            _require_inactive_defaults(
                lagrangian_fields,
                context=f"algorithm={algorithm}",
            )

        if algorithm == "td3":
            if self.use_sde or self.sde_sample_freq != -1:
                raise ValueError("TD3 does not support SDE settings")

        if not sequence_active:
            _require_inactive_defaults(
                (
                    ("sequence_tcn_capacity", self.sequence_tcn_capacity, "standard"),
                    ("sequence_d_model", self.sequence_d_model, 320),
                    (
                        "sequence_timeframe_attention_heads",
                        self.sequence_timeframe_attention_heads,
                        8,
                    ),
                    (
                        "sequence_timeframe_attention_layers",
                        self.sequence_timeframe_attention_layers,
                        2,
                    ),
                    (
                        "sequence_timeframe_ffn_multiplier",
                        self.sequence_timeframe_ffn_multiplier,
                        3,
                    ),
                    (
                        "sequence_timeframe_gate_bias",
                        self.sequence_timeframe_gate_bias,
                        -2.0,
                    ),
                    (
                        "sequence_asset_attention_heads",
                        self.sequence_asset_attention_heads,
                        8,
                    ),
                    (
                        "sequence_asset_attention_layers",
                        self.sequence_asset_attention_layers,
                        2,
                    ),
                    (
                        "sequence_asset_ffn_multiplier",
                        self.sequence_asset_ffn_multiplier,
                        3,
                    ),
                    (
                        "sequence_asset_gate_bias",
                        self.sequence_asset_gate_bias,
                        -2.0,
                    ),
                    ("sequence_dropout", self.sequence_dropout, 0.05),
                    ("sequence_compile", self.sequence_compile, False),
                    (
                        "sequence_compile_mode",
                        self.sequence_compile_mode,
                        "reduce-overhead",
                    ),
                    (
                        "sequence_transfer_mode",
                        self.sequence_transfer_mode,
                        "synchronous",
                    ),
                ),
                context=f"observation_encoder={encoder}",
            )
        if encoder != "asset_set":
            _require_inactive_defaults(
                (
                    ("asset_embedding_dim", self.asset_embedding_dim, 64),
                    ("global_embedding_dim", self.global_embedding_dim, 64),
                ),
                context=f"observation_encoder={encoder}",
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
                    (
                        "behavior_cloning_teacher",
                        self.behavior_cloning_teacher,
                        "oracle",
                    ),
                    ("behavior_cloning_seed", self.behavior_cloning_seed, None),
                    (
                        "behavior_cloning_required_relative_improvement",
                        self.behavior_cloning_required_relative_improvement,
                        0.0,
                    ),
                    (
                        "behavior_cloning_gate_loss_weight",
                        self.behavior_cloning_gate_loss_weight,
                        1.0,
                    ),
                    (
                        "behavior_cloning_target_loss_weight",
                        self.behavior_cloning_target_loss_weight,
                        1.0,
                    ),
                    (
                        "behavior_cloning_composed_loss_weight",
                        self.behavior_cloning_composed_loss_weight,
                        1.0,
                    ),
                    (
                        "behavior_cloning_gate_change_threshold",
                        self.behavior_cloning_gate_change_threshold,
                        0.05,
                    ),
                    (
                        "behavior_cloning_gate_prediction_threshold",
                        self.behavior_cloning_gate_prediction_threshold,
                        0.5,
                    ),
                    (
                        "behavior_cloning_max_positive_class_weight",
                        self.behavior_cloning_max_positive_class_weight,
                        20.0,
                    ),
                    (
                        "behavior_cloning_min_gate_precision",
                        self.behavior_cloning_min_gate_precision,
                        0.0,
                    ),
                    (
                        "behavior_cloning_min_gate_recall",
                        self.behavior_cloning_min_gate_recall,
                        0.0,
                    ),
                    (
                        "behavior_cloning_max_active_target_rmse",
                        self.behavior_cloning_max_active_target_rmse,
                        1.0,
                    ),
                    (
                        "behavior_cloning_min_activity_ratio",
                        self.behavior_cloning_min_activity_ratio,
                        0.0,
                    ),
                    (
                        "behavior_cloning_max_activity_ratio",
                        self.behavior_cloning_max_activity_ratio,
                        1.0,
                    ),
                    (
                        "behavior_cloning_min_causal_holdout_trades",
                        self.behavior_cloning_min_causal_holdout_trades,
                        0,
                    ),
                    (
                        "behavior_cloning_max_causal_holdout_regret",
                        self.behavior_cloning_max_causal_holdout_regret,
                        0.0,
                    ),
                    (
                        "behavior_cloning_causal_holdout_bootstrap_resamples",
                        self.behavior_cloning_causal_holdout_bootstrap_resamples,
                        2_000,
                    ),
                    (
                        "behavior_cloning_causal_holdout_confidence_level",
                        self.behavior_cloning_causal_holdout_confidence_level,
                        0.95,
                    ),
                ),
                context="behavior cloning disabled",
            )

    @property
    def rounded_timesteps(self) -> int:
        if self.algorithm in {"ppo", "cost_critic_ppo", "lagrangian_ppo"}:
            rollout_size = self.n_steps * self.n_envs
            return math.ceil(self.timesteps / rollout_size) * rollout_size
        return self.timesteps

    @property
    def resolved_checkpoint_interval(self) -> int:
        if self.checkpoint_interval_steps is not None:
            return self.checkpoint_interval_steps
        return max(1, math.ceil(self.timesteps / self.max_checkpoints))

    def digest_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm": self.algorithm,
            "asset_embedding_dim": self.asset_embedding_dim,
            "observation_encoder": str(self.observation_encoder),
            "policy_actor_head": self.policy_actor_head,
            "hierarchical_gate_temperature": self.hierarchical_gate_temperature,
            "batch_size": self.batch_size,
            "behavior_cloning_batch_size": self.behavior_cloning_batch_size,
            "behavior_cloning_epochs": self.behavior_cloning_epochs,
            "behavior_cloning_learning_rate": self.behavior_cloning_learning_rate,
            "behavior_cloning_validation_fraction": self.behavior_cloning_validation_fraction,
            "behavior_cloning_patience": self.behavior_cloning_patience,
            "behavior_cloning_minimum_improvement": self.behavior_cloning_minimum_improvement,
            "behavior_cloning_teacher": self.behavior_cloning_teacher,
            "behavior_cloning_seed": self.behavior_cloning_seed,
            "behavior_cloning_required_relative_improvement": (
                self.behavior_cloning_required_relative_improvement
            ),
            "behavior_cloning_gate_loss_weight": self.behavior_cloning_gate_loss_weight,
            "behavior_cloning_target_loss_weight": self.behavior_cloning_target_loss_weight,
            "behavior_cloning_composed_loss_weight": self.behavior_cloning_composed_loss_weight,
            "behavior_cloning_gate_change_threshold": (
                self.behavior_cloning_gate_change_threshold
            ),
            "behavior_cloning_gate_prediction_threshold": (
                self.behavior_cloning_gate_prediction_threshold
            ),
            "behavior_cloning_max_positive_class_weight": (
                self.behavior_cloning_max_positive_class_weight
            ),
            "behavior_cloning_min_gate_precision": (
                self.behavior_cloning_min_gate_precision
            ),
            "behavior_cloning_min_gate_recall": self.behavior_cloning_min_gate_recall,
            "behavior_cloning_max_active_target_rmse": (
                self.behavior_cloning_max_active_target_rmse
            ),
            "behavior_cloning_min_activity_ratio": (
                self.behavior_cloning_min_activity_ratio
            ),
            "behavior_cloning_max_activity_ratio": (
                self.behavior_cloning_max_activity_ratio
            ),
            "behavior_cloning_min_causal_holdout_trades": (
                self.behavior_cloning_min_causal_holdout_trades
            ),
            "behavior_cloning_max_causal_holdout_regret": (
                self.behavior_cloning_max_causal_holdout_regret
            ),
            "behavior_cloning_causal_holdout_bootstrap_resamples": (
                self.behavior_cloning_causal_holdout_bootstrap_resamples
            ),
            "behavior_cloning_causal_holdout_confidence_level": (
                self.behavior_cloning_causal_holdout_confidence_level
            ),
            "buffer_size": self.buffer_size,
            "global_embedding_dim": self.global_embedding_dim,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "clip_range": self.clip_range,
            "decision_hours": self.decision_hours,
            "device": self.device,
            "cuda_runtime_mode": str(self.cuda_runtime_mode),
            "discount_half_life_hours": self.discount_half_life_hours,
            "ent_coef": self.ent_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "gradient_steps": self.gradient_steps,
            "learning_rate": self.learning_rate,
            "learning_rate_final_ratio": self.learning_rate_final_ratio,
            "learning_rate_schedule": self.learning_rate_schedule,
            "learning_starts": self.learning_starts,
            "log_std_init": self.log_std_init,
            "max_checkpoints": self.max_checkpoints,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "n_envs": self.n_envs,
            "vector_environment_mode": self.vector_environment_mode,
            "n_steps": self.n_steps,
            "normalize_advantage": self.normalize_advantage,
            "policy": self.policy,
            "policy_net_arch": self.policy_net_arch,
            "value_net_arch": self.value_net_arch,
            "sequence_tcn_capacity": self.sequence_tcn_capacity,
            "sequence_d_model": self.sequence_d_model,
            "sequence_timeframe_attention_heads": self.sequence_timeframe_attention_heads,
            "sequence_asset_attention_heads": self.sequence_asset_attention_heads,
            "sequence_timeframe_attention_layers": self.sequence_timeframe_attention_layers,
            "sequence_timeframe_ffn_multiplier": self.sequence_timeframe_ffn_multiplier,
            "sequence_timeframe_gate_bias": self.sequence_timeframe_gate_bias,
            "sequence_asset_attention_layers": self.sequence_asset_attention_layers,
            "sequence_asset_ffn_multiplier": self.sequence_asset_ffn_multiplier,
            "sequence_asset_gate_bias": self.sequence_asset_gate_bias,
            "sequence_dropout": self.sequence_dropout,
            "sequence_compile": self.sequence_compile,
            "sequence_compile_mode": self.sequence_compile_mode,
            "sequence_transfer_mode": self.sequence_transfer_mode,
            "max_policy_parameters": self.max_policy_parameters,
            "max_rollout_buffer_bytes": self.max_rollout_buffer_bytes,
            "sde_sample_freq": self.sde_sample_freq,
            "seeds": self.seeds,
            "target_kl": self.target_kl,
            "tensorboard_enabled": self.tensorboard_enabled,
            "tensorboard_log_interval": self.tensorboard_log_interval,
            "timesteps": self.timesteps,
            "train_freq": self.train_freq,
            "use_sde": self.use_sde,
            "vf_coef": self.vf_coef,
        }
        if self.algorithm in {"cost_critic_ppo", "lagrangian_ppo"}:
            cost_schema = canonical_cost_learning_schema(
                continuous_gae_lambda=self.cost_continuous_gae_lambda,
                event_gae_lambda=self.cost_event_gae_lambda,
                value_loss_coefficient=self.cost_value_loss_coefficient,
                auxiliary_event_loss_coefficient=(
                    self.cost_auxiliary_event_loss_coefficient
                ),
            )
            payload["cost_critic"] = {
                "architecture_variant": self.cost_architecture_variant,
                "batch_size": self.cost_batch_size,
                "continuous_hidden_dims": self.cost_continuous_hidden_dims,
                "event_hidden_dims": self.cost_event_hidden_dims,
                "learning_rate": self.cost_learning_rate,
                "max_grad_norm": self.cost_max_grad_norm,
                "n_epochs": self.cost_n_epochs,
                "schema": cost_schema.digest_payload(),
                "schema_digest": cost_schema.digest,
            }
        if self.algorithm == "lagrangian_ppo":
            from trade_rl.rl.lagrangian import canonical_lagrangian_schema

            lagrangian_schema = canonical_lagrangian_schema(
                names=cost_schema.names,
                budgets=self.lagrangian_budgets,
                dual_learning_rates=self.lagrangian_dual_learning_rates,
                ema_betas=self.lagrangian_ema_betas,
                initial_multipliers=self.lagrangian_initial_multipliers,
                max_multipliers=self.lagrangian_max_multipliers,
                warmup_rollouts=self.lagrangian_warmup_rollouts,
                update_interval_rollouts=(self.lagrangian_update_interval_rollouts),
                minimum_completed_episodes=(self.lagrangian_minimum_completed_episodes),
            )
            payload["lagrangian"] = {
                "actor_composition_mode": "raw_lagrangian_then_sb3_normalize_v1",
                "probe_episodes": self.lagrangian_probe_episodes,
                "probe_max_steps_per_episode": (
                    self.lagrangian_probe_max_steps_per_episode
                ),
                "schema": lagrangian_schema.digest_payload(),
                "schema_digest": lagrangian_schema.digest,
            }
        return payload


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
    structured_export_manifest_path: Path | None = None
    structured_export_manifest_digest: str | None = None
    structured_export_model_path: Path | None = None
    structured_export_model_digest: str | None = None
    architecture_digest: str | None = None

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
        structured_export_values = (
            self.structured_export_manifest_path,
            self.structured_export_manifest_digest,
            self.structured_export_model_path,
            self.structured_export_model_digest,
        )
        if any(value is not None for value in structured_export_values) and (
            any(value is None for value in structured_export_values)
            or self.architecture_digest is None
        ):
            raise ValueError("structured export identity must be complete")
        for field_name, value in (
            (
                "structured_export_manifest_digest",
                self.structured_export_manifest_digest,
            ),
            ("structured_export_model_digest", self.structured_export_model_digest),
            ("architecture_digest", self.architecture_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)


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
        "architecture_digest",
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
    if values["architecture_digest"] is not None:
        digest_payload["architecture_digest"] = values["architecture_digest"]
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
        architecture_digest=values["architecture_digest"],
    )


__all__ = [
    "PolicyTrainingBackend",
    "PolicyTrainingResult",
    "ResidualTrainingConfig",
    "gamma_from_half_life",
    "train_residual_ensemble",
]
