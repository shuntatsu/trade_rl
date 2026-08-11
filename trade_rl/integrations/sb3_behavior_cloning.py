"""Behavior-cloning helpers used by Stable-Baselines3 orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest
from trade_rl.learning import (
    BehaviorCloningConfig,
    BehaviorCloningGateEvaluation,
    BehaviorCloningGateThresholds,
    SupervisedPolicyDataset,
    evaluate_behavior_cloning_gates,
)
from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    build_hierarchical_teacher_labels,
)
from trade_rl.rl.checkpointing import save_policy_without_runtime_state
from trade_rl.rl.training import ResidualTrainingConfig


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


def _resolve_behavior_cloning_seed(
    config: ResidualTrainingConfig,
    *,
    member_seed: int,
) -> int:
    """Keep supervised initialization stable without removing PPO diversity."""

    configured = config.behavior_cloning_seed
    return member_seed if configured is None else configured


def _save_behavior_cloning_policy_candidate(
    model: Any,
    *,
    output_dir: Path,
) -> tuple[Path, str]:
    """Persist the pre-PPO policy so selection can reject harmful fine-tuning."""

    output_dir.mkdir(parents=True, exist_ok=True)
    save_target = output_dir / "behavior-cloning-policy"
    save_policy_without_runtime_state(model, str(save_target))
    policy_path = save_target.with_suffix(".zip")
    if not policy_path.is_file():
        raise FileNotFoundError(
            "behavior cloning model save did not create behavior-cloning-policy.zip"
        )
    return policy_path, file_digest(policy_path, field="behavior cloning policy")


def _restore_member_seed_after_behavior_cloning(
    model: Any,
    *,
    behavior_cloning_seed: int,
    member_seed: int,
) -> None:
    """Restore the member RNG after deterministic supervised pretraining."""

    model.set_random_seed(member_seed)


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
        minimum_causal_holdout_episodes=(
            config.behavior_cloning_min_causal_holdout_episodes
        ),
        maximum_causal_holdout_regret_upper_bound=float(
            _required_hierarchical_config(
                config, "behavior_cloning_max_causal_holdout_regret"
            )
        ),
        minimum_causal_holdout_net_return_lower_bound=(
            config.behavior_cloning_min_causal_holdout_net_return_lower_bound
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
