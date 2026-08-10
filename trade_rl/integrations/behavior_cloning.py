"""Torch implementation of behavior-cloning warm starts for SB3 policies."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as functional

from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    ObservationBatchProvider,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.hierarchical_bc_metrics import (
    HierarchicalBehaviorCloningLosses,
    HierarchicalBehaviorCloningMetrics,
    hierarchical_bc_metrics,
)
from trade_rl.learning.hierarchical_teacher_labels import HierarchicalTeacherLabels
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


def actor_mean(policy: Any, observations: Any) -> Any:
    distribution = policy.get_distribution(observations)
    action = distribution.get_actions(deterministic=True)
    if action is None or not hasattr(action, "shape"):
        raise ValueError(
            "policy distribution does not expose deterministic action-space output"
        )
    return action


def _tensor_observations(observations: object, *, device: Any) -> Any:
    if isinstance(observations, Mapping):
        return {
            key: torch.as_tensor(np.asarray(observations[key]), device=device)
            for key in sorted(observations)
        }
    return torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)


def _observation_batch(
    dataset: SupervisedPolicyDataset,
    indices: np.ndarray,
    *,
    provider: ObservationBatchProvider | None,
) -> object:
    if provider is not None:
        if provider.sample_count != dataset.sample_count:
            raise ValueError("observation provider sample count mismatch")
        return provider.get(indices)
    observations = dataset.observations
    if isinstance(observations, Mapping):
        return {key: np.asarray(value)[indices] for key, value in observations.items()}
    return np.asarray(observations)[indices]


def _mean_squared_error(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    indices: np.ndarray,
    batch_size: int,
    device: Any,
    provider: ObservationBatchProvider | None,
) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            observations = _tensor_observations(
                _observation_batch(dataset, batch_indices, provider=provider),
                device=device,
            )
            actions = torch.as_tensor(
                np.asarray(dataset.actions)[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            mean = actor_mean(policy, observations)
            if mean.shape != actions.shape:
                raise ValueError("teacher action shape does not match policy output")
            total += float(
                functional.mse_loss(mean, actions, reduction="sum").detach().cpu()
            )
            count += int(actions.numel())
    if count <= 0:
        raise ValueError("behavior-cloning evaluation batch is empty")
    return total / count


@dataclass(frozen=True, slots=True)
class _HierarchicalEvaluation:
    losses: HierarchicalBehaviorCloningLosses
    metrics: HierarchicalBehaviorCloningMetrics


def _positive_class_weight(
    labels: HierarchicalTeacherLabels,
    train_indices: np.ndarray,
    *,
    maximum: float,
) -> float:
    active = labels.active_mask[train_indices]
    positive = labels.gate_labels[train_indices] & active
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(active & ~positive))
    if positive_count == 0 or negative_count == 0:
        return 1.0
    # ``pos_weight`` balances both minority-positive and majority-positive
    # labels.  Clamping it to at least one makes a majority-positive gate's
    # constant optimum exceed 0.5, which is indistinguishable from an
    # all-trade policy at inference time.
    return min(maximum, max(1.0 / maximum, negative_count / positive_count))


def _hierarchical_batch_losses(
    policy: Any,
    observations: Any,
    labels: HierarchicalTeacherLabels,
    indices: np.ndarray,
    *,
    config: BehaviorCloningConfig,
    positive_class_weight: float,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    method = getattr(policy, "hierarchical_behavior_cloning_outputs", None)
    if not callable(method):
        method = getattr(policy, "hierarchical_actor_outputs", None)
    if not callable(method):
        raise ValueError("hierarchical BC requires policy hierarchical_actor_outputs")
    outputs = method(observations)
    gate_logits = outputs.gate_logits
    target_actions = outputs.target_actions
    composed_actions = outputs.composed_actions
    expected_shape = (len(indices), labels.action_count)
    if tuple(gate_logits.shape) != expected_shape:
        raise ValueError("hierarchical policy output does not match teacher labels")
    active = torch.as_tensor(
        labels.active_mask[indices], dtype=torch.bool, device=device
    )
    gate_labels = torch.as_tensor(
        labels.gate_labels[indices], dtype=torch.float32, device=device
    )
    targets = torch.as_tensor(
        labels.target_actions[indices], dtype=torch.float32, device=device
    )
    if not bool(torch.any(active)):
        raise ValueError("hierarchical BC batch contains no active action dimensions")
    gate_loss = functional.binary_cross_entropy_with_logits(
        gate_logits[active],
        gate_labels[active],
        pos_weight=torch.as_tensor(positive_class_weight, device=device),
    )
    event_mask = active & (gate_labels > 0.5)
    if bool(torch.any(event_mask)):
        target_loss = functional.smooth_l1_loss(
            target_actions[event_mask], targets[event_mask]
        )
    else:
        target_loss = target_actions.sum() * 0.0
    composed_loss = functional.smooth_l1_loss(composed_actions[active], targets[active])
    weighted = (
        config.gate_loss_weight * gate_loss
        + config.target_loss_weight * target_loss
        + config.composed_loss_weight * composed_loss
    )
    return outputs, gate_loss, target_loss, composed_loss, weighted


def _evaluate_hierarchical(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    labels: HierarchicalTeacherLabels,
    *,
    indices: np.ndarray,
    batch_size: int,
    config: BehaviorCloningConfig,
    positive_class_weight: float,
    device: torch.device,
    provider: ObservationBatchProvider | None,
) -> _HierarchicalEvaluation:
    gate_batches: list[np.ndarray] = []
    proposal_batches: list[np.ndarray] = []
    composed_batches: list[np.ndarray] = []
    gate_total = 0.0
    target_total = 0.0
    composed_total = 0.0
    weighted_total = 0.0
    batch_weight = 0
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            observations = _tensor_observations(
                _observation_batch(dataset, batch_indices, provider=provider),
                device=device,
            )
            outputs, gate, target, composed, weighted = _hierarchical_batch_losses(
                policy,
                observations,
                labels,
                batch_indices,
                config=config,
                positive_class_weight=positive_class_weight,
                device=device,
            )
            size = len(batch_indices)
            gate_total += float(gate.detach().cpu()) * size
            target_total += float(target.detach().cpu()) * size
            composed_total += float(composed.detach().cpu()) * size
            weighted_total += float(weighted.detach().cpu()) * size
            batch_weight += size
            gate_batches.append(outputs.gate_probabilities.detach().cpu().numpy())
            proposal_batches.append(outputs.target_actions.detach().cpu().numpy())
            composed_batches.append(outputs.composed_actions.detach().cpu().numpy())
    if batch_weight <= 0:
        raise ValueError("hierarchical BC evaluation batch is empty")
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.concatenate(gate_batches, axis=0),
        proposal_actions=np.concatenate(proposal_batches, axis=0),
        composed_actions=np.concatenate(composed_batches, axis=0),
        labels=labels,
        gate_threshold=config.gate_prediction_threshold,
        indices=indices,
    )
    return _HierarchicalEvaluation(
        losses=HierarchicalBehaviorCloningLosses(
            gate=gate_total / batch_weight,
            target=target_total / batch_weight,
            composed=composed_total / batch_weight,
            weighted=weighted_total / batch_weight,
        ),
        metrics=metrics,
    )


def _validate_hierarchical_labels(
    dataset: SupervisedPolicyDataset,
    labels: HierarchicalTeacherLabels,
) -> None:
    if labels.sample_count != dataset.sample_count:
        raise ValueError("hierarchical teacher labels sample count mismatch")
    actions = np.asarray(dataset.actions)
    if actions.ndim != 2 or labels.action_count != actions.shape[1]:
        raise ValueError("hierarchical teacher labels action count mismatch")
    if not np.array_equal(labels.target_actions, actions):
        raise ValueError("hierarchical teacher targets do not match supervised actions")


def _behavior_cloning_indices(
    *,
    sample_count: int,
    config: BehaviorCloningConfig,
    split: BehaviorCloningSplit | None,
) -> tuple[np.ndarray, np.ndarray]:
    if split is None:
        validation_count = (
            0
            if config.validation_fraction == 0.0
            else max(
                1,
                int(math.floor(sample_count * config.validation_fraction)),
            )
        )
        train_count = sample_count - validation_count
        if train_count <= 0:
            raise ValueError("behavior-cloning validation leaves no training samples")
        indices = np.arange(sample_count, dtype=np.int64)
        return indices[:train_count], indices[train_count:]

    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
    purged_indices = np.asarray(split.purged_indices, dtype=np.int64)
    for name, indices in (
        ("training", train_indices),
        ("validation", validation_indices),
        ("purged", purged_indices),
    ):
        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(f"behavior-cloning {name} index is outside the dataset")
    partition = np.concatenate((train_indices, validation_indices, purged_indices))
    expected_partition = np.arange(sample_count, dtype=np.int64)
    if partition.size != sample_count or not np.array_equal(
        np.sort(partition), expected_partition
    ):
        raise ValueError("explicit behavior-cloning split must partition the dataset")
    expected_validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(
            1,
            int(math.floor(sample_count * config.validation_fraction)),
        )
    )
    if validation_indices.size != expected_validation_count:
        raise ValueError(
            "explicit behavior-cloning split disagrees with validation_fraction"
        )
    return train_indices, validation_indices


def pretrain_policy(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    config: BehaviorCloningConfig,
    seed: int,
    split: BehaviorCloningSplit | None = None,
    observation_provider: ObservationBatchProvider | None = None,
    hierarchical_labels: HierarchicalTeacherLabels | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> BehaviorCloningResult:
    """Fit BC using a validated split while excluding purged samples."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior-cloning seed must be non-negative")
    if hierarchical_labels is not None:
        _validate_hierarchical_labels(dataset, hierarchical_labels)
    device = torch.device(policy.device)
    sample_count = dataset.sample_count
    train_indices, validation_indices = _behavior_cloning_indices(
        sample_count=sample_count,
        config=config,
        split=split,
    )
    train_count = int(train_indices.size)
    validation_count = int(validation_indices.size)
    all_indices = np.concatenate((train_indices, validation_indices))
    # Teacher targets and their reconstruction gates are deterministic contracts.
    # Eval mode disables stochastic regularizers while preserving autograd, so
    # both optimization and metrics fit the exact function deployed to PPO.
    policy.train(False)
    initial_mse = _mean_squared_error(
        policy,
        dataset,
        indices=all_indices,
        batch_size=config.batch_size,
        device=device,
        provider=observation_provider,
    )
    positive_class_weight = (
        1.0
        if hierarchical_labels is None
        else _positive_class_weight(
            hierarchical_labels,
            train_indices,
            maximum=config.max_positive_class_weight,
        )
    )
    initial_hierarchical = (
        None
        if hierarchical_labels is None
        else _evaluate_hierarchical(
            policy,
            dataset,
            hierarchical_labels,
            indices=all_indices,
            batch_size=config.batch_size,
            config=config,
            positive_class_weight=positive_class_weight,
            device=device,
            provider=observation_provider,
        )
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_state = copy.deepcopy(policy.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    started_at = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(train_count, generator=generator).numpy()
        shuffled = train_indices[permutation]
        for offset in range(0, train_count, config.batch_size):
            batch_indices = shuffled[offset : offset + config.batch_size]
            observations = _tensor_observations(
                _observation_batch(
                    dataset, batch_indices, provider=observation_provider
                ),
                device=device,
            )
            if hierarchical_labels is None:
                actions = torch.as_tensor(
                    np.asarray(dataset.actions)[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                mean = actor_mean(policy, observations)
                loss = functional.mse_loss(mean, actions)
            else:
                _, _, _, _, loss = _hierarchical_batch_losses(
                    policy,
                    observations,
                    hierarchical_labels,
                    batch_indices,
                    config=config,
                    positive_class_weight=positive_class_weight,
                    device=device,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_evaluation: _HierarchicalEvaluation | None = None
        validation_loss: float | None = None
        if validation_count == 0:
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
        elif hierarchical_labels is None:
            validation_loss = _mean_squared_error(
                policy,
                dataset,
                indices=validation_indices,
                batch_size=config.batch_size,
                device=device,
                provider=observation_provider,
            )
        else:
            validation_evaluation = _evaluate_hierarchical(
                policy,
                dataset,
                hierarchical_labels,
                indices=validation_indices,
                batch_size=config.batch_size,
                config=config,
                positive_class_weight=positive_class_weight,
                device=device,
                provider=observation_provider,
            )
            validation_loss = validation_evaluation.losses.weighted
        should_stop = False
        if validation_loss is not None:
            if validation_loss + config.minimum_improvement < best_validation:
                best_validation = validation_loss
                best_state = copy.deepcopy(policy.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                should_stop = stale_epochs >= config.early_stopping_patience
        if progress_callback is not None:
            elapsed_seconds = time.monotonic() - started_at
            estimated_remaining_seconds = (
                elapsed_seconds / epoch * (config.epochs - epoch)
            )
            metrics = (
                None if validation_evaluation is None else validation_evaluation.metrics
            )
            losses = (
                None if validation_evaluation is None else validation_evaluation.losses
            )
            progress_callback(
                {
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "best_epoch": best_epoch,
                    "elapsed_seconds": elapsed_seconds,
                    "estimated_remaining_seconds": estimated_remaining_seconds,
                    "validation_loss": validation_loss,
                    "gate_loss": None if losses is None else losses.gate,
                    "target_loss": None if losses is None else losses.target,
                    "composed_loss": None if losses is None else losses.composed,
                    "gate_precision": None
                    if metrics is None
                    else metrics.gate_precision,
                    "gate_recall": None if metrics is None else metrics.gate_recall,
                    "activity_ratio": None
                    if metrics is None
                    else metrics.activity_ratio,
                    "all_hold_collapse": None
                    if metrics is None
                    else metrics.all_hold_collapse,
                    "all_trade_collapse": None
                    if metrics is None
                    else metrics.all_trade_collapse,
                    "early_stopping": should_stop,
                }
            )
        if should_stop:
            break
    policy.load_state_dict(best_state)
    policy.train(False)

    final_mse = _mean_squared_error(
        policy,
        dataset,
        indices=all_indices,
        batch_size=config.batch_size,
        device=device,
        provider=observation_provider,
    )
    validation_mse = (
        None
        if validation_count == 0
        else _mean_squared_error(
            policy,
            dataset,
            indices=validation_indices,
            batch_size=config.batch_size,
            device=device,
            provider=observation_provider,
        )
    )
    final_hierarchical = (
        None
        if hierarchical_labels is None
        else _evaluate_hierarchical(
            policy,
            dataset,
            hierarchical_labels,
            indices=all_indices,
            batch_size=config.batch_size,
            config=config,
            positive_class_weight=positive_class_weight,
            device=device,
            provider=observation_provider,
        )
    )
    validation_hierarchical = (
        None
        if hierarchical_labels is None or validation_count == 0
        else _evaluate_hierarchical(
            policy,
            dataset,
            hierarchical_labels,
            indices=validation_indices,
            batch_size=config.batch_size,
            config=config,
            positive_class_weight=positive_class_weight,
            device=device,
            provider=observation_provider,
        )
    )
    return BehaviorCloningResult(
        initial_mse=initial_mse,
        final_mse=final_mse,
        sample_count=sample_count,
        observation_digest=dataset.observation_digest,
        action_digest=dataset.action_digest,
        teacher_config_digest=dataset.teacher_config_digest,
        config=config,
        seed=seed,
        validation_mse=validation_mse,
        validation_sample_count=validation_count,
        best_epoch=best_epoch,
        hierarchical_label_digest=(
            None
            if hierarchical_labels is None
            else hierarchical_labels.label_config_digest
        ),
        initial_hierarchical_losses=(
            None if initial_hierarchical is None else initial_hierarchical.losses
        ),
        final_hierarchical_losses=(
            None if final_hierarchical is None else final_hierarchical.losses
        ),
        validation_hierarchical_losses=(
            None if validation_hierarchical is None else validation_hierarchical.losses
        ),
        initial_hierarchical_metrics=(
            None if initial_hierarchical is None else initial_hierarchical.metrics
        ),
        final_hierarchical_metrics=(
            None if final_hierarchical is None else final_hierarchical.metrics
        ),
        validation_hierarchical_metrics=(
            None if validation_hierarchical is None else validation_hierarchical.metrics
        ),
    )


__all__ = ["actor_mean", "pretrain_policy"]
