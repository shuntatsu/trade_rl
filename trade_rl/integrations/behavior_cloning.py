"""Torch implementation of behavior-cloning warm starts for SB3 policies."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    ObservationBatchProvider,
)
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


def pretrain_policy(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    config: BehaviorCloningConfig,
    seed: int,
    observation_provider: ObservationBatchProvider | None = None,
) -> BehaviorCloningResult:
    """Fit actor-mean MSE with bounded batches and a chronological validation tail."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior-cloning seed must be non-negative")
    device = torch.device(policy.device)
    sample_count = dataset.sample_count
    validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(1, int(math.floor(sample_count * config.validation_fraction)))
    )
    train_count = sample_count - validation_count
    if train_count <= 0:
        raise ValueError("behavior-cloning validation leaves no training samples")
    all_indices = np.arange(sample_count, dtype=np.int64)
    train_indices = all_indices[:train_count]
    validation_indices = all_indices[train_count:]
    initial_mse = _mean_squared_error(
        policy,
        dataset,
        indices=all_indices,
        batch_size=config.batch_size,
        device=device,
        provider=observation_provider,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_state = copy.deepcopy(policy.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
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
            actions = torch.as_tensor(
                np.asarray(dataset.actions)[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            mean = actor_mean(policy, observations)
            loss = functional.mse_loss(mean, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if validation_count == 0:
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            continue
        validation_loss = _mean_squared_error(
            policy,
            dataset,
            indices=validation_indices,
            batch_size=config.batch_size,
            device=device,
            provider=observation_provider,
        )
        if validation_loss + config.minimum_improvement < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    policy.load_state_dict(best_state)

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
    )


__all__ = ["actor_mean", "pretrain_policy"]
