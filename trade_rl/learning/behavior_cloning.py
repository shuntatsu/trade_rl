"""Deterministic behavior-cloning warm start for continuous SB3 policies."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


class ObservationBatchProvider(Protocol):
    sample_count: int

    def get(self, indices: np.ndarray) -> object: ...


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    epochs: int = 15
    learning_rate: float = 1e-3
    batch_size: int = 256
    validation_fraction: float = 0.0
    early_stopping_patience: int = 3
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("early_stopping_patience", self.early_stopping_patience),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.validation_fraction)
            or not 0.0 <= self.validation_fraction < 0.5
        ):
            raise ValueError("validation_fraction must be within [0, 0.5)")
        if (
            not math.isfinite(self.minimum_improvement)
            or self.minimum_improvement < 0.0
        ):
            raise ValueError("minimum_improvement must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BehaviorCloningResult:
    initial_mse: float
    final_mse: float
    sample_count: int
    observation_digest: str
    action_digest: str
    teacher_config_digest: str
    config: BehaviorCloningConfig
    seed: int
    validation_mse: float | None = None
    validation_sample_count: int = 0
    best_epoch: int = 0

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "action_digest": self.action_digest,
                "best_epoch": self.best_epoch,
                "config": asdict(self.config),
                "final_mse": self.final_mse,
                "initial_mse": self.initial_mse,
                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v2",
                "seed": self.seed,
                "teacher_config_digest": self.teacher_config_digest,
                "validation_mse": self.validation_mse,
                "validation_sample_count": self.validation_sample_count,
            }
        )


def _actor_mean(policy: Any, observations: Any) -> Any:
    distribution = policy.get_distribution(observations)
    mean = getattr(distribution.distribution, "mean", None)
    if mean is None:
        raise ValueError("policy distribution does not expose a continuous mean")
    return mean


def _tensor_observations(observations: object, *, device: Any) -> Any:
    import torch

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
    import torch
    import torch.nn.functional as functional

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
            mean = _actor_mean(policy, observations)
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
    """Fit actor-mean MSE with bounded mini-batches and a chronological validation tail."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior-cloning seed must be non-negative")
    import torch
    import torch.nn.functional as functional

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
            mean = _actor_mean(policy, observations)
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


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "ObservationBatchProvider",
    "pretrain_policy",
]
