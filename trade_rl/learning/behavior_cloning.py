"""Deterministic behavior-cloning warm start for continuous SB3 policies."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


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
            key: torch.as_tensor(
                np.asarray(observations[key]).copy(),
                device=device,
            )
            for key in sorted(observations)
        }
    return torch.as_tensor(
        np.asarray(observations).copy(), dtype=torch.float32, device=device
    )


def _index_observations(observations: Any, indices: Any) -> Any:
    if isinstance(observations, dict):
        return {key: value[indices] for key, value in observations.items()}
    return observations[indices]


def _slice_observations(observations: Any, start: int, stop: int) -> Any:
    if isinstance(observations, dict):
        return {key: value[start:stop] for key, value in observations.items()}
    return observations[start:stop]


def pretrain_policy(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    config: BehaviorCloningConfig,
    seed: int,
) -> BehaviorCloningResult:
    """Fit actor-mean MSE using a chronological train-only validation tail."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior-cloning seed must be non-negative")
    import torch
    import torch.nn.functional as functional

    device = torch.device(policy.device)
    observations = _tensor_observations(dataset.observations, device=device)
    actions = torch.as_tensor(
        np.asarray(dataset.actions).copy(), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        initial_mean = _actor_mean(policy, observations)
        if initial_mean.shape != actions.shape:
            raise ValueError("teacher action shape does not match policy output")
        initial_mse = float(functional.mse_loss(initial_mean, actions).cpu())

    sample_count = len(actions)
    validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(1, int(math.floor(sample_count * config.validation_fraction)))
    )
    train_count = sample_count - validation_count
    if train_count <= 0:
        raise ValueError("behavior-cloning validation leaves no training samples")
    train_observations = _slice_observations(observations, 0, train_count)
    train_actions = actions[:train_count]
    validation_observations = _slice_observations(
        observations, train_count, sample_count
    )
    validation_actions = actions[train_count:]

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_state = copy.deepcopy(policy.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(
            train_count,
            generator=generator,
            device=device,
        )
        for offset in range(0, train_count, config.batch_size):
            indices = permutation[offset : offset + config.batch_size]
            mean = _actor_mean(policy, _index_observations(train_observations, indices))
            loss = functional.mse_loss(mean, train_actions[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if validation_count == 0:
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            continue
        with torch.no_grad():
            validation_loss = float(
                functional.mse_loss(
                    _actor_mean(policy, validation_observations),
                    validation_actions,
                ).cpu()
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

    with torch.no_grad():
        final_mse = float(
            functional.mse_loss(_actor_mean(policy, observations), actions).cpu()
        )
        validation_mse = (
            None
            if validation_count == 0
            else float(
                functional.mse_loss(
                    _actor_mean(policy, validation_observations), validation_actions
                ).cpu()
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
    "pretrain_policy",
]
