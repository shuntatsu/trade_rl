"""Deterministic behavior-cloning warm start for continuous SB3 policies."""

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        for name, value in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")


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

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "action_digest": self.action_digest,
                "config": asdict(self.config),
                "final_mse": self.final_mse,
                "initial_mse": self.initial_mse,
                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v1",
                "seed": self.seed,
                "teacher_config_digest": self.teacher_config_digest,
            }
        )


def _actor_mean(policy: Any, observations: Any) -> Any:
    distribution = policy.get_distribution(observations)
    mean = getattr(distribution.distribution, "mean", None)
    if mean is None:
        raise ValueError("policy distribution does not expose a continuous mean")
    return mean


def pretrain_policy(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    config: BehaviorCloningConfig,
    seed: int,
) -> BehaviorCloningResult:
    """Fit only parameters reached by the actor-mean MSE computation."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior-cloning seed must be non-negative")
    import torch
    import torch.nn.functional as functional

    device = torch.device(policy.device)
    observations = torch.as_tensor(
        np.asarray(dataset.observations).copy(), dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(
        np.asarray(dataset.actions).copy(), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        initial_mean = _actor_mean(policy, observations)
        if initial_mean.shape != actions.shape:
            raise ValueError("teacher action shape does not match policy output")
        initial_mse = float(functional.mse_loss(initial_mean, actions).cpu())

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample_count = len(observations)
    for _ in range(config.epochs):
        permutation = torch.randperm(
            sample_count,
            generator=generator,
            device=device,
        )
        for offset in range(0, sample_count, config.batch_size):
            indices = permutation[offset : offset + config.batch_size]
            mean = _actor_mean(policy, observations[indices])
            loss = functional.mse_loss(mean, actions[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_mse = float(
            functional.mse_loss(_actor_mean(policy, observations), actions).cpu()
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
    )


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "pretrain_policy",
]
