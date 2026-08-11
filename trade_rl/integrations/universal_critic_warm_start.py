"""Finite-horizon critic warm-start for universal behavior-cloned policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from trade_rl.learning.behavior_cloning import ObservationBatchProvider
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.learning.teacher_artifact import ObservationBatch, SupervisedPolicyDataset
from trade_rl.learning.universal_bc import CriticWarmStartPlan

TensorObservationBatch = torch.Tensor | dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class CriticWarmStartResult:
    sample_count: int
    initial_value_mse: float
    critic_only_value_mse: float
    final_value_mse: float
    initial_actor_mse: float
    final_actor_mse: float
    actor_max_abs_drift_critic_only: float
    actor_max_abs_drift_joint: float


def _validated_gamma(gamma: float) -> float:
    value = float(gamma)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("critic warm-start gamma must be finite and in (0, 1]")
    return value


def _discounted_return_to_go(rewards: Sequence[float], *, gamma: float) -> np.ndarray:
    result = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def collect_episode_return_targets(
    environment: Any,
    episode_batch: EpisodeOracleBatch,
    *,
    gamma: float,
) -> np.ndarray:
    """Replay teacher actions and compute finite-horizon reward-to-go per episode."""

    resolved_gamma = _validated_gamma(gamma)
    all_returns: list[np.ndarray] = []
    for contract, raw_targets in zip(
        episode_batch.contracts, episode_batch.targets, strict=True
    ):
        targets = np.asarray(raw_targets, dtype=np.float32)
        expected_steps = contract.stop - contract.start - 1
        if targets.ndim != 2 or len(targets) != expected_steps:
            raise ValueError("oracle episode targets do not match the episode contract")
        environment.reset(
            options={
                "start_idx": contract.start,
                "episode_bars": expected_steps,
                "initial_state_mode": contract.initial_state_mode,
            }
        )
        rewards: list[float] = []
        for offset, target in enumerate(targets):
            _, reward, terminated, truncated, _ = environment.step(target)
            resolved_reward = float(reward)
            if not math.isfinite(resolved_reward):
                raise ValueError("critic warm-start rewards must be finite")
            final_step = offset == expected_steps - 1
            if final_step:
                if not bool(terminated) or bool(truncated):
                    raise RuntimeError(
                        "critic warm-start teacher replay must terminate exactly at the finite horizon"
                    )
            elif bool(terminated) or bool(truncated):
                raise RuntimeError(
                    "critic warm-start teacher replay ended before the episode horizon"
                )
            rewards.append(resolved_reward)
        all_returns.append(_discounted_return_to_go(rewards, gamma=resolved_gamma))
    if not all_returns:
        raise ValueError("critic warm-start requires at least one teacher episode")
    return np.concatenate(all_returns).astype(np.float32, copy=False)


def _validated_sample_indices(
    sample_indices: np.ndarray | None,
    *,
    sample_count: int,
) -> np.ndarray:
    if sample_indices is None:
        return np.arange(sample_count, dtype=np.int64)
    raw = np.asarray(sample_indices)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("critic warm-start sample_indices must be an integer vector")
    resolved = np.asarray(raw, dtype=np.int64).copy(order="C")
    if resolved.size == 0:
        raise ValueError("critic warm-start sample_indices must not be empty")
    if np.any(resolved < 0) or np.any(resolved >= sample_count):
        raise ValueError("critic warm-start sample_indices contain an out-of-range index")
    if np.unique(resolved).size != resolved.size:
        raise ValueError("critic warm-start sample_indices must not contain duplicates")
    return resolved


def _observation_batch(
    dataset: SupervisedPolicyDataset,
    indices: np.ndarray,
    *,
    provider: ObservationBatchProvider | None,
) -> object:
    if provider is not None:
        if provider.sample_count != dataset.sample_count:
            raise ValueError("critic warm-start observation provider sample count mismatch")
        return provider.get(indices)
    observations = dataset.observations
    if isinstance(observations, Mapping):
        return {key: np.asarray(value)[indices] for key, value in observations.items()}
    return np.asarray(observations)[indices]


def _tensor_observations(
    observations: object,
    *,
    device: torch.device,
) -> TensorObservationBatch:
    if isinstance(observations, Mapping):
        return {
            str(key): torch.as_tensor(
                np.array(value, copy=True, order="C"),
                device=device,
            )
            for key, value in observations.items()
        }
    return torch.as_tensor(
        np.array(observations, dtype=np.float32, copy=True, order="C"),
        dtype=torch.float32,
        device=device,
    )


def _policy_actions(policy: Any, observations: TensorObservationBatch) -> torch.Tensor:
    distribution = policy.get_distribution(observations)
    actions = distribution.get_actions(deterministic=True)
    if not isinstance(actions, torch.Tensor):
        raise TypeError("policy distribution must return torch actions")
    return actions.float()


def _policy_values(policy: Any, observations: TensorObservationBatch) -> torch.Tensor:
    values = policy.predict_values(observations)
    if not isinstance(values, torch.Tensor):
        raise TypeError("policy critic must return torch values")
    return values.float().reshape(-1)


def _mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(left - right))


def _critic_parameters(policy: Any) -> tuple[nn.Parameter, ...]:
    extractor = getattr(policy, "mlp_extractor", None)
    critic_net = getattr(extractor, "critic_net", None)
    value_net = getattr(policy, "value_net", None)
    if not isinstance(critic_net, nn.Module) or not isinstance(value_net, nn.Module):
        raise TypeError(
            "critic warm-start requires policy.mlp_extractor.critic_net and policy.value_net"
        )
    parameters = tuple(critic_net.parameters()) + tuple(value_net.parameters())
    if not parameters:
        raise ValueError("critic warm-start found no critic parameters")
    return parameters


def _evaluation_batches(indices: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, ...]:
    return tuple(
        indices[offset : offset + batch_size]
        for offset in range(0, len(indices), batch_size)
    )


def _actor_snapshot(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    sample_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    provider: ObservationBatchProvider | None,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in _evaluation_batches(sample_indices, batch_size=batch_size):
            observations = _tensor_observations(
                _observation_batch(dataset, indices, provider=provider),
                device=device,
            )
            batches.append(_policy_actions(policy, observations).detach().cpu())
    if not batches:
        raise ValueError("critic warm-start actor evaluation scope is empty")
    return torch.cat(batches, dim=0).clone()


def _value_mse(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    targets: np.ndarray,
    sample_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    provider: ObservationBatchProvider | None,
) -> float:
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for indices in _evaluation_batches(sample_indices, batch_size=batch_size):
            observations = _tensor_observations(
                _observation_batch(dataset, indices, provider=provider),
                device=device,
            )
            expected = torch.as_tensor(
                np.array(targets[indices], dtype=np.float32, copy=True),
                dtype=torch.float32,
                device=device,
            )
            errors = torch.square(_policy_values(policy, observations) - expected)
            squared_error += float(errors.sum().cpu().item())
            count += int(errors.numel())
    if count <= 0:
        raise ValueError("critic warm-start value evaluation scope is empty")
    return squared_error / count


def _actor_mse(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    actions: np.ndarray,
    sample_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    provider: ObservationBatchProvider | None,
) -> float:
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for indices in _evaluation_batches(sample_indices, batch_size=batch_size):
            observations = _tensor_observations(
                _observation_batch(dataset, indices, provider=provider),
                device=device,
            )
            expected = torch.as_tensor(
                np.array(actions[indices], dtype=np.float32, copy=True),
                dtype=torch.float32,
                device=device,
            )
            errors = torch.square(_policy_actions(policy, observations) - expected)
            squared_error += float(errors.sum().cpu().item())
            count += int(errors.numel())
    if count <= 0:
        raise ValueError("critic warm-start actor evaluation scope is empty")
    return squared_error / count


def _max_abs_drift(before: torch.Tensor, after: torch.Tensor) -> float:
    if before.shape != after.shape:
        raise RuntimeError("actor output shape changed during critic warm-start")
    if before.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(before - after)).item())


def _sample_batch_indices(
    rng: np.random.Generator,
    *,
    sample_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size >= len(sample_indices):
        return sample_indices.copy(order="C")
    return np.asarray(
        rng.choice(sample_indices, size=batch_size, replace=False),
        dtype=np.int64,
    )


def _batch_targets(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.array(values[indices], dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )


def warm_start_policy_actor_critic(
    policy: Any,
    teacher_dataset: SupervisedPolicyDataset,
    critic_targets: np.ndarray,
    *,
    plan: CriticWarmStartPlan,
    batch_size: int,
    learning_rate: float,
    seed: int,
    sample_indices: np.ndarray | None = None,
    observation_provider: ObservationBatchProvider | None = None,
) -> CriticWarmStartResult:
    """Fit critic first with actor frozen, then conservatively fine-tune both."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("critic warm-start batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("critic warm-start learning_rate must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("critic warm-start seed must be non-negative")

    actions_np = np.asarray(teacher_dataset.actions, dtype=np.float32)
    returns_np = np.asarray(critic_targets, dtype=np.float32).reshape(-1)
    if len(returns_np) != len(actions_np):
        raise ValueError("critic target count must match teacher action count")
    if not np.isfinite(returns_np).all():
        raise ValueError("critic warm-start targets must be finite")
    training_indices = _validated_sample_indices(
        sample_indices,
        sample_count=len(actions_np),
    )
    if observation_provider is not None and (
        observation_provider.sample_count != teacher_dataset.sample_count
    ):
        raise ValueError("critic warm-start observation provider sample count mismatch")

    device = torch.device(getattr(policy, "device", "cpu"))
    evaluation_batch_size = min(batch_size, len(training_indices))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    initial_actor = _actor_snapshot(
        policy,
        teacher_dataset,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )
    initial_value_mse = _value_mse(
        policy,
        teacher_dataset,
        returns_np,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )
    initial_actor_mse = _actor_mse(
        policy,
        teacher_dataset,
        actions_np,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )

    critic_parameters = _critic_parameters(policy)
    critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
    all_parameters = tuple(policy.parameters())
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in all_parameters
    }
    for parameter in all_parameters:
        parameter.requires_grad_(id(parameter) in critic_parameter_ids)

    critic_optimizer = torch.optim.Adam(critic_parameters, lr=learning_rate)
    try:
        for _ in range(plan.critic_only_steps):
            indices = _sample_batch_indices(
                rng,
                sample_indices=training_indices,
                batch_size=batch_size,
            )
            batch_observations = _tensor_observations(
                _observation_batch(
                    teacher_dataset,
                    indices,
                    provider=observation_provider,
                ),
                device=device,
            )
            batch_values = _batch_targets(returns_np, indices, device=device)
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss = _mse(_policy_values(policy, batch_observations), batch_values)
            value_loss.backward()
            critic_optimizer.step()
    finally:
        for parameter in all_parameters:
            parameter.requires_grad_(original_requires_grad[id(parameter)])

    after_critic_actor = _actor_snapshot(
        policy,
        teacher_dataset,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )
    actor_drift_critic_only = _max_abs_drift(initial_actor, after_critic_actor)
    critic_only_value_mse = _value_mse(
        policy,
        teacher_dataset,
        returns_np,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )

    critic_group = [
        parameter
        for parameter in all_parameters
        if original_requires_grad[id(parameter)]
        and id(parameter) in critic_parameter_ids
    ]
    actor_group = [
        parameter
        for parameter in all_parameters
        if original_requires_grad[id(parameter)]
        and id(parameter) not in critic_parameter_ids
    ]
    parameter_groups: list[dict[str, object]] = []
    if critic_group:
        parameter_groups.append({"params": critic_group, "lr": learning_rate})
    if actor_group:
        parameter_groups.append(
            {
                "params": actor_group,
                "lr": learning_rate * plan.joint_actor_learning_rate_scale,
            }
        )
    if not parameter_groups:
        raise ValueError("critic warm-start found no trainable policy parameters")
    joint_optimizer = torch.optim.Adam(parameter_groups)

    for _ in range(plan.joint_fine_tune_steps):
        indices = _sample_batch_indices(
            rng,
            sample_indices=training_indices,
            batch_size=batch_size,
        )
        batch_observations = _tensor_observations(
            _observation_batch(
                teacher_dataset,
                indices,
                provider=observation_provider,
            ),
            device=device,
        )
        batch_actions = _batch_targets(actions_np, indices, device=device)
        batch_values = _batch_targets(returns_np, indices, device=device)
        joint_optimizer.zero_grad(set_to_none=True)
        actor_loss = _mse(_policy_actions(policy, batch_observations), batch_actions)
        value_loss = _mse(_policy_values(policy, batch_observations), batch_values)
        (actor_loss + value_loss).backward()
        joint_optimizer.step()

    final_actor = _actor_snapshot(
        policy,
        teacher_dataset,
        training_indices,
        batch_size=evaluation_batch_size,
        device=device,
        provider=observation_provider,
    )
    return CriticWarmStartResult(
        sample_count=len(training_indices),
        initial_value_mse=initial_value_mse,
        critic_only_value_mse=critic_only_value_mse,
        final_value_mse=_value_mse(
            policy,
            teacher_dataset,
            returns_np,
            training_indices,
            batch_size=evaluation_batch_size,
            device=device,
            provider=observation_provider,
        ),
        initial_actor_mse=initial_actor_mse,
        final_actor_mse=_actor_mse(
            policy,
            teacher_dataset,
            actions_np,
            training_indices,
            batch_size=evaluation_batch_size,
            device=device,
            provider=observation_provider,
        ),
        actor_max_abs_drift_critic_only=actor_drift_critic_only,
        actor_max_abs_drift_joint=_max_abs_drift(after_critic_actor, final_actor),
    )
