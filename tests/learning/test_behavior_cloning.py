from __future__ import annotations

import numpy as np
import torch

from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


class _Distribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.distribution = self
        self.mean = mean

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.mean


class _LinearPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.actor.bias)
        self.device = torch.device("cpu")

    def get_distribution(self, observations: torch.Tensor) -> _Distribution:
        return _Distribution(torch.tanh(self.actor(observations)))


def teacher_dataset() -> SupervisedPolicyDataset:
    observations = np.array(
        [[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    actions = observations[:, :1].copy()
    return SupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id="a" * 64,
        train_start=0,
        train_stop=5,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )


def test_behavior_cloning_reduces_actor_mean_error_and_reports_identity() -> None:
    policy = _LinearPolicy()
    dataset = teacher_dataset()

    result = pretrain_policy(
        policy,
        dataset,
        config=BehaviorCloningConfig(
            epochs=80,
            learning_rate=0.05,
            batch_size=4,
        ),
        seed=7,
    )

    assert result.final_mse < result.initial_mse * 0.1
    assert result.sample_count == 4
    assert result.teacher_config_digest == "d" * 64
    assert len(result.digest) == 64


def test_behavior_cloning_is_reproducible_for_fixed_seed() -> None:
    config = BehaviorCloningConfig(epochs=5, learning_rate=0.01, batch_size=2)
    first = _LinearPolicy()
    second = _LinearPolicy()

    first_result = pretrain_policy(first, teacher_dataset(), config=config, seed=11)
    second_result = pretrain_policy(second, teacher_dataset(), config=config, seed=11)

    assert first_result.digest == second_result.digest
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


class _StructuredLinearPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(3, 1)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.actor.bias)
        self.device = torch.device("cpu")

    def get_distribution(self, observations: dict[str, torch.Tensor]) -> _Distribution:
        joined = torch.cat((observations["left"], observations["right"]), dim=1)
        return _Distribution(torch.tanh(self.actor(joined)))


def test_behavior_cloning_supports_structured_observations_and_chronological_validation() -> (
    None
):
    observations = {
        "left": np.array(
            [[-1.0], [-0.5], [0.5], [1.0], [0.75], [-0.75]], dtype=np.float32
        ),
        "right": np.zeros((6, 2), dtype=np.float32),
    }
    dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=observations["left"].copy(),
        dataset_id="a" * 64,
        train_start=0,
        train_stop=7,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )
    result = pretrain_policy(
        _StructuredLinearPolicy(),
        dataset,
        config=BehaviorCloningConfig(
            epochs=80,
            learning_rate=0.05,
            batch_size=3,
            validation_fraction=1 / 3,
            early_stopping_patience=10,
        ),
        seed=5,
    )
    assert result.final_mse < result.initial_mse
    assert result.validation_sample_count == 2
    assert result.best_epoch > 0


class _TrackingProvider:
    def __init__(self, observations: np.ndarray) -> None:
        self.observations = observations
        self.sample_count = len(observations)
        self.maximum_requested_batch = 0

    def get(self, indices: np.ndarray) -> np.ndarray:
        self.maximum_requested_batch = max(self.maximum_requested_batch, len(indices))
        return self.observations[indices]


def test_behavior_cloning_never_materializes_more_than_one_configured_batch() -> None:
    dataset = teacher_dataset()
    assert isinstance(dataset.observations, np.ndarray)
    provider = _TrackingProvider(dataset.observations)
    config = BehaviorCloningConfig(epochs=3, learning_rate=0.01, batch_size=2)
    pretrain_policy(
        _LinearPolicy(),
        dataset,
        config=config,
        seed=4,
        observation_provider=provider,
    )
    assert provider.maximum_requested_batch <= config.batch_size


class _SquashedDistribution:
    def __init__(self, raw_mean: torch.Tensor) -> None:
        self.distribution = type("Gaussian", (), {"mean": raw_mean})()
        self.action_mode = torch.tanh(raw_mean)

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.action_mode


class _SquashedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float32))
        self.device = torch.device("cpu")

    def get_distribution(self, observations: torch.Tensor) -> _SquashedDistribution:
        return _SquashedDistribution(self.raw.expand(len(observations), -1))


def test_behavior_cloning_uses_deterministic_action_space_output() -> None:
    from trade_rl.integrations.behavior_cloning import actor_mean

    observations = torch.zeros((3, 1), dtype=torch.float32)
    policy = _SquashedPolicy()
    action = actor_mean(policy, observations)

    torch.testing.assert_close(action, torch.tanh(policy.raw).expand(3, -1))
    assert not torch.allclose(action, policy.raw.expand(3, -1))
