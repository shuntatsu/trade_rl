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


class _HierarchicalOutputs:
    def __init__(
        self,
        *,
        gate_logits: torch.Tensor,
        target_actions: torch.Tensor,
        composed_actions: torch.Tensor,
    ) -> None:
        self.gate_logits = gate_logits
        self.gate_probabilities = torch.sigmoid(gate_logits)
        self.target_actions = target_actions
        self.composed_actions = composed_actions


class _HierarchicalPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(1, 1)
        self.target = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, -2.0)
        torch.nn.init.zeros_(self.target.weight)
        torch.nn.init.zeros_(self.target.bias)
        self.device = torch.device("cpu")

    def hierarchical_actor_outputs(
        self, observations: dict[str, torch.Tensor]
    ) -> _HierarchicalOutputs:
        feature = observations["feature"]
        current = observations["current_weights"]
        gate_logits = self.gate(feature)
        target = torch.tanh(self.target(feature))
        probability = torch.sigmoid(gate_logits)
        composed = current + probability * (target - current)
        return _HierarchicalOutputs(
            gate_logits=gate_logits,
            target_actions=target,
            composed_actions=composed,
        )

    def get_distribution(self, observations: dict[str, torch.Tensor]) -> _Distribution:
        return _Distribution(
            self.hierarchical_actor_outputs(observations).composed_actions
        )


def _hierarchical_case() -> tuple[SupervisedPolicyDataset, object]:
    from trade_rl.learning.hierarchical_teacher_labels import (
        build_hierarchical_teacher_labels,
    )

    feature = np.array(
        [[-1.0], [-0.8], [-0.4], [-0.1], [0.1], [0.4], [0.8], [1.0]],
        dtype=np.float32,
    )
    current = np.zeros_like(feature)
    actions = np.where(feature > 0.0, 0.7, 0.0).astype(np.float32)
    dataset = SupervisedPolicyDataset(
        observations={"feature": feature, "current_weights": current},
        actions=actions,
        dataset_id="a" * 64,
        train_start=0,
        train_stop=9,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )
    labels = build_hierarchical_teacher_labels(
        teacher_targets=actions,
        current_weights=current,
        active_mask=np.ones_like(actions, dtype=np.bool_),
        change_threshold=0.05,
        source_teacher_digest=dataset.action_digest,
    )
    return dataset, labels


def test_hierarchical_behavior_cloning_trains_component_losses_and_metrics() -> None:
    dataset, labels = _hierarchical_case()
    result = pretrain_policy(
        _HierarchicalPolicy(),
        dataset,
        config=BehaviorCloningConfig(
            epochs=120,
            learning_rate=0.05,
            batch_size=4,
            validation_fraction=0.25,
            early_stopping_patience=20,
            gate_loss_weight=2.0,
            target_loss_weight=1.0,
            composed_loss_weight=1.0,
        ),
        seed=13,
        hierarchical_labels=labels,
    )

    assert result.initial_hierarchical_losses is not None
    assert result.final_hierarchical_losses is not None
    assert result.final_hierarchical_metrics is not None
    assert result.validation_hierarchical_metrics is not None
    assert (
        result.final_hierarchical_losses.weighted
        < result.initial_hierarchical_losses.weighted
    )
    assert result.final_hierarchical_metrics.positive_support == 4
    assert result.final_hierarchical_metrics.all_hold_collapse is False
    assert result.hierarchical_label_digest == labels.label_config_digest


def test_hierarchical_behavior_cloning_reports_hold_collapse_without_events() -> None:
    dataset, labels = _hierarchical_case()
    policy = _HierarchicalPolicy()
    result = pretrain_policy(
        policy,
        dataset,
        config=BehaviorCloningConfig(
            epochs=1,
            learning_rate=1e-12,
            batch_size=8,
        ),
        seed=3,
        hierarchical_labels=labels,
    )

    assert result.final_hierarchical_metrics is not None
    assert result.final_hierarchical_metrics.all_hold_collapse is True
    assert result.final_hierarchical_metrics.gate_recall == 0.0
