from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trade_rl.integrations.behavior_cloning import _evaluate_hierarchical
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.hierarchical_teacher_labels import (
    build_hierarchical_teacher_labels,
)
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


class _FixedHierarchicalPolicy:
    def hierarchical_actor_outputs(
        self,
        observations: dict[str, torch.Tensor],
    ) -> SimpleNamespace:
        sample_count = len(observations["feature"])
        gate_logits = torch.zeros((sample_count, 1), dtype=torch.float32)
        target_actions = torch.zeros((sample_count, 1), dtype=torch.float32)
        return SimpleNamespace(
            gate_logits=gate_logits,
            gate_probabilities=torch.sigmoid(gate_logits),
            target_actions=target_actions,
            composed_actions=target_actions,
        )


def test_hierarchical_evaluation_losses_are_batch_size_invariant() -> None:
    target_actions = np.asarray(
        [[0.0], [0.0], [0.0], [0.0], [0.2], [0.4], [0.6], [0.8]],
        dtype=np.float32,
    )
    current_weights = np.zeros_like(target_actions)
    dataset = SupervisedPolicyDataset(
        observations={
            "current_weights": current_weights,
            "feature": np.zeros_like(target_actions),
        },
        actions=target_actions,
        dataset_id="a" * 64,
        train_start=0,
        train_stop=9,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )
    labels = build_hierarchical_teacher_labels(
        teacher_targets=target_actions,
        current_weights=current_weights,
        active_mask=np.ones_like(target_actions, dtype=np.bool_),
        change_threshold=0.05,
        source_teacher_digest=dataset.action_digest,
    )
    config = BehaviorCloningConfig(epochs=1, batch_size=8)
    indices = np.arange(dataset.sample_count, dtype=np.int64)
    policy = _FixedHierarchicalPolicy()

    split_batches = _evaluate_hierarchical(
        policy,
        dataset,
        labels,
        indices=indices,
        batch_size=4,
        config=config,
        positive_class_weight=1.0,
        device=torch.device("cpu"),
        provider=None,
    )
    single_batch = _evaluate_hierarchical(
        policy,
        dataset,
        labels,
        indices=indices,
        batch_size=8,
        config=config,
        positive_class_weight=1.0,
        device=torch.device("cpu"),
        provider=None,
    )

    assert split_batches.metrics.digest == single_batch.metrics.digest
    assert split_batches.losses.gate == pytest.approx(single_batch.losses.gate)
    assert split_batches.losses.target == pytest.approx(single_batch.losses.target)
    assert split_batches.losses.composed == pytest.approx(
        single_batch.losses.composed
    )
    assert split_batches.losses.weighted == pytest.approx(
        single_batch.losses.weighted
    )
    assert single_batch.losses.target == pytest.approx(0.15)
    assert single_batch.losses.composed == pytest.approx(0.075)
