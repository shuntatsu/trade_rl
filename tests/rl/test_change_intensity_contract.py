from __future__ import annotations

import numpy as np
import pytest
import torch

from trade_rl.rl.action_telemetry import hierarchical_action_stage_metrics
from trade_rl.rl.policies import HierarchicalActorOutputs


def test_hierarchical_gate_is_exposed_as_change_intensity() -> None:
    intensity = torch.tensor([[0.25, 0.75]])
    outputs = HierarchicalActorOutputs(
        torch.zeros_like(intensity),
        intensity,
        torch.ones_like(intensity),
        torch.full_like(intensity, 0.5),
        torch.zeros_like(intensity),
        torch.zeros_like(intensity),
        torch.ones_like(intensity, dtype=torch.bool),
    )
    assert outputs.change_intensity is outputs.gate_probabilities


def test_action_stage_metrics_separate_intended_change_from_exploration() -> None:
    metrics = hierarchical_action_stage_metrics(
        current_weights=np.array([0.1, -0.1]),
        deterministic_composed=np.array([0.1, -0.1]),
        sampled_policy_action=np.array([0.4, -0.2]),
        submitted_order_target=np.array([0.3, -0.1]),
        effective_filled_weights=np.array([0.25, -0.05]),
    )

    assert metrics["deterministic_change_l1"] == pytest.approx(0.0)
    assert metrics["exploration_l1"] == pytest.approx(0.4)
    assert metrics["sampled_change_l1"] == pytest.approx(0.4)
    assert metrics["submission_l1"] == pytest.approx(0.2)
    assert metrics["effective_action_l1"] == pytest.approx(0.3)
