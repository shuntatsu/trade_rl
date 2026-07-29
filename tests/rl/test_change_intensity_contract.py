from __future__ import annotations

import numpy as np
import torch

from trade_rl.rl.action_telemetry import hierarchical_action_stage_metrics
from trade_rl.rl.policies import HierarchicalActorOutputs


def test_hierarchical_gate_is_exposed_as_change_intensity() -> None:
    intensity = torch.tensor([[0.25, 0.75]])
    outputs = HierarchicalActorOutputs(torch.zeros_like(intensity), intensity, torch.ones_like(intensity), torch.full_like(intensity, 0.5), torch.zeros_like(intensity), torch.zeros_like(intensity), torch.ones_like(intensity, dtype=torch.bool))
    assert outputs.change_intensity is outputs.gate_probabilities


def test_action_stage_metrics_measure_exploration_and_effective_action() -> None:
    metrics = hierarchical_action_stage_metrics(
        deterministic_composed=np.array([0.1, 0.2]),
        sampled_policy_action=np.array([0.4, 0.0]),
        submitted_target=np.array([0.3, 0.0]),
        effective_filled_weights=np.array([0.25, 0.05]),
    )
    assert metrics == {"exploration_l1": 0.5, "submission_l1": 0.1, "effective_action_l1": 0.2}
