from __future__ import annotations

import importlib.util

import torch

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


def test_action_telemetry_module_is_not_maintained() -> None:
    assert importlib.util.find_spec("trade_rl.rl.action_telemetry") is None
