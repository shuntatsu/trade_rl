from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from trade_rl.integrations.cost_rollout_buffer import (
    CostRolloutStorage,
    estimate_cost_rollout_storage_bytes,
)
from trade_rl.rl.cost_critics import FamilySeparatedCostCritic
from trade_rl.rl.cost_evidence import (
    build_cost_critic_compute_evidence,
    write_cost_critic_compute_evidence,
)
from trade_rl.rl.cost_learning import canonical_cost_learning_schema


def _trained_model() -> SimpleNamespace:
    schema = canonical_cost_learning_schema(
        auxiliary_event_loss_coefficient=0.25,
    )
    critic = FamilySeparatedCostCritic(
        input_dim=3,
        schema=schema,
        continuous_hidden_dims=(8,),
        event_hidden_dims=(6,),
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output = critic(torch.ones((4, 3), dtype=torch.float32))
    output.values.square().mean().backward()
    optimizer.step()
    storage = CostRolloutStorage(buffer_size=5, n_envs=2, schema=schema)
    return SimpleNamespace(
        algorithm_identifier="cost_critic_ppo",
        cost_schema=schema,
        cost_critic=critic,
        cost_critic_optimizer=optimizer,
        cost_rollout_storage=storage,
        cost_update_count=3,
        last_cost_training_metrics={
            "support/drawdown_stop_event": 2.0,
            "support/forced_liquidation_event": 1.0,
        },
    )


def test_cost_critic_compute_evidence_is_deterministic_and_complete(tmp_path) -> None:
    model = _trained_model()

    first = build_cost_critic_compute_evidence(model)
    second = build_cost_critic_compute_evidence(model)

    assert first == second
    assert first.algorithm_identifier == "cost_critic_ppo"
    assert first.cost_names == model.cost_schema.names
    assert first.cost_schema_digest == model.cost_schema.digest
    assert first.architecture_digest == model.cost_critic.architecture_digest
    assert first.cost_parameter_count == sum(
        parameter.numel() for parameter in model.cost_critic.parameters()
    )
    assert first.cost_parameter_bytes == sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.cost_critic.parameters()
    )
    assert first.cost_optimizer_state_bytes > 0
    assert first.rollout_storage_bytes == estimate_cost_rollout_storage_bytes(
        5,
        2,
        len(model.cost_schema.names),
    )
    assert first.rollout_transition_count == 10
    assert first.cost_optimizer_steps == 3
    assert first.event_positive_support == {
        "drawdown_stop_event": 2,
        "forced_liquidation_event": 1,
    }
    assert first.training_seconds is None
    assert first.environment_steps_per_second is None
    assert first.cost_optimizer_steps_per_second is None

    target = tmp_path / "cost-critic-evidence.json"
    write_cost_critic_compute_evidence(target, first)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cost_critic_compute_evidence_v1"
    assert payload["digest"] == first.digest
    assert payload["cost_names"] == list(model.cost_schema.names)
    assert payload["event_positive_support"] == first.event_positive_support


def test_cost_critic_compute_evidence_derives_timing_rates() -> None:
    evidence = build_cost_critic_compute_evidence(
        _trained_model(),
        training_seconds=2.0,
        environment_steps=8,
        peak_device_memory_bytes=4096,
    )

    assert evidence.training_seconds == pytest.approx(2.0)
    assert evidence.environment_steps == 8
    assert evidence.environment_steps_per_second == pytest.approx(4.0)
    assert evidence.cost_optimizer_steps_per_second == pytest.approx(1.5)
    assert evidence.peak_device_memory_bytes == 4096


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"training_seconds": 0.0}, "training_seconds"),
        ({"environment_steps": 0}, "environment_steps"),
        ({"peak_device_memory_bytes": -1}, "peak_device_memory_bytes"),
    ],
)
def test_cost_critic_compute_evidence_rejects_invalid_runtime_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_cost_critic_compute_evidence(_trained_model(), **kwargs)
