from __future__ import annotations

import torch

from trade_rl.rl.cost_learning import (
    CostLearningSchema,
    canonical_cost_learning_schema,
)


def _gradient_norm(parameters: object) -> float:
    total = 0.0
    for parameter in parameters:  # type: ignore[union-attr]
        gradient = parameter.grad
        if gradient is not None:
            total += float(gradient.detach().abs().sum())
    return total


def test_family_separated_cost_critic_preserves_schema_order() -> None:
    from trade_rl.rl.cost_critics import FamilySeparatedCostCritic

    schema = canonical_cost_learning_schema(auxiliary_event_loss_coefficient=0.25)
    critic = FamilySeparatedCostCritic(
        input_dim=6,
        schema=schema,
        continuous_hidden_dims=(8, 4),
        event_hidden_dims=(7, 3),
    )
    with torch.no_grad():
        for index, name in enumerate(schema.names):
            critic.value_heads[name].weight.zero_()
            critic.value_heads[name].bias.fill_(float(index))
        for index, name in enumerate(schema.event_names):
            critic.event_logit_heads[name].weight.zero_()
            critic.event_logit_heads[name].bias.fill_(float(10 + index))

    output = critic(torch.zeros(5, 6))

    assert output.cost_names == schema.names
    assert output.values.shape == (5, 7)
    torch.testing.assert_close(
        output.values,
        torch.arange(7, dtype=torch.float32).repeat(5, 1),
    )
    assert output.auxiliary_event_names == schema.event_names
    assert output.auxiliary_event_logits is not None
    torch.testing.assert_close(
        output.auxiliary_event_logits,
        torch.tensor([10.0, 11.0]).repeat(5, 1),
    )


def test_cost_critic_omits_disabled_heads_instead_of_zero_filling() -> None:
    from trade_rl.rl.cost_critics import FamilySeparatedCostCritic

    canonical = canonical_cost_learning_schema()
    schema = CostLearningSchema(
        (
            canonical["drawdown_excess"],
            canonical["forced_liquidation_event"],
        )
    )
    critic = FamilySeparatedCostCritic(
        input_dim=4,
        schema=schema,
        continuous_hidden_dims=(5,),
        event_hidden_dims=(6,),
    )

    output = critic(torch.randn(3, 4))

    assert tuple(critic.value_heads) == schema.names
    assert output.values.shape == (3, 2)
    assert output.auxiliary_event_names == ()
    assert output.auxiliary_event_logits is None


def test_continuous_loss_does_not_update_rare_event_adapter() -> None:
    from trade_rl.rl.cost_critics import FamilySeparatedCostCritic

    schema = canonical_cost_learning_schema()
    critic = FamilySeparatedCostCritic(
        input_dim=5,
        schema=schema,
        continuous_hidden_dims=(9, 5),
        event_hidden_dims=(8, 4),
    )
    output = critic(torch.randn(7, 5))
    continuous_indices = [schema.names.index(name) for name in schema.continuous_names]

    loss = (output.values[:, continuous_indices] - 1.0).square().mean()
    loss.backward()

    assert _gradient_norm(critic.continuous_adapter.parameters()) > 0.0
    assert _gradient_norm(
        parameter
        for name in schema.continuous_names
        for parameter in critic.value_heads[name].parameters()
    ) > 0.0
    assert _gradient_norm(critic.event_adapter.parameters()) == 0.0
    assert _gradient_norm(
        parameter
        for name in schema.event_names
        for parameter in critic.value_heads[name].parameters()
    ) == 0.0


def test_positive_event_targets_produce_rare_adapter_gradients() -> None:
    from trade_rl.rl.cost_critics import FamilySeparatedCostCritic

    schema = canonical_cost_learning_schema(auxiliary_event_loss_coefficient=0.5)
    critic = FamilySeparatedCostCritic(
        input_dim=5,
        schema=schema,
        continuous_hidden_dims=(9, 5),
        event_hidden_dims=(8, 4),
    )
    output = critic(torch.randn(8, 5))
    event_indices = [schema.names.index(name) for name in schema.event_names]
    assert output.auxiliary_event_logits is not None
    cumulative_loss = (output.values[:, event_indices] - 1.0).square().mean()
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.auxiliary_event_logits,
        torch.ones_like(output.auxiliary_event_logits),
    )

    (cumulative_loss + classification_loss).backward()

    assert _gradient_norm(critic.event_adapter.parameters()) > 0.0
    assert _gradient_norm(
        parameter
        for name in schema.event_names
        for parameter in critic.value_heads[name].parameters()
    ) > 0.0
    assert _gradient_norm(
        parameter
        for name in schema.event_names
        for parameter in critic.event_logit_heads[name].parameters()
    ) > 0.0


def test_cost_critic_architecture_identity_tracks_schema_and_widths() -> None:
    from trade_rl.rl.cost_critics import FamilySeparatedCostCritic

    schema = canonical_cost_learning_schema()
    baseline = FamilySeparatedCostCritic(
        input_dim=6,
        schema=schema,
        continuous_hidden_dims=(8, 4),
        event_hidden_dims=(7, 3),
    )
    wider = FamilySeparatedCostCritic(
        input_dim=6,
        schema=schema,
        continuous_hidden_dims=(9, 4),
        event_hidden_dims=(7, 3),
    )
    event_schema = canonical_cost_learning_schema(
        auxiliary_event_loss_coefficient=0.25
    )
    auxiliary = FamilySeparatedCostCritic(
        input_dim=6,
        schema=event_schema,
        continuous_hidden_dims=(8, 4),
        event_hidden_dims=(7, 3),
    )

    assert baseline.architecture_digest != wider.architecture_digest
    assert baseline.architecture_digest != auxiliary.architecture_digest
    assert baseline.parameter_count == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert baseline.architecture_payload()["cost_names"] == list(schema.names)
