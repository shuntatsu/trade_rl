"""Family-separated value modules for independent constraint-cost learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.cost_learning import CostFamily, CostLearningSchema


@dataclass(frozen=True, slots=True)
class CostCriticOutput:
    """Ordered Cost Critic predictions and optional rare-event logits."""

    cost_names: tuple[str, ...]
    values: torch.Tensor
    auxiliary_event_names: tuple[str, ...]
    auxiliary_event_logits: torch.Tensor | None


def _validated_hidden_dims(
    values: tuple[int, ...],
    *,
    field_name: str,
) -> tuple[int, ...]:
    dims = tuple(values)
    if not dims or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        for width in dims
    ):
        raise ValueError(f"{field_name} must contain positive integer widths")
    return dims


def _adapter(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend(
            (
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            )
        )
        width = hidden
    return nn.Sequential(*layers)


class FamilySeparatedCostCritic(nn.Module):
    """Use disjoint adapters for continuous and rare-event cost families."""

    def __init__(
        self,
        *,
        input_dim: int,
        schema: CostLearningSchema,
        continuous_hidden_dims: tuple[int, ...],
        event_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim <= 0
        ):
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(schema, CostLearningSchema):
            raise TypeError("schema must be a CostLearningSchema")
        continuous_dims = _validated_hidden_dims(
            continuous_hidden_dims,
            field_name="continuous_hidden_dims",
        )
        event_dims = _validated_hidden_dims(
            event_hidden_dims,
            field_name="event_hidden_dims",
        )
        self.input_dim = input_dim
        self.schema = schema
        self.cost_names = schema.names
        self.continuous_hidden_dims = continuous_dims
        self.event_hidden_dims = event_dims
        self.continuous_adapter = _adapter(input_dim, continuous_dims)
        self.event_adapter = _adapter(input_dim, event_dims)
        self.value_heads = nn.ModuleDict(
            {
                spec.name: nn.Linear(
                    continuous_dims[-1]
                    if spec.family is CostFamily.CONTINUOUS
                    else event_dims[-1],
                    1,
                )
                for spec in schema.specs
            }
        )
        auxiliary_event_names = tuple(
            spec.name
            for spec in schema.specs
            if spec.family is CostFamily.EVENT
            and spec.auxiliary_event_loss_coefficient > 0.0
        )
        self.auxiliary_event_names = auxiliary_event_names
        self.event_logit_heads = nn.ModuleDict(
            {name: nn.Linear(event_dims[-1], 1) for name in auxiliary_event_names}
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def architecture_payload(self) -> dict[str, object]:
        return {
            "architecture": "family_separated_cost_critic_v1",
            "auxiliary_event_names": list(self.auxiliary_event_names),
            "continuous_hidden_dims": list(self.continuous_hidden_dims),
            "continuous_names": list(self.schema.continuous_names),
            "cost_names": list(self.cost_names),
            "event_hidden_dims": list(self.event_hidden_dims),
            "event_names": list(self.schema.event_names),
            "input_dim": self.input_dim,
            "schema_digest": self.schema.digest,
        }

    @property
    def architecture_digest(self) -> str:
        return content_digest(self.architecture_payload())

    def forward(self, features: torch.Tensor) -> CostCriticOutput:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"cost critic features must have shape [batch, {self.input_dim}]"
            )
        continuous_latent = self.continuous_adapter(features)
        event_latent = self.event_adapter(features)
        values = torch.cat(
            tuple(
                self.value_heads[spec.name](
                    continuous_latent
                    if spec.family is CostFamily.CONTINUOUS
                    else event_latent
                )
                for spec in self.schema.specs
            ),
            dim=1,
        )
        auxiliary_logits: torch.Tensor | None = None
        if self.auxiliary_event_names:
            auxiliary_logits = torch.cat(
                tuple(
                    self.event_logit_heads[name](event_latent)
                    for name in self.auxiliary_event_names
                ),
                dim=1,
            )
        return CostCriticOutput(
            cost_names=self.cost_names,
            values=values,
            auxiliary_event_names=self.auxiliary_event_names,
            auxiliary_event_logits=auxiliary_logits,
        )


__all__ = [
    "CostCriticOutput",
    "FamilySeparatedCostCritic",
]
