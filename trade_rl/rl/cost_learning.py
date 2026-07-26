"""Typed optimization identity for independent constraint-cost value learning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES

_EVENT_COST_NAMES = frozenset(
    {
        "drawdown_stop_event",
        "forced_liquidation_event",
    }
)


class CostFamily(str, Enum):
    """Representation family used by a maintained Cost Critic head."""

    CONTINUOUS = "continuous"
    EVENT = "event"


def _expected_family(name: str) -> CostFamily:
    return CostFamily.EVENT if name in _EVENT_COST_NAMES else CostFamily.CONTINUOUS


@dataclass(frozen=True, slots=True)
class CostValueSpec:
    """One cost head's return semantics and explicit loss identity."""

    name: str
    family: CostFamily
    gamma: float
    gae_lambda: float
    value_loss_coefficient: float = 1.0
    auxiliary_event_loss_coefficient: float = 0.0
    objective_altering_discount: bool = False

    def __post_init__(self) -> None:
        if self.name not in CONSTRAINT_COST_NAMES:
            raise ValueError(f"unknown constraint cost: {self.name}")
        if not isinstance(self.family, CostFamily):
            raise ValueError("family must be a CostFamily")
        expected = _expected_family(self.name)
        if self.family is not expected:
            raise ValueError(
                f"cost family mismatch for {self.name}: expected {expected.value}"
            )
        for field_name in (
            "gamma",
            "gae_lambda",
            "value_loss_coefficient",
            "auxiliary_event_loss_coefficient",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("value_loss_coefficient must be non-negative")
        if self.auxiliary_event_loss_coefficient < 0.0:
            raise ValueError("auxiliary_event_loss_coefficient must be non-negative")
        if self.family is CostFamily.CONTINUOUS and (
            self.auxiliary_event_loss_coefficient != 0.0
        ):
            raise ValueError(
                "auxiliary_event_loss_coefficient is valid only for event costs"
            )
        if (
            self.family is CostFamily.EVENT
            and self.gamma != 1.0
            and not self.objective_altering_discount
        ):
            raise ValueError(
                "discounted event cost is objective-altering and must be explicit"
            )
        if not isinstance(self.objective_altering_discount, bool):
            raise ValueError("objective_altering_discount must be a boolean")

    def digest_payload(self) -> dict[str, object]:
        return {
            "auxiliary_event_loss_coefficient": (self.auxiliary_event_loss_coefficient),
            "family": self.family.value,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "name": self.name,
            "objective_altering_discount": self.objective_altering_discount,
            "value_loss_coefficient": self.value_loss_coefficient,
        }


@dataclass(frozen=True, slots=True)
class CostLearningSchema:
    """Ordered Cost Critic schema included in experiment and checkpoint identity."""

    specs: tuple[CostValueSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("cost learning schema must not be empty")
        names = tuple(spec.name for spec in specs)
        if len(set(names)) != len(names):
            raise ValueError("cost learning schema contains duplicate cost names")
        unknown = tuple(name for name in names if name not in CONSTRAINT_COST_NAMES)
        if unknown:
            raise ValueError(f"unknown constraint cost: {unknown[0]}")
        canonical_indices = tuple(CONSTRAINT_COST_NAMES.index(name) for name in names)
        if canonical_indices != tuple(sorted(canonical_indices)):
            raise ValueError("cost learning schema must preserve canonical order")
        object.__setattr__(self, "specs", specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def continuous_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in self.specs if spec.family is CostFamily.CONTINUOUS
        )

    @property
    def event_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in self.specs if spec.family is CostFamily.EVENT
        )

    def __getitem__(self, name: str) -> CostValueSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def digest_payload(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "specs": [spec.digest_payload() for spec in self.specs],
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


def canonical_cost_learning_schema(
    *,
    continuous_gae_lambda: float = 0.95,
    event_gae_lambda: float = 0.95,
    value_loss_coefficient: float = 1.0,
    auxiliary_event_loss_coefficient: float = 0.0,
) -> CostLearningSchema:
    """Return the maintained seven-head schema in environment contract order."""

    return CostLearningSchema(
        tuple(
            CostValueSpec(
                name=name,
                family=_expected_family(name),
                gamma=1.0,
                gae_lambda=(
                    event_gae_lambda
                    if name in _EVENT_COST_NAMES
                    else continuous_gae_lambda
                ),
                value_loss_coefficient=value_loss_coefficient,
                auxiliary_event_loss_coefficient=(
                    auxiliary_event_loss_coefficient
                    if name in _EVENT_COST_NAMES
                    else 0.0
                ),
            )
            for name in CONSTRAINT_COST_NAMES
        )
    )


__all__ = [
    "CostFamily",
    "CostLearningSchema",
    "CostValueSpec",
    "canonical_cost_learning_schema",
]
