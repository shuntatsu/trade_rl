"""Detached allocation decisions and explicit four-action target mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.allocation import AllocationProposal


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True, slots=True)
class AllocationActionContract:
    mode: Literal["direct", "residual"] = "direct"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in ("direct", "residual"):
            raise ValueError("action mode must be direct or residual")
        scale = _number(self.scale, "scale")
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be within (0, 1]")
        object.__setattr__(self, "scale", scale)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_action_v1",
            "mode": self.mode,
            "scale": self.scale,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    baseline: AllocationProposal
    action_contract: AllocationActionContract
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    max_drawdown: float
    initial_capital: float
    remaining_steps: int
    pending_gross: float
    pending_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, AllocationProposal) or not isinstance(
            self.action_contract, AllocationActionContract
        ):
            raise ValueError(
                "decision requires an allocation baseline and action contract"
            )
        if (
            self.baseline.allocator.propose(self.baseline.inputs, self.baseline.context)
            != self.baseline
        ):
            raise ValueError("baseline must retain its complete optimizer identity")
        if (
            not isinstance(self.feature_names, tuple)
            or not self.feature_names
            or any(not isinstance(n, str) or not n.strip() for n in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
            or not isinstance(self.feature_values, tuple)
            or len(self.feature_values) != len(self.feature_names)
        ):
            raise ValueError(
                "features must be paired immutable values and unique nonempty names"
            )
        object.__setattr__(
            self,
            "feature_values",
            tuple(_number(v, "feature value") for v in self.feature_values),
        )
        for name in ("max_drawdown", "initial_capital", "pending_gross"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be within [0, 1]")
        if self.initial_capital <= 0.0 or self.pending_gross < 0.0:
            raise ValueError("capital must be positive and pending gross nonnegative")
        for name, minimum in (("remaining_steps", 1), ("pending_count", 0)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))

    @property
    def feasible_bounds(self) -> tuple[float, float]:
        allocator, current = (
            self.baseline.allocator,
            self.baseline.context.current_weight,
        )
        lower, upper = allocator.lower_weight, allocator.upper_weight
        if allocator.max_turnover is not None:
            lower = max(lower, current - allocator.max_turnover)
            upper = min(upper, current + allocator.max_turnover)
        return float(lower), float(upper)

    @property
    def decision_digest(self) -> str:
        return content_digest(
            {
                "schema": "allocation_decision_v1",
                "baseline": self.baseline.decision_digest,
                "action_contract": self.action_contract.payload(),
                "feasible_bounds": self.feasible_bounds,
                "feature_names": self.feature_names,
                "feature_values": self.feature_values,
                "max_drawdown": self.max_drawdown,
                "initial_capital": self.initial_capital,
                "remaining_steps": self.remaining_steps,
                "pending_gross": self.pending_gross,
                "pending_count": self.pending_count,
            }
        )

    def _action_fields(self, action: object) -> tuple[int, str, float, float, bool]:
        values = np.asarray(action).reshape(-1)
        if values.size != 1:
            raise ValueError("action must contain exactly one integer in {0, 1, 2, 3}")
        code = _integer(values[0], "action", 0)
        if code > 3:
            raise ValueError("action must be within {0, 1, 2, 3}")
        current, baseline = (
            self.baseline.context.current_weight,
            self.baseline.target_weight,
        )
        lower, upper = self.feasible_bounds
        if code == 0:
            # HOLD bypasses alpha/turnover bounds; final hard risk still applies.
            return code, "quantity_hold", current, current, True
        if self.action_contract.mode == "direct":
            raw = (-self.action_contract.scale, 0.0, self.action_contract.scale)[
                code - 1
            ]
            target = min(upper, max(lower, raw))
            label = ("negative", "flat", "positive")[code - 1]
        else:
            label = ("negative", "baseline", "positive")[code - 1]
            raw = baseline
            if code != 2:
                endpoint = lower if code == 1 else upper
                raw = (
                    endpoint
                    if self.action_contract.scale == 1.0
                    else baseline + self.action_contract.scale * (endpoint - baseline)
                )
            target = raw
        return code, label, float(raw), float(target), target == current

    def propose(self, action: object) -> AllocationActionProposal:
        return AllocationActionProposal(self, *self._action_fields(action))


@dataclass(frozen=True, slots=True)
class AllocationActionProposal:
    decision: AllocationDecision
    raw_action: int
    action: str
    raw_target_weight: float
    target_weight: float
    is_hold: bool

    def __post_init__(self) -> None:
        if not isinstance(self.decision, AllocationDecision):
            raise ValueError("proposal requires an allocation decision")
        _number(self.raw_target_weight, "raw_target_weight")
        _number(self.target_weight, "target_weight")
        if not isinstance(self.is_hold, (bool, np.bool_)):
            raise ValueError("is_hold must be boolean")
        expected = self.decision._action_fields(self.raw_action)
        if (
            self.raw_action,
            self.action,
            self.raw_target_weight,
            self.target_weight,
            self.is_hold,
        ) != expected:
            raise ValueError("proposal must retain its declared action mapping")
        for name, value in zip(
            ("raw_action", "action", "raw_target_weight", "target_weight", "is_hold"),
            expected,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema": "allocation_action_proposal_v1",
                "decision": self.decision.decision_digest,
                "raw_action": self.raw_action,
                "action": self.action,
                "raw_target_weight": self.raw_target_weight,
                "target_weight": self.target_weight,
                "is_hold": self.is_hold,
            }
        )


__all__ = ["AllocationActionContract", "AllocationDecision", "AllocationActionProposal"]
