"""Pure settled timing: actor transitions and physical closing bars are distinct.

This declaration executes no account, verifies no source and authorizes no fit.
GAE weights actor transitions; only the final reward spans the native reserve.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest


def _finite(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite native non-boolean number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True, kw_only=True)
class SettledAllocationClockContract:
    """One fixed actor prefix followed by a nonempty native closing reserve.

    Intervals describe regular timestamp spacing, not calendar carry duration.
    The future runtime must bind calendar kind and effective execution bar hours
    separately. A rollout boundary neither consumes reserve nor resets a book.
    """

    actor_steps: int
    reserve_steps: int
    execution_interval_seconds: int
    rollout_steps: int
    gamma: float
    gae_lambda: float

    def __post_init__(self) -> None:
        for field, minimum in (
            ("actor_steps", 1),
            ("reserve_steps", 1),
            ("execution_interval_seconds", 1),
            ("rollout_steps", 2),
        ):
            value = getattr(self, field)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field} must be a native integer >= {minimum}")
        gamma = _finite(self.gamma, "gamma")
        if gamma != 1.0:
            raise ValueError("settled fixed-capital reward requires gamma exactly 1")
        gae = _finite(self.gae_lambda, "gae_lambda")
        if not 0.0 <= gae <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae)

    @property
    def physical_horizon_steps(self) -> int:
        return self.actor_steps + self.reserve_steps

    def _actor_step(self, value: int) -> int:
        if type(value) is not int or not 0 <= value < self.actor_steps:
            raise ValueError("actor step must belong to the declared policy prefix")
        return value

    def processing_span(self, actor_step: int) -> tuple[int, int]:
        """Return half-open physical offsets attributed to this actor reward."""
        index = self._actor_step(actor_step)
        stop = (
            self.physical_horizon_steps if index == self.actor_steps - 1 else index + 1
        )
        return index, stop

    def transition_duration_seconds(self, actor_step: int) -> int:
        """Physical timestamp duration; GAE still counts one actor transition."""
        start, stop = self.processing_span(actor_step)
        return (stop - start) * self.execution_interval_seconds

    def remaining_horizon_fraction(self, actor_step: int) -> float:
        """Full physical horizon denominator for actor observations/order offsets."""
        index = self._actor_step(actor_step)
        return (self.physical_horizon_steps - index) / self.physical_horizon_steps

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_settled_clock_v1",
            **asdict(self),
            "physical_horizon_steps": self.physical_horizon_steps,
            "policy_horizon_seconds": self.actor_steps
            * self.execution_interval_seconds,
            "economic_horizon_seconds": self.physical_horizon_steps
            * self.execution_interval_seconds,
            "final_reward_interval_seconds": (self.reserve_steps + 1)
            * self.execution_interval_seconds,
            "reward_schema": "equity_delta_v1",
            "reward_timing": "last_actor_includes_all_native_reserved_bars_v1",
            "gae_clock": "actor_transition",
            "reserve_execution": "zero_target_through_unchanged_hard_risk",
            "reserve_coverage": "all_declared_bars_even_if_flat",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> SettledAllocationClockContract:
        """Admit only the exact resolved native JSON declaration, not a legacy clock."""
        if type(value) is not dict:
            raise ValueError("settled clock payload must be a native JSON mapping")
        data: dict[str, Any] = value
        if any(type(key) is not str for key in data) or any(
            type(item) not in (str, int, float) for item in data.values()
        ):
            raise ValueError("settled clock payload must contain native JSON values")
        try:
            clock = cls(
                actor_steps=data["actor_steps"],
                reserve_steps=data["reserve_steps"],
                execution_interval_seconds=data["execution_interval_seconds"],
                rollout_steps=data["rollout_steps"],
                gamma=data["gamma"],
                gae_lambda=data["gae_lambda"],
            )
            if canonical_json_bytes(data) != canonical_json_bytes(clock.payload()):
                raise ValueError("settled clock differs from its closed declaration")
            return clock
        except (KeyError, TypeError, OverflowError, RecursionError) as error:
            raise ValueError("invalid settled allocation clock payload") from error


__all__ = ["SettledAllocationClockContract"]
