"""Bind a new objective's evaluation period to its declared financial clock."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.objectives.clock import FinancialClockContract
from trade_rl.evaluation.objectives.contract import ObjectiveContract


@dataclass(frozen=True, slots=True)
class BoundObjectiveClock:
    """Bind one finite evaluation window, not a fit span or holding limit.

    No profile or runtime verification is implied.
    """

    objective: ObjectiveContract
    clock: FinancialClockContract

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveContract):
            raise ValueError("objective must be an ObjectiveContract")
        if not isinstance(self.clock, FinancialClockContract):
            raise ValueError("clock must be a FinancialClockContract")
        elapsed = (
            self.objective.evaluation_stop_exclusive - self.objective.evaluation_start
        )
        seconds = elapsed.days * 86400 + elapsed.seconds
        if elapsed.microseconds or seconds != self.clock.economic_horizon_seconds:
            raise ValueError(
                "evaluation period must equal the declared economic horizon"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "bound_objective_clock_v1",
            "objective_contract_digest": self.objective.digest,
            "financial_clock_digest": self.clock.digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())
