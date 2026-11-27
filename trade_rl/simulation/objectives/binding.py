"""Bind an objective evaluation period to its declared financial clock."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.objectives.clock import (
    CalendarMonthFinancialClockContract,
    FinancialClockContract,
    _calendar_month_endpoint,
)
from trade_rl.simulation.objectives.contract import ObjectiveContract


@dataclass(frozen=True, slots=True)
class BoundObjectiveClock:
    """Bind one finite evaluation window, not a fit span or holding limit.

    No profile or runtime verification is implied.
    """

    objective: ObjectiveContract
    clock: FinancialClockContract | CalendarMonthFinancialClockContract

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveContract):
            raise ValueError("objective must be an ObjectiveContract")
        if not isinstance(
            self.clock, (FinancialClockContract, CalendarMonthFinancialClockContract)
        ):
            raise ValueError(
                "clock must be a FinancialClockContract or calendar month clock"
            )
        elapsed = (
            self.objective.evaluation_stop_exclusive - self.objective.evaluation_start
        )
        seconds = elapsed.days * 86400 + elapsed.seconds
        if elapsed.microseconds or seconds != self.clock.economic_horizon_seconds:
            raise ValueError(
                "evaluation period must equal the declared economic horizon"
            )

        if isinstance(self.clock, CalendarMonthFinancialClockContract) and (
            _calendar_month_endpoint(
                self.objective.evaluation_start, self.clock.economic_horizon_months
            )
            != self.objective.evaluation_stop_exclusive
        ):
            raise ValueError("calendar month horizon differs from its objective window")

    def payload(self) -> dict[str, object]:
        return {
            "schema": (
                "bound_calendar_month_objective_clock_v1"
                if isinstance(self.clock, CalendarMonthFinancialClockContract)
                else "bound_objective_clock_v1"
            ),
            "objective_contract_digest": self.objective.digest,
            "financial_clock_digest": self.clock.digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())
