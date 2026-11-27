"""Pure declarations for financial objectives and their explicit clocks."""

from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import (
    CalendarMonthFinancialClockContract,
    FinancialClockContract,
)
from trade_rl.simulation.objectives.contract import (
    CapitalContract,
    ObjectiveContract,
    net_equity_increment,
)

__all__ = [
    "CalendarMonthFinancialClockContract",
    "CapitalContract",
    "FinancialClockContract",
    "ObjectiveContract",
    "BoundObjectiveClock",
    "net_equity_increment",
]
