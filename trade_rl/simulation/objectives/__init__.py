"""Pure declarations for financial objectives and their regular clocks."""

from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import (
    CapitalContract,
    ObjectiveContract,
    net_equity_increment,
)

__all__ = [
    "CapitalContract",
    "FinancialClockContract",
    "ObjectiveContract",
    "BoundObjectiveClock",
    "net_equity_increment",
]
