"""Declarations for new net-profit studies; no execution authorization."""

from trade_rl.evaluation.objectives.binding import BoundObjectiveClock
from trade_rl.evaluation.objectives.clock import FinancialClockContract
from trade_rl.evaluation.objectives.contract import (
    CapitalContract,
    ObjectiveContract,
    net_equity_increment,
)

__all__ = [
    "BoundObjectiveClock",
    "CapitalContract",
    "FinancialClockContract",
    "ObjectiveContract",
    "net_equity_increment",
]
