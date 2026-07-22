"""Portfolio execution and accounting simulation."""

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionResult
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor

MarketExecutor = StatefulCompatibilityMarketExecutor

__all__ = [
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionResult",
    "MarketExecutor",
]
