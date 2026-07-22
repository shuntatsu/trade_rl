"""Portfolio execution and accounting simulation."""

from trade_rl.simulation import execution as _execution
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionResult
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor

# Direct imports of ``trade_rl.simulation.execution.MarketExecutor`` occur only
# after this package initializer completes. Replace the compatibility facade
# while retaining the original implementation as its base class.
setattr(_execution, "MarketExecutor", StatefulCompatibilityMarketExecutor)
MarketExecutor = StatefulCompatibilityMarketExecutor

__all__ = [
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionResult",
    "MarketExecutor",
]
