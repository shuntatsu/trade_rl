from __future__ import annotations

import numpy as np

from tests.simulation.test_critical_branch_coverage import market
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def test_slippage_executor_setup() -> None:
    executor = MarketExecutor(
        market(
            lot_size=np.full((5, 1), 0.25),
            borrow_available=np.zeros((5, 1), dtype=np.bool_),
        ),
        ExecutionCostConfig(
            slippage_std=0.01,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=2.0,
            random_seed=7,
        ),
    )
    assert executor.dataset.n_symbols == 1
