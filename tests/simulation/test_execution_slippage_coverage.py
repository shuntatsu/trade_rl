from __future__ import annotations

import numpy as np
import pytest

from tests.simulation.test_critical_branch_coverage import market
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def test_slippage_lot_rounding_and_unavailable_borrow_paths() -> None:
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
    assert executor._slippage_rates(0).size == 0
    assert np.isfinite(executor._slippage_rates(3)).all()
    assert executor._round_quantities(
        np.array([1.13]), index=1
    ).tolist() == pytest.approx([1.0])
    constrained = executor._constrain_borrow(
        np.array([-2.0]), current=np.array([-1.0]), index=1
    )
    assert constrained.tolist() == [-1.0]
