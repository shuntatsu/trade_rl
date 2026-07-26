from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import (
    ConstraintCostRequest,
    ConstraintCostVector,
    calculate_constraint_costs,
)


def test_constraint_cost_calculation_binds_transition_elapsed_hours() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.asarray([0.0], dtype=np.float64),
            max_gross=1.0,
            decision_hours=0.75,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=0.0,
            initial_capital=100.0,
            previous_equity=100.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            interval_funding=0.0,
            interval_borrow_cost=0.0,
            termination_reason=None,
            emergency_deleverage=False,
            liquidation_terminal=False,
            liquidation_complete=True,
        )
    )

    assert costs.transition_elapsed_hours == pytest.approx(0.75)
    assert costs.as_dict()["transition_elapsed_hours"] == pytest.approx(0.75)
    assert "transition_elapsed_hours" not in costs.constraint_dict()


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("nan"), float("inf")])
def test_constraint_cost_vector_rejects_invalid_elapsed_metadata(
    elapsed: float,
) -> None:
    with pytest.raises(ValueError, match="transition_elapsed_hours"):
        ConstraintCostVector(
            drawdown_excess=0.0,
            drawdown_stop_event=0.0,
            margin_deficit_fraction=0.0,
            forced_liquidation_event=0.0,
            gross_exposure_request_excess=0.0,
            daily_turnover=0.0,
            execution_cost_fraction=0.0,
            funding_credit_fraction=0.0,
            transition_elapsed_hours=elapsed,
        )
