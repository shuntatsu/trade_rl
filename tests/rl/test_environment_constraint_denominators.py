from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import (
    ConstraintCostRequest,
    calculate_constraint_costs,
)


def test_margin_and_execution_costs_use_their_normative_denominators() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.0]),
            max_gross=1.0,
            decision_hours=1.0,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=250.0,
            initial_capital=50_000.0,
            previous_equity=100_000.0,
            filled_turnover=0.0,
            interval_cost=20.0,
            interval_funding=5.0,
            interval_borrow_cost=0.0,
            termination_reason=None,
            emergency_deleverage=False,
            liquidation_terminal=False,
            liquidation_complete=True,
        )
    )

    assert costs.margin_deficit_fraction == pytest.approx(0.005)
    assert costs.execution_cost_fraction == pytest.approx(0.0002)
    assert costs.funding_credit_fraction == pytest.approx(0.00005)


def test_constraint_dictionary_excludes_funding_credit_diagnostic() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.0]),
            max_gross=1.0,
            decision_hours=1.0,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=0.0,
            initial_capital=100.0,
            previous_equity=100.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            interval_funding=1.0,
            interval_borrow_cost=0.0,
            termination_reason=None,
            emergency_deleverage=False,
            liquidation_terminal=False,
            liquidation_complete=True,
        )
    )

    assert tuple(costs.constraint_dict()) == (
        "drawdown_excess",
        "drawdown_stop_event",
        "margin_deficit_fraction",
        "forced_liquidation_event",
        "gross_exposure_request_excess",
        "daily_turnover",
        "execution_cost_fraction",
    )
    assert "funding_credit_fraction" not in costs.constraint_dict()
    assert costs.funding_credit_fraction == pytest.approx(0.01)
