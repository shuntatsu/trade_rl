from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from tools.market_b4_oracle.allocation_choice import (
    AllocationAssumptions,
    AllocationChoice,
    choose_after_cost_weight,
)
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)

_TIME = np.datetime64("2026-01-01T00:00:00", "ns")
_TURNOVERS = (None, 0.0, 0.3, 2.0)
_RISK_AVERSIONS = (0.0, 1.0, 10.0)


def _production_choice(
    assumptions: AllocationAssumptions,
    *,
    current_weight: float,
    risk_aversion: float,
    max_turnover: float | None,
) -> tuple[float, float]:
    production_inputs = AllocationInputs(
        symbol="MANUFACTURED",
        decision_time=_TIME,
        available_at=_TIME,
        horizon_end=_TIME + np.timedelta64(24, "h"),
        source_identity="allocation-choice-differential-only",
        expected_simple_return=assumptions.expected_simple_return,
        return_variance=assumptions.return_variance,
        buy_cost=assumptions.buy_cost,
        sell_cost=assumptions.sell_cost,
        exit_cost=assumptions.exit_cost,
        funding_return=assumptions.funding_return,
        borrow_return=assumptions.borrow_return,
        cash_return=assumptions.cash_return,
    )
    context = AllocationContext(
        account_id="manufactured-differential-only",
        symbol="MANUFACTURED",
        decision_time=_TIME,
        state_digest="a" * 64,
        current_weight=current_weight,
        quantity="0",
        cash=100.0,
        equity=100.0,
        pending_remaining="0",
    )
    proposal = AfterCostTargetAllocator(
        lower_weight=-1.0,
        upper_weight=1.0,
        max_turnover=max_turnover,
        risk_aversion=risk_aversion,
    ).propose(production_inputs, context)
    return proposal.target_weight, proposal.objective_value


def _manufactured_case(
    index: int,
) -> tuple[AllocationAssumptions, float, float]:
    rng = random.Random(0xB4A110C + index)
    current_values = (-1.5, 1.5, -1.0, 1.0)
    current_weight = (
        current_values[index] if index < len(current_values) else rng.uniform(-1.5, 1.5)
    )
    assumptions = AllocationAssumptions(
        expected_simple_return=rng.choice(
            (-1.0, -0.2, -0.05, -0.001, 0.0, 0.003, 0.04, 0.2)
        ),
        return_variance=rng.choice((0.0, 0.0, 1e-4, 0.01, 0.2)),
        buy_cost=rng.choice((0.0, 0.0, 0.0005, 0.01, 0.05)),
        sell_cost=rng.choice((0.0, 0.0, 0.001, 0.015, 0.04)),
        exit_cost=rng.choice((0.0, 0.0, 0.001, 0.02)),
        funding_return=rng.choice((-0.03, -0.005, 0.0, 0.0, 0.025)),
        borrow_return=rng.choice((0.0, 0.0, 0.002, 0.03)),
        cash_return=rng.choice((-0.02, -0.003, 0.0, 0.0, 0.015)),
    )
    if index % 8 == 0:
        assumptions = AllocationAssumptions(
            expected_simple_return=assumptions.expected_simple_return,
            return_variance=assumptions.return_variance,
        )
    return assumptions, current_weight, _RISK_AVERSIONS[index % len(_RISK_AVERSIONS)]


@pytest.mark.parametrize("case_index", range(64))
@pytest.mark.parametrize("turnover", _TURNOVERS)
def test_256_deterministic_manufactured_cases_match_production_exactly(
    case_index: int, turnover: float | None
) -> None:
    assumptions, current_weight, risk_aversion = _manufactured_case(case_index)
    lower = max(-1.0, current_weight - turnover) if turnover is not None else -1.0
    upper = min(1.0, current_weight + turnover) if turnover is not None else 1.0
    case = (
        f"index={case_index}, turnover={turnover}, current={current_weight}, "
        f"assumptions={assumptions}, risk_aversion={risk_aversion}"
    )

    if lower > upper:
        with pytest.raises(ValueError):
            choose_after_cost_weight(
                assumptions,
                current_weight=current_weight,
                lower_weight=-1.0,
                upper_weight=1.0,
                risk_aversion=risk_aversion,
                max_turnover=turnover,
            )
        with pytest.raises(ValueError):
            _production_choice(
                assumptions,
                current_weight=current_weight,
                risk_aversion=risk_aversion,
                max_turnover=turnover,
            )
        return

    independent = choose_after_cost_weight(
        assumptions,
        current_weight=current_weight,
        lower_weight=-1.0,
        upper_weight=1.0,
        risk_aversion=risk_aversion,
        max_turnover=turnover,
    )
    production = _production_choice(
        assumptions,
        current_weight=current_weight,
        risk_aversion=risk_aversion,
        max_turnover=turnover,
    )
    assert (independent.target_weight, independent.objective_value) == production, case


def _fraction_utility(
    assumptions: AllocationAssumptions,
    *,
    weight: float,
    current_weight: float,
    risk_aversion: float,
) -> Fraction:
    exact = {
        name: Fraction(float(getattr(assumptions, name)))
        for name in (
            "expected_simple_return",
            "return_variance",
            "buy_cost",
            "sell_cost",
            "exit_cost",
            "funding_return",
            "borrow_return",
            "cash_return",
        )
    }
    w = Fraction(float(weight))
    held = Fraction(float(current_weight))
    return (
        exact["cash_return"] * (1 - w)
        + (exact["expected_simple_return"] - exact["funding_return"]) * w
        - Fraction(float(risk_aversion)) * exact["return_variance"] * w**2
        - exact["buy_cost"] * max(w - held, Fraction(0))
        - exact["sell_cost"] * max(held - w, Fraction(0))
        - exact["exit_cost"] * abs(w)
        - exact["borrow_return"] * max(-w, Fraction(0))
    )


def test_fraction_utility_checks_hand_calculated_candidate_grid_lower_bound() -> None:
    assumptions = AllocationAssumptions(
        expected_simple_return=0.25,
        return_variance=0.5,
        buy_cost=0.03125,
        sell_cost=0.0625,
        exit_cost=0.015625,
        funding_return=0.015625,
        borrow_return=0.125,
        cash_return=0.03125,
    )
    current = 0.125
    choice = choose_after_cost_weight(
        assumptions,
        current_weight=current,
        lower_weight=-1.0,
        upper_weight=1.0,
        risk_aversion=1.0,
    )
    assert isinstance(choice, AllocationChoice)
    assert choice.target_weight == 0.15625

    exact_choice = _fraction_utility(
        assumptions,
        weight=choice.target_weight,
        current_weight=current,
        risk_aversion=1.0,
    )
    candidate_grid = (-1.0, -0.5, 0.0, current, 0.15625, 0.5, 1.0)
    grid_values = tuple(
        _fraction_utility(
            assumptions,
            weight=weight,
            current_weight=current,
            risk_aversion=1.0,
        )
        for weight in candidate_grid
    )
    assert exact_choice == Fraction(97, 2048)
    assert exact_choice >= max(grid_values)
    assert choice.objective_value == float(exact_choice)


@pytest.mark.parametrize(
    ("expected_return", "expected_target"),
    [
        (0.125, 0.0),
        (math.nextafter(0.125, 0.0), 0.0),
        (math.nextafter(0.125, math.inf), 1.0),
    ],
)
def test_fraction_utility_and_production_agree_at_near_ulp_cost_breakpoint(
    expected_return: float, expected_target: float
) -> None:
    assumptions = AllocationAssumptions(
        expected_simple_return=expected_return,
        buy_cost=0.125,
    )
    independent = choose_after_cost_weight(
        assumptions,
        current_weight=0.0,
        lower_weight=-1.0,
        upper_weight=1.0,
    )
    production = _production_choice(
        assumptions,
        current_weight=0.0,
        risk_aversion=0.0,
        max_turnover=None,
    )
    assert independent.target_weight == expected_target
    assert (independent.target_weight, independent.objective_value) == production
    exact = _fraction_utility(
        assumptions,
        weight=expected_target,
        current_weight=0.0,
        risk_aversion=0.0,
    )
    assert independent.objective_value == float(exact)


@pytest.mark.parametrize(
    ("assumptions", "risk_aversion"),
    [
        (
            AllocationAssumptions(
                expected_simple_return=0.0,
                return_variance=1e308,
            ),
            1e308,
        ),
        (
            AllocationAssumptions(
                expected_simple_return=1e308,
                funding_return=-1e308,
            ),
            0.0,
        ),
    ],
)
def test_fraction_comparison_products_that_cannot_be_returned_as_floats_fail_closed(
    assumptions: AllocationAssumptions, risk_aversion: float
) -> None:
    with pytest.raises(ValueError):
        choose_after_cost_weight(
            assumptions,
            current_weight=0.0,
            lower_weight=-1.0,
            upper_weight=1.0,
            risk_aversion=risk_aversion,
        )
    with pytest.raises(ValueError):
        _production_choice(
            assumptions,
            current_weight=0.0,
            risk_aversion=risk_aversion,
            max_turnover=None,
        )


def test_scope_is_arithmetic_only() -> None:
    choice = choose_after_cost_weight(
        AllocationAssumptions(expected_simple_return=0.1),
        current_weight=0.0,
    )
    assert choice.scope == "AFTER_COST_ALLOCATION_ARITHMETIC_ONLY"
