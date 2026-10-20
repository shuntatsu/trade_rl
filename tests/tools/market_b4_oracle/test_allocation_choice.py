from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from tools.market_b4_oracle.allocation_choice import (
    AllocationAssumptions,
    choose_after_cost_weight,
)


def _inputs(**overrides: float) -> AllocationAssumptions:
    values = {"expected_simple_return": 0.0, **overrides}
    return AllocationAssumptions(**values)


def test_long_choice_accounts_for_buy_and_exit_costs() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.10, buy_cost=0.02, exit_cost=0.01),
        current_weight=0.0,
        lower_weight=-0.5,
        upper_weight=0.5,
    )

    assert choice.target_weight == 0.5
    assert choice.objective_value == pytest.approx(0.035)


def test_short_choice_accounts_for_sell_exit_and_borrow_costs() -> None:
    choice = choose_after_cost_weight(
        _inputs(
            expected_simple_return=-0.10,
            sell_cost=0.02,
            exit_cost=0.01,
            borrow_return=0.03,
        ),
        current_weight=0.0,
        lower_weight=-0.5,
        upper_weight=0.5,
    )

    assert choice.target_weight == -0.5
    assert choice.objective_value == pytest.approx(0.02)


def test_funding_return_sign_changes_the_preferred_side() -> None:
    short = choose_after_cost_weight(
        _inputs(funding_return=0.04),
        current_weight=0.0,
        lower_weight=-0.5,
        upper_weight=0.5,
    )
    long = choose_after_cost_weight(
        _inputs(funding_return=-0.04),
        current_weight=0.0,
        lower_weight=-0.5,
        upper_weight=0.5,
    )

    assert short.target_weight == -0.5
    assert short.objective_value == pytest.approx(0.02)
    assert long.target_weight == 0.5
    assert long.objective_value == pytest.approx(0.02)


def test_quadratic_risk_penalty_selects_interior_stationary_weight() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.2, return_variance=0.25),
        current_weight=0.0,
        lower_weight=-1.0,
        upper_weight=1.0,
        risk_aversion=2.0,
    )

    assert choice.target_weight == pytest.approx(0.2)
    assert choice.objective_value == pytest.approx(0.02)


def test_zero_variance_uses_the_linear_objective_boundary() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.03, return_variance=0.0),
        current_weight=0.0,
        lower_weight=-0.4,
        upper_weight=0.4,
        risk_aversion=3.0,
    )

    assert choice.target_weight == 0.4
    assert choice.objective_value == pytest.approx(0.012)


def test_asymmetric_trade_costs_make_current_weight_the_kink_optimum() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.01, buy_cost=0.02, sell_cost=0.01),
        current_weight=0.2,
        lower_weight=0.0,
        upper_weight=0.5,
    )

    assert choice.target_weight == 0.2
    assert choice.objective_value == pytest.approx(0.002)


def test_flat_objective_prefers_exact_hold() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.05, cash_return=0.05),
        current_weight=0.35,
        lower_weight=-0.5,
        upper_weight=0.8,
    )

    assert choice.target_weight == 0.35
    assert choice.objective_value == pytest.approx(0.05)


def test_exact_fraction_comparison_keeps_subnormal_positive_gain() -> None:
    tiny_positive = math.nextafter(0.0, 1.0)
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=tiny_positive),
        current_weight=0.0,
        lower_weight=-1.0,
        upper_weight=1.0,
    )

    assert choice.target_weight == 1.0
    assert choice.objective_value == tiny_positive


def test_turnover_bounds_clip_the_optimizer_feasible_interval() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=1.0),
        current_weight=0.2,
        lower_weight=-1.0,
        upper_weight=1.0,
        max_turnover=0.1,
    )

    assert choice.target_weight == pytest.approx(0.3)
    assert choice.objective_value == pytest.approx(0.3)


def test_empty_turnover_intersection_is_rejected() -> None:
    with pytest.raises(ValueError):
        choose_after_cost_weight(
            _inputs(expected_simple_return=0.1),
            current_weight=2.0,
            lower_weight=-1.0,
            upper_weight=1.0,
            max_turnover=0.1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_simple_return", math.nan),
        ("expected_simple_return", True),
        ("expected_simple_return", math.inf),
        ("expected_simple_return", -1.0001),
        ("return_variance", -0.01),
        ("buy_cost", -0.01),
        ("buy_cost", True),
        ("sell_cost", -0.01),
        ("exit_cost", -0.01),
        ("borrow_return", -0.01),
        ("cash_return", math.nan),
        ("funding_return", math.inf),
    ],
)
def test_assumptions_reject_invalid_finite_or_sign_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        AllocationAssumptions(**{"expected_simple_return": 0.0, field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_weight": math.nan},
        {"lower_weight": -10.1},
        {"upper_weight": 10.1},
        {"lower_weight": 0.5, "upper_weight": -0.5},
        {"risk_aversion": -0.1},
        {"risk_aversion": math.inf},
        {"max_turnover": -0.1},
        {"max_turnover": math.nan},
    ],
)
def test_choice_rejects_invalid_closed_numeric_parameters(
    kwargs: dict[str, float],
) -> None:
    arguments = {"current_weight": 0.0, **kwargs}
    with pytest.raises(ValueError):
        choose_after_cost_weight(_inputs(expected_simple_return=0.1), **arguments)


def test_objective_overflow_fails_closed() -> None:
    with pytest.raises(ValueError):
        choose_after_cost_weight(
            _inputs(expected_simple_return=1e308),
            current_weight=0.0,
            lower_weight=-10.0,
            upper_weight=10.0,
        )


def test_result_is_immutable() -> None:
    choice = choose_after_cost_weight(
        _inputs(expected_simple_return=0.1), current_weight=0.0
    )

    with pytest.raises(FrozenInstanceError):
        choice.target_weight = 0.25  # type: ignore[misc]
