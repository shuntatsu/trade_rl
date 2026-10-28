from __future__ import annotations

import numpy as np
import pytest

from trade_rl.strategies.allocation.solver import (
    AllocationSolverError,
    allocation_objective,
    solve_cost_aware_allocation,
)

SIGMA = np.array([[0.04, 0.01], [0.01, 0.09]])


def _solve(**overrides: object):
    kwargs: dict[str, object] = {
        "expected_returns": np.array([0.02, 0.01]),
        "covariance": SIGMA,
        "current_weights": np.zeros(2),
        "costs": np.zeros(2),
        "gamma": 5.0,
        "w_max": 10.0,
        "gross_max": 100.0,
    }
    kwargs.update(overrides)
    return solve_cost_aware_allocation(**kwargs)  # type: ignore[arg-type]


def test_zero_cost_unconstrained_matches_closed_form() -> None:
    solution = _solve()
    expected = np.linalg.solve(5.0 * SIGMA, np.array([0.02, 0.01]))
    np.testing.assert_allclose(solution.weights, expected, atol=1e-8)
    assert solution.gross_multiplier == 0.0


def test_l1_cost_creates_exact_no_trade_region() -> None:
    # The marginal utility of the first unit of weight is below the cost.
    solution = _solve(
        expected_returns=np.array([0.001, -0.001]),
        costs=np.array([0.002, 0.002]),
        current_weights=np.zeros(2),
    )
    np.testing.assert_array_equal(solution.weights, np.zeros(2))
    anchored = _solve(
        expected_returns=np.array([0.0, 0.0]),
        costs=np.array([0.002, 0.002]),
        current_weights=np.array([0.3, -0.2]),
        gamma=0.001,  # risk gradient stays inside the cost band around w0
    )
    np.testing.assert_allclose(anchored.weights, [0.3, -0.2], atol=1e-12)


def test_larger_cost_never_increases_distance_from_current_weights() -> None:
    previous = np.inf
    for cost in (0.0, 0.0005, 0.002, 0.01):
        solution = _solve(costs=np.array([cost, cost]))
        distance = float(np.abs(solution.weights).sum())
        assert distance <= previous + 1e-12
        previous = distance


@pytest.mark.parametrize("long_only", [False, True])
def test_matches_grid_search_with_all_constraints(long_only: bool) -> None:
    mu = np.array([0.03, -0.02])
    w0 = np.array([0.1, 0.1])
    costs = np.array([0.004, 0.001])
    solution = solve_cost_aware_allocation(
        expected_returns=mu,
        covariance=SIGMA,
        current_weights=w0,
        costs=costs,
        gamma=3.0,
        w_max=0.6,
        gross_max=0.7,
        long_only=long_only,
    )
    axis = np.linspace(0.0 if long_only else -0.6, 0.6, 601)
    best = -np.inf
    for x in axis:
        for y in axis:
            if abs(x) + abs(y) > 0.7 + 1e-12:
                continue
            value = allocation_objective(
                np.array([x, y]),
                expected_returns=mu,
                covariance=SIGMA,
                current_weights=w0,
                costs=costs,
                gamma=3.0,
            )
            best = max(best, value)
    assert solution.objective >= best - 1e-9
    assert np.abs(solution.weights).sum() <= 0.7 + 1e-9
    assert np.all(np.abs(solution.weights) <= 0.6 + 1e-12)
    if long_only:
        assert np.all(solution.weights >= 0.0)


def test_gross_constraint_binds_and_is_never_violated() -> None:
    solution = _solve(
        expected_returns=np.array([0.2, 0.2, 0.2]),
        covariance=np.diag([0.01, 0.01, 0.01]),
        current_weights=np.zeros(3),
        costs=np.zeros(3),
        w_max=0.9,
        gross_max=1.0,
    )
    assert np.abs(solution.weights).sum() == pytest.approx(1.0, abs=1e-8)
    assert np.abs(solution.weights).sum() <= 1.0 + 1e-9
    assert solution.gross_multiplier > 0.0


def test_solver_is_deterministic() -> None:
    first = _solve(gross_max=0.2, w_max=0.5)
    second = _solve(gross_max=0.2, w_max=0.5)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_non_convergence_raises_instead_of_returning_weights() -> None:
    with pytest.raises(AllocationSolverError) as caught:
        _solve(
            covariance=np.array([[1.0, 0.99], [0.99, 1.0]]),
            expected_returns=np.array([1.0, -1.0]),
            max_iter=1,
            tol=1e-15,
            gamma=1.0,
        )
    assert caught.value.reason == "not_converged"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"expected_returns": np.array([np.nan, 0.0])}, "non_finite"),
        ({"current_weights": np.array([np.inf, 0.0])}, "non_finite"),
        ({"covariance": np.array([[0.04, np.nan], [np.nan, 0.09]])}, "non_finite"),
        (
            {"covariance": np.array([[0.04, 0.02], [0.0, 0.09]])},
            "asymmetric_covariance",
        ),
        ({"covariance": np.array([[0.0, 0.0], [0.0, 0.09]])}, "non_positive_variance"),
        (
            {"covariance": np.array([[0.01, 0.02], [0.02, 0.01]])},
            "covariance_not_positive_definite",
        ),
        ({"costs": np.array([-0.1, 0.0])}, "negative_cost"),
        ({"costs": np.array([0.0])}, "shape"),
    ],
)
def test_invalid_inputs_fail_closed(overrides: dict[str, object], reason: str) -> None:
    with pytest.raises(AllocationSolverError) as caught:
        _solve(**overrides)
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"gamma": 0.0},
        {"w_max": -1.0},
        {"gross_max": float("nan")},
        {"tol": 0.0},
        {"max_iter": 0},
        {"long_only": 1},
    ],
)
def test_invalid_parameters_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _solve(**overrides)
