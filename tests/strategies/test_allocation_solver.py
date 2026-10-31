from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_rl.strategies.allocation import solver as solver_module
from trade_rl.strategies.allocation.cost_aware import ewma_covariance
from trade_rl.strategies.allocation.solver import (
    AllocationSolution,
    AllocationSolverError,
    allocation_objective,
    kkt_residual,
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


def _grid_optimum(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    costs: np.ndarray,
    gamma: float,
    w_max: float,
    gross_max: float,
    points: int = 801,
) -> float:
    axis = np.linspace(-w_max, w_max, points)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    weights = np.stack((first, second), axis=-1)
    feasible = np.abs(first) + np.abs(second) <= gross_max + 1e-12
    risk = 0.5 * gamma * np.einsum("...i,ij,...j->...", weights, sigma, weights)
    cost = np.sum(costs * np.abs(weights - w0), axis=-1)
    utility = weights @ mu - risk - cost
    return float(np.max(np.where(feasible, utility, -np.inf)))


def test_grid_search_with_binding_gross_and_weight_limits() -> None:
    """Strong signals: asset 0 sits on w_max and the gross limit binds as well."""

    mu = np.array([0.08, 0.05])
    w0 = np.array([0.0, 0.1])
    costs = np.array([0.001, 0.002])
    solution = solve_cost_aware_allocation(
        expected_returns=mu,
        covariance=SIGMA,
        current_weights=w0,
        costs=costs,
        gamma=2.0,
        w_max=0.4,
        gross_max=0.55,
    )
    assert solution.weights[0] == pytest.approx(0.4, abs=1e-9)  # w_max binds
    assert np.abs(solution.weights).sum() == pytest.approx(0.55, abs=1e-8)
    assert solution.gross_multiplier > 0.0
    best = _grid_optimum(
        mu=mu, sigma=SIGMA, w0=w0, costs=costs, gamma=2.0, w_max=0.4, gross_max=0.55
    )
    assert solution.objective >= best - 1e-9
    assert solution.objective <= best + 1e-4  # grid resolution bound


def test_grid_search_with_binding_gross_costs_and_a_short() -> None:
    mu = np.array([0.04, -0.06])
    w0 = np.array([0.2, -0.1])
    costs = np.array([0.003, 0.0005])
    solution = solve_cost_aware_allocation(
        expected_returns=mu,
        covariance=SIGMA,
        current_weights=w0,
        costs=costs,
        gamma=1.0,
        w_max=0.45,
        gross_max=0.6,
    )
    assert np.abs(solution.weights).sum() == pytest.approx(0.6, abs=1e-8)
    best = _grid_optimum(
        mu=mu, sigma=SIGMA, w0=w0, costs=costs, gamma=1.0, w_max=0.45, gross_max=0.6
    )
    assert solution.objective >= best - 1e-9
    assert solution.objective <= best + 1e-4  # grid resolution bound


def _realistic_instance(
    rng: np.random.Generator, *, ill_conditioned: bool
) -> dict[str, Any]:
    """Five crypto-like assets: daily variances 1e-3..4e-3, costs ~7e-4."""

    n = 5
    vol = np.sqrt(rng.uniform(1e-3, 4e-3, n))
    beta = rng.uniform(0.6, 1.4, n)
    idiosyncratic = rng.uniform(0.2, 0.9, n)
    days = int(rng.integers(40, 400))
    factor = rng.normal(0.0, 0.03, days)
    returns = np.outer(factor, beta) + rng.normal(0.0, 1.0, (days, n)) * (
        idiosyncratic * 0.03
    )
    returns = returns * (vol / returns.std(axis=0))
    covariance = ewma_covariance(
        returns,
        halflife_periods=float(rng.uniform(20.0, 120.0)),
        shrinkage=float(rng.choice([0.0, 0.1, 0.2, 0.5])),
        variance_floor=1e-8,
    )
    if ill_conditioned:
        # Condition numbers of 300..1500, the regime in which plain coordinate
        # descent needed tens of thousands of sweeps.
        basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
        top = float(rng.uniform(2e-3, 4e-3))
        eigenvalues = np.exp(rng.uniform(np.log(top / 1500.0), np.log(top), n))
        eigenvalues[0] = top
        eigenvalues[-1] = top / float(rng.uniform(300.0, 1500.0))
        covariance = (basis * eigenvalues) @ basis.T
        covariance = 0.5 * (covariance + covariance.T)
    w_max = 0.5
    gross_max = 0.95
    current = rng.uniform(-w_max, w_max, n) * (rng.random(n) < 0.7)
    gross = float(np.abs(current).sum())
    if gross > gross_max:
        current = current * (gross_max / gross * 0.99)
    return {
        "expected_returns": rng.normal(0.0, 0.004, n),
        "covariance": covariance,
        "current_weights": current,
        "costs": np.full(n, 7e-4),
        "gamma": 5.0,
        "w_max": w_max,
        "gross_max": gross_max,
        "long_only": bool(rng.random() < 0.3),
    }


def _bounds(instance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = instance["expected_returns"].shape[0]
    low = 0.0 if instance["long_only"] else -instance["w_max"]
    return np.full(n, low), np.full(n, instance["w_max"])


def _assert_feasible_and_certified(
    instance: dict[str, Any], solution: AllocationSolution
) -> None:
    weights = solution.weights
    low, high = _bounds(instance)
    assert np.all(weights >= low - 1e-12)
    assert np.all(weights <= high + 1e-12)
    assert np.abs(weights).sum() <= instance["gross_max"] + 1e-9
    scale = max(
        float(np.max(np.abs(instance["expected_returns"]))),
        float(np.max(instance["costs"])),
        instance["gamma"]
        * float(np.max(np.diag(instance["covariance"])))
        * instance["w_max"],
    )
    residual = kkt_residual(
        weights,
        expected_returns=instance["expected_returns"],
        covariance=instance["covariance"],
        current_weights=instance["current_weights"],
        costs=instance["costs"],
        gamma=instance["gamma"],
        low=low,
        high=high,
        multiplier=solution.gross_multiplier,
    )
    assert residual / scale <= 1e-8
    assert solution.kkt_residual <= 1e-8
    slack = instance["gross_max"] - float(np.abs(weights).sum())
    assert solution.gross_multiplier * max(slack, 0.0) <= 1e-8 * scale


def _has_no_feasible_local_improvement(
    instance: dict[str, Any], solution: AllocationSolution
) -> bool:
    """Independent of the KKT code: random feasible nearby points never win."""

    rng = np.random.default_rng(0)
    low, high = _bounds(instance)
    for _ in range(40):
        direction = rng.normal(size=solution.weights.shape)
        direction[rng.random(direction.shape) < 0.5] = 0.0
        for step in (1e-3, -1e-3, 1e-5):
            point = solution.weights + step * direction
            if (
                np.any(point < low)
                or np.any(point > high)
                or np.abs(point).sum() > instance["gross_max"]
            ):
                continue
            value = allocation_objective(
                point,
                expected_returns=instance["expected_returns"],
                covariance=instance["covariance"],
                current_weights=instance["current_weights"],
                costs=instance["costs"],
                gamma=instance["gamma"],
            )
            if value > solution.objective + 1e-11:
                return False
    return True


@pytest.mark.parametrize("ill_conditioned", [False, True])
def test_realistic_random_instances_always_converge_and_are_certified(
    ill_conditioned: bool,
) -> None:
    rng = np.random.default_rng(20261029 + int(ill_conditioned))
    for _ in range(500):
        instance = _realistic_instance(rng, ill_conditioned=ill_conditioned)
        solution = solve_cost_aware_allocation(**instance)
        _assert_feasible_and_certified(instance, solution)
        assert _has_no_feasible_local_improvement(instance, solution)


def test_kkt_certificate_rejects_non_optimal_points() -> None:
    instance = _realistic_instance(np.random.default_rng(3), ill_conditioned=False)
    solution = solve_cost_aware_allocation(**instance)
    low, high = _bounds(instance)
    common: dict[str, Any] = {
        "expected_returns": instance["expected_returns"],
        "covariance": instance["covariance"],
        "current_weights": instance["current_weights"],
        "costs": instance["costs"],
        "gamma": instance["gamma"],
        "low": low,
        "high": high,
        "multiplier": solution.gross_multiplier,
    }
    assert kkt_residual(solution.weights, **common) < 1e-9
    moved = solution.weights.copy()
    moved[0] += 0.01
    assert kkt_residual(moved, **common) > 1e-6


def test_failed_certificate_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong(**_: object) -> tuple[np.ndarray, int]:
        return np.full(2, 0.123), 1  # converged-looking but not optimal

    monkeypatch.setattr(solver_module, "_coordinate_descent", wrong)
    with pytest.raises(AllocationSolverError) as caught:
        _solve()
    assert caught.value.reason == "kkt_violation"


def test_kkt_tolerance_is_validated() -> None:
    with pytest.raises(ValueError):
        _solve(kkt_tol=0.0)
