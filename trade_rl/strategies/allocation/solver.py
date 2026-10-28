"""Exact convex solver for one cost-aware long/short allocation decision.

Maximizes ``mu.w - (gamma / 2) w'Sigma w - sum_i c_i |w_i - w0_i|`` subject to
``lo_i <= w_i <= hi_i`` and ``sum_i |w_i| <= gross_max``.

The objective is strongly convex (``Sigma`` is positive definite) with a
separable L1 transaction-cost term. It is solved by cyclic coordinate descent
whose one-dimensional steps are exact (soft-threshold style: the minimizer of a
convex piecewise-quadratic is a kink, a sign-region vertex, or a box bound).
The coupling gross constraint is handled by an exact Lagrangian L1 multiplier
found by bisection (the dual is concave, so the constraint value is monotone in
the multiplier). Non-convergence, non-finite values and any constraint
violation raise :class:`AllocationSolverError`; callers must fail closed.
Pure NumPy, deterministic, no random state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_FEASIBILITY_TOLERANCE = 1e-9
_MAX_LAMBDA_DOUBLINGS = 200
_MAX_BISECTIONS = 200


class AllocationSolverError(RuntimeError):
    """The allocation problem is invalid or could not be solved to tolerance."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AllocationSolution:
    """Solved weights and the audit scalars of the solve."""

    weights: np.ndarray
    objective: float
    iterations: int
    gross_multiplier: float


def _finite_vector(value: object, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise AllocationSolverError("shape", f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise AllocationSolverError("non_finite", f"{name} must be finite")
    return array


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def allocation_objective(
    weights: np.ndarray,
    *,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    costs: np.ndarray,
    gamma: float,
) -> float:
    """Return the maximized net utility ``mu.w - risk - transaction cost``."""

    return float(
        expected_returns @ weights
        - 0.5 * gamma * (weights @ covariance @ weights)
        - costs @ np.abs(weights - current_weights)
    )


def _minimize_coordinate(
    curvature: float,
    linear: float,
    anchor: float,
    cost: float,
    multiplier: float,
    low: float,
    high: float,
) -> float:
    """Exact minimizer of a convex piecewise quadratic on ``[low, high]``.

    ``f(x) = curvature / 2 x^2 - linear x + cost |x - anchor| + multiplier |x|``.
    Candidates are the two kinks, the four sign-region vertices (the
    soft-threshold solutions) and both box bounds. Ties keep the earlier
    candidate, which prefers the no-trade anchor.
    """

    def value(x: float) -> float:
        return (
            0.5 * curvature * x * x
            - linear * x
            + cost * abs(x - anchor)
            + multiplier * abs(x)
        )

    candidates = [anchor, 0.0, low, high]
    for sign_trade in (-1.0, 1.0):
        for sign_level in (-1.0, 1.0):
            candidates.append(
                (linear - cost * sign_trade - multiplier * sign_level) / curvature
            )
    best_x = min(max(candidates[0], low), high)
    best_value = value(best_x)
    for candidate in candidates[1:]:
        clipped = min(max(candidate, low), high)
        candidate_value = value(clipped)
        if candidate_value < best_value:
            best_x = clipped
            best_value = candidate_value
    return best_x


def _coordinate_descent(
    *,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    costs: np.ndarray,
    gamma: float,
    low: np.ndarray,
    high: np.ndarray,
    multiplier: float,
    start: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    weights = start.copy()
    n_assets = weights.shape[0]
    for iteration in range(1, max_iter + 1):
        largest_change = 0.0
        for i in range(n_assets):
            cross = float(covariance[i] @ weights - covariance[i, i] * weights[i])
            updated = _minimize_coordinate(
                gamma * float(covariance[i, i]),
                float(expected_returns[i]) - gamma * cross,
                float(current_weights[i]),
                float(costs[i]),
                multiplier,
                float(low[i]),
                float(high[i]),
            )
            largest_change = max(largest_change, abs(updated - float(weights[i])))
            weights[i] = updated
        if not math.isfinite(largest_change):
            raise AllocationSolverError("non_finite", "coordinate step is not finite")
        if largest_change <= tol:
            return weights, iteration
    raise AllocationSolverError(
        "not_converged", f"coordinate descent did not converge in {max_iter} sweeps"
    )


def solve_cost_aware_allocation(
    *,
    expected_returns: object,
    covariance: object,
    current_weights: object,
    costs: object,
    gamma: float,
    w_max: float,
    gross_max: float,
    long_only: bool = False,
    max_iter: int = 2_000,
    tol: float = 1e-10,
) -> AllocationSolution:
    """Solve one allocation exactly or raise :class:`AllocationSolverError`."""

    gamma = _positive_finite(gamma, name="gamma")
    w_max = _positive_finite(w_max, name="w_max")
    gross_max = _positive_finite(gross_max, name="gross_max")
    tol = _positive_finite(tol, name="tol")
    if type(long_only) is not bool:
        raise TypeError("long_only must be a boolean")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    mu = np.asarray(expected_returns, dtype=np.float64)
    if mu.ndim != 1 or mu.size == 0:
        raise AllocationSolverError("shape", "expected_returns must be a vector")
    n_assets = int(mu.size)
    mu = _finite_vector(mu, name="expected_returns", size=n_assets)
    w0 = _finite_vector(current_weights, name="current_weights", size=n_assets)
    cost_vector = _finite_vector(costs, name="costs", size=n_assets)
    if np.any(cost_vector < 0.0):
        raise AllocationSolverError("negative_cost", "costs must be non-negative")
    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.shape != (n_assets, n_assets):
        raise AllocationSolverError("shape", "covariance must be square")
    if not np.all(np.isfinite(sigma)):
        raise AllocationSolverError("non_finite", "covariance must be finite")
    if not np.allclose(sigma, sigma.T, rtol=1e-9, atol=1e-14):
        raise AllocationSolverError("asymmetric_covariance", "covariance not symmetric")
    sigma = 0.5 * (sigma + sigma.T)
    if np.any(np.diag(sigma) <= 0.0):
        raise AllocationSolverError(
            "non_positive_variance", "covariance diagonal must be positive"
        )
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise AllocationSolverError(
            "covariance_not_positive_definite", "covariance is not positive definite"
        ) from exc

    low = np.full(n_assets, 0.0 if long_only else -w_max)
    high = np.full(n_assets, w_max)
    start = np.clip(w0, low, high)

    def solve(multiplier: float, warm: np.ndarray) -> tuple[np.ndarray, int]:
        return _coordinate_descent(
            expected_returns=mu,
            covariance=sigma,
            current_weights=w0,
            costs=cost_vector,
            gamma=gamma,
            low=low,
            high=high,
            multiplier=multiplier,
            start=warm,
            max_iter=max_iter,
            tol=tol,
        )

    weights, iterations = solve(0.0, start)
    multiplier = 0.0
    if float(np.abs(weights).sum()) > gross_max:
        lower = 0.0
        upper = max(1e-12, float(np.max(np.abs(mu))))
        for _ in range(_MAX_LAMBDA_DOUBLINGS):
            candidate, used = solve(upper, weights)
            iterations += used
            if float(np.abs(candidate).sum()) <= gross_max:
                break
            lower = upper
            upper *= 2.0
        else:
            raise AllocationSolverError(
                "gross_multiplier_unbounded", "gross constraint could not be met"
            )
        best = candidate
        for _ in range(_MAX_BISECTIONS):
            if upper - lower <= 1e-15 * (1.0 + upper):
                break
            middle = 0.5 * (lower + upper)
            candidate, used = solve(middle, best)
            iterations += used
            if float(np.abs(candidate).sum()) <= gross_max:
                upper = middle
                best = candidate
            else:
                lower = middle
        weights = best
        multiplier = upper

    gross = float(np.abs(weights).sum())
    if not np.all(np.isfinite(weights)):
        raise AllocationSolverError("non_finite", "solution is not finite")
    if gross > gross_max + _FEASIBILITY_TOLERANCE:
        raise AllocationSolverError("constraint_violation", "gross limit exceeded")
    if gross > gross_max:
        weights = weights * (gross_max / gross)
    weights = np.clip(weights, low, high)
    if float(np.abs(weights).sum()) > gross_max + _FEASIBILITY_TOLERANCE:
        raise AllocationSolverError("constraint_violation", "gross limit exceeded")
    objective = allocation_objective(
        weights,
        expected_returns=mu,
        covariance=sigma,
        current_weights=w0,
        costs=cost_vector,
        gamma=gamma,
    )
    if not math.isfinite(objective):
        raise AllocationSolverError("non_finite", "objective is not finite")
    return AllocationSolution(
        weights=weights,
        objective=objective,
        iterations=iterations,
        gross_multiplier=multiplier,
    )


__all__ = [
    "AllocationSolution",
    "AllocationSolverError",
    "allocation_objective",
    "solve_cost_aware_allocation",
]
