"""Convex solver for one cost-aware long/short allocation decision.

Maximizes ``mu.w - (gamma / 2) w'Sigma w - sum_i c_i |w_i - w0_i|`` subject to
``lo_i <= w_i <= hi_i`` and ``sum_i |w_i| <= gross_max``.

The objective is strongly convex (``Sigma`` is positive definite) with a
separable L1 transaction-cost term, so the optimum is unique. It is computed by
cyclic coordinate descent whose one-dimensional steps are exact (the minimizer
of a convex piecewise quadratic is a kink, a sign-region vertex, or a box
bound), accelerated by an active-set Newton step: coordinates sitting on a kink
or bound are frozen, the remaining coordinates solve the smooth stationarity
system for their current sign pattern, and an exact piecewise-quadratic line
search along that direction guarantees a monotone objective decrease. The
coupling gross constraint is handled by an L1 Lagrange multiplier found by
bisection (the gross exposure of the multiplier-penalized optimum is continuous
and non-increasing in the multiplier).

The result is *tolerance-accurate and certified*, not symbolically exact: a
solution is returned only after a KKT optimality certificate (the subgradient
condition, including the box normal cone and gross complementary slackness)
passes within ``kkt_tol``. Non-convergence, non-finite values, any constraint
violation and any failed certificate raise :class:`AllocationSolverError`;
callers must fail closed. Pure NumPy, deterministic, no random state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_FEASIBILITY_TOLERANCE = 1e-9
_GROSS_SLACK_TOLERANCE = 1e-10
_SNAP_TOLERANCE = 1e-11
_MAX_LAMBDA_DOUBLINGS = 200
_MAX_BISECTIONS = 200


class AllocationSolverError(RuntimeError):
    """The allocation problem is invalid or could not be solved to tolerance."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AllocationSolution:
    """Solved weights and the audit scalars of the solve.

    ``kkt_residual`` is the largest per-coordinate violation of the subgradient
    optimality condition divided by the problem's gradient scale; it never
    exceeds the ``kkt_tol`` the solve was asked to certify.
    """

    weights: np.ndarray
    objective: float
    iterations: int
    gross_multiplier: float
    kkt_residual: float = 0.0


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


def kkt_residual(
    weights: np.ndarray,
    *,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    costs: np.ndarray,
    gamma: float,
    low: np.ndarray,
    high: np.ndarray,
    multiplier: float,
) -> float:
    """Largest violation of the subgradient optimality condition (absolute units).

    Stationarity of the penalized problem requires, per coordinate,
    ``0 in g_i + c_i d|w_i - w0_i| + multiplier d|w_i| + N_box_i(w_i)`` with
    ``g = gamma Sigma w - mu``. Each set is an interval (a point away from its
    kink, a symmetric interval on it, a half line on an active bound); the
    residual is the distance from ``-g_i`` to their sum.
    """

    gradient = gamma * (covariance @ weights) - expected_returns
    delta = weights - current_weights
    on_anchor = np.abs(delta) <= _SNAP_TOLERANCE
    anchor_sign = np.where(on_anchor, 0.0, np.sign(delta))
    cost_low = np.where(on_anchor, -costs, anchor_sign * costs)
    cost_high = np.where(on_anchor, costs, anchor_sign * costs)
    if multiplier > 0.0:
        on_zero = np.abs(weights) <= _SNAP_TOLERANCE
        zero_sign = np.where(on_zero, 0.0, np.sign(weights))
        level_low = np.where(on_zero, -multiplier, zero_sign * multiplier)
        level_high = np.where(on_zero, multiplier, zero_sign * multiplier)
    else:
        level_low = level_high = np.zeros_like(weights)
    at_high = weights >= high - _SNAP_TOLERANCE
    at_low = weights <= low + _SNAP_TOLERANCE
    normal_low = np.where(at_low & ~at_high, -np.inf, 0.0)
    normal_high = np.where(at_high & ~at_low, np.inf, 0.0)
    lower = cost_low + level_low + normal_low
    upper = cost_high + level_high + normal_high
    target = -gradient
    violation = np.maximum(np.maximum(lower - target, target - upper), 0.0)
    return float(np.max(violation))


def _line_search(
    weights: np.ndarray,
    direction: np.ndarray,
    alpha_max: float,
    *,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    costs: np.ndarray,
    gamma: float,
    multiplier: float,
) -> np.ndarray:
    """Exact minimization of the convex piecewise quadratic loss along a ray.

    The directional derivative is non-decreasing and piecewise linear in the
    step, so the minimizer is found by walking the kink-delimited segments and
    comparing derivative signs. Loss *values* are deliberately not compared:
    along a flat valley the improvement is far below the floating-point
    resolution of the loss itself.
    """

    curvature = gamma * float(direction @ (covariance @ direction))
    if not curvature > 0.0 or not alpha_max > 0.0:
        return weights
    smooth_slope = -float(expected_returns @ direction) + gamma * float(
        weights @ (covariance @ direction)
    )
    points = {0.0, alpha_max}
    for i in np.flatnonzero(direction):
        kinks = [current_weights[i]]
        if multiplier > 0.0:
            kinks.append(0.0)
        for kink in kinks:
            alpha = float((kink - weights[i]) / direction[i])
            if 0.0 < alpha < alpha_max:
                points.add(alpha)
    ordered = sorted(points)
    step = alpha_max
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        middle = weights + 0.5 * (left + right) * direction
        slope = smooth_slope + float(
            np.sum(costs * np.sign(middle - current_weights) * direction)
        )
        if multiplier > 0.0:
            slope += multiplier * float(np.sum(np.sign(middle) * direction))
        stationary = -slope / curvature
        if stationary <= left:
            step = left
            break
        if stationary < right:
            step = stationary
            break
    if step <= 0.0:
        return weights
    return weights + step * direction


def _newton_step(
    weights: np.ndarray,
    *,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    costs: np.ndarray,
    gamma: float,
    low: np.ndarray,
    high: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Active-set Newton extrapolation with a monotone exact line search.

    Coordinates on a kink or a bound are frozen; the free coordinates solve the
    stationarity system of the current sign pattern. The returned point never
    has a larger loss than ``weights`` and stays inside the box.
    """

    on_bound = (weights >= high - _SNAP_TOLERANCE) | (weights <= low + _SNAP_TOLERANCE)
    on_anchor = np.abs(weights - current_weights) <= _SNAP_TOLERANCE
    on_zero = (
        (np.abs(weights) <= _SNAP_TOLERANCE)
        if multiplier > 0.0
        else np.zeros_like(on_bound)
    )
    frozen = on_bound | on_anchor | on_zero
    free = ~frozen
    if not np.any(free):
        return weights
    anchor_sign = np.sign(weights - current_weights)
    level_sign = np.sign(weights) if multiplier > 0.0 else np.zeros_like(weights)
    right_hand_side = (
        expected_returns[free]
        - gamma * (covariance[np.ix_(free, frozen)] @ weights[frozen])
        - costs[free] * anchor_sign[free]
        - multiplier * level_sign[free]
    )
    try:
        solved = np.linalg.solve(
            gamma * covariance[np.ix_(free, free)], right_hand_side
        )
    except np.linalg.LinAlgError:
        return weights
    if not np.all(np.isfinite(solved)):
        return weights
    direction = np.zeros_like(weights)
    direction[free] = solved - weights[free]
    alpha_max = 1.0
    for i in np.flatnonzero(direction):
        if direction[i] > 0.0:
            alpha_max = min(alpha_max, float((high[i] - weights[i]) / direction[i]))
        else:
            alpha_max = min(alpha_max, float((low[i] - weights[i]) / direction[i]))
    return _line_search(
        weights,
        direction,
        max(alpha_max, 0.0),
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=current_weights,
        costs=costs,
        gamma=gamma,
        multiplier=multiplier,
    )


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
        before = weights.copy()
        for i in range(n_assets):
            cross = float(covariance[i] @ weights - covariance[i, i] * weights[i])
            weights[i] = _minimize_coordinate(
                gamma * float(covariance[i, i]),
                float(expected_returns[i]) - gamma * cross,
                float(current_weights[i]),
                float(costs[i]),
                multiplier,
                float(low[i]),
                float(high[i]),
            )
        # The Newton extrapolation runs *before* the stopping test so that slow
        # creeping along an ill-conditioned valley (tiny sweep changes far from
        # the optimum) cannot be mistaken for convergence.
        weights = _newton_step(
            weights,
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=current_weights,
            costs=costs,
            gamma=gamma,
            low=low,
            high=high,
            multiplier=multiplier,
        )
        largest_change = float(np.max(np.abs(weights - before)))
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
    kkt_tol: float = 1e-8,
) -> AllocationSolution:
    """Solve one allocation to a certified tolerance or raise.

    ``tol`` is the largest coordinate change that ends a descent sweep;
    ``kkt_tol`` bounds the final subgradient residual relative to the gradient
    scale ``max(|mu|, c, gamma * max Sigma_ii * w_max)``. The returned weights are
    feasible within ``1e-9`` and satisfy that certificate.
    """

    gamma = _positive_finite(gamma, name="gamma")
    w_max = _positive_finite(w_max, name="w_max")
    gross_max = _positive_finite(gross_max, name="gross_max")
    tol = _positive_finite(tol, name="tol")
    kkt_tol = _positive_finite(kkt_tol, name="kkt_tol")
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
    scale = max(
        float(np.max(np.abs(mu))),
        float(np.max(cost_vector)),
        gamma * float(np.max(np.diag(sigma))) * w_max,
    )

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
            if gross_max - float(np.abs(best).sum()) <= _GROSS_SLACK_TOLERANCE:
                break
            middle = 0.5 * (lower + upper)
            if not lower < middle < upper:
                break  # the multiplier bracket reached floating-point resolution
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
    gross = float(np.abs(weights).sum())
    if gross > gross_max + _FEASIBILITY_TOLERANCE:
        raise AllocationSolverError("constraint_violation", "gross limit exceeded")
    slackness = multiplier * max(gross_max - gross, 0.0) / scale
    if not slackness <= kkt_tol:
        raise AllocationSolverError(
            "kkt_violation",
            f"gross complementary slackness {slackness:.3e} exceeds {kkt_tol:.3e}",
        )
    residual = (
        kkt_residual(
            weights,
            expected_returns=mu,
            covariance=sigma,
            current_weights=w0,
            costs=cost_vector,
            gamma=gamma,
            low=low,
            high=high,
            multiplier=multiplier,
        )
        / scale
    )
    if not math.isfinite(residual) or residual > kkt_tol:
        raise AllocationSolverError(
            "kkt_violation", f"optimality residual {residual:.3e} exceeds {kkt_tol:.3e}"
        )
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
        kkt_residual=residual,
    )


__all__ = [
    "AllocationSolution",
    "AllocationSolverError",
    "allocation_objective",
    "kkt_residual",
    "solve_cost_aware_allocation",
]
