"""Private HiGHS model for the perfect-information linear benchmark."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearProgramSolution:
    """Raw lexicographic LP solution before economic replay validation."""

    target_weights: np.ndarray
    linearized_upper_bound: float
    selected_linearized_objective: float
    primary_status: int
    primary_message: str
    primary_iterations: int
    secondary_status: int
    secondary_message: str
    secondary_iterations: int


@dataclass(frozen=True, slots=True)
class _Layout:
    n_steps: int
    n_assets: int

    @property
    def weight_count(self) -> int:
        return self.n_steps * self.n_assets

    @property
    def absolute_start(self) -> int:
        return self.weight_count

    @property
    def turnover_start(self) -> int:
        return 2 * self.weight_count

    @property
    def variable_count(self) -> int:
        return self.turnover_start + (self.n_steps + 1) * self.n_assets

    def weight(self, step: int, asset: int) -> int:
        return step * self.n_assets + asset

    def absolute(self, step: int, asset: int) -> int:
        return self.absolute_start + step * self.n_assets + asset

    def turnover(self, step: int, asset: int) -> int:
        return self.turnover_start + step * self.n_assets + asset


def _scipy_modules() -> tuple[Any, Any]:
    try:
        optimize = importlib.import_module("scipy.optimize")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as error:
        raise RuntimeError(
            "perfect-information bound requires the optional 'oracle' dependency"
        ) from error
    return optimize, sparse


def _build_problem(
    returns: np.ndarray,
    *,
    transaction_cost_rate: np.ndarray,
    liquidation_cost_rate: np.ndarray,
    max_abs_weight: np.ndarray,
    max_gross: float,
    max_net_exposure: float | None,
    initial_weights: np.ndarray,
    minimum_period_net_return: float,
    feasibility_tolerance: float,
) -> tuple[
    np.ndarray,
    Any,
    np.ndarray,
    list[tuple[float | None, float | None]],
    _Layout,
]:
    _, sparse = _scipy_modules()
    n_steps, n_assets = returns.shape
    layout = _Layout(n_steps=n_steps, n_assets=n_assets)
    objective = np.zeros(layout.variable_count, dtype=np.float64)

    for step in range(n_steps):
        for asset in range(n_assets):
            objective[layout.weight(step, asset)] = -returns[step, asset]
            objective[layout.turnover(step, asset)] = transaction_cost_rate[asset]
    for asset in range(n_assets):
        objective[layout.turnover(n_steps, asset)] = liquidation_cost_rate[asset]

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    right_hand_side: list[float] = []

    def add_constraint(coefficients: dict[int, float], bound: float) -> None:
        row = len(right_hand_side)
        for column, coefficient in coefficients.items():
            if coefficient != 0.0:
                rows.append(row)
                columns.append(column)
                data.append(coefficient)
        right_hand_side.append(bound)

    for step in range(n_steps):
        for asset in range(n_assets):
            weight = layout.weight(step, asset)
            absolute = layout.absolute(step, asset)
            add_constraint({weight: 1.0, absolute: -1.0}, 0.0)
            add_constraint({weight: -1.0, absolute: -1.0}, 0.0)
        add_constraint(
            {layout.absolute(step, asset): 1.0 for asset in range(n_assets)},
            max_gross,
        )
        if max_net_exposure is not None:
            add_constraint(
                {layout.weight(step, asset): 1.0 for asset in range(n_assets)},
                max_net_exposure,
            )
            add_constraint(
                {layout.weight(step, asset): -1.0 for asset in range(n_assets)},
                max_net_exposure,
            )

    for step in range(n_steps):
        for asset in range(n_assets):
            current = layout.weight(step, asset)
            turnover = layout.turnover(step, asset)
            if step == 0:
                add_constraint(
                    {current: 1.0, turnover: -1.0},
                    float(initial_weights[asset]),
                )
                add_constraint(
                    {current: -1.0, turnover: -1.0},
                    float(-initial_weights[asset]),
                )
            else:
                previous = layout.weight(step - 1, asset)
                add_constraint({current: 1.0, previous: -1.0, turnover: -1.0}, 0.0)
                add_constraint({current: -1.0, previous: 1.0, turnover: -1.0}, 0.0)

    for asset in range(n_assets):
        final_weight = layout.weight(n_steps - 1, asset)
        terminal_turnover = layout.turnover(n_steps, asset)
        add_constraint({final_weight: 1.0, terminal_turnover: -1.0}, 0.0)
        add_constraint({final_weight: -1.0, terminal_turnover: -1.0}, 0.0)

    for step in range(n_steps):
        coefficients = {
            layout.weight(step, asset): -float(returns[step, asset])
            for asset in range(n_assets)
        }
        coefficients.update(
            {
                layout.turnover(step, asset): float(transaction_cost_rate[asset])
                for asset in range(n_assets)
            }
        )
        add_constraint(coefficients, -minimum_period_net_return)

    add_constraint(
        {
            layout.turnover(n_steps, asset): float(liquidation_cost_rate[asset])
            for asset in range(n_assets)
        },
        1.0 - feasibility_tolerance,
    )

    matrix = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(len(right_hand_side), layout.variable_count),
    ).tocsr()
    bounds: list[tuple[float | None, float | None]] = []
    for _step in range(n_steps):
        bounds.extend((-float(limit), float(limit)) for limit in max_abs_weight)
    for _step in range(n_steps):
        bounds.extend((0.0, float(limit)) for limit in max_abs_weight)
    bounds.extend((0.0, None) for _ in range((n_steps + 1) * n_assets))
    return (
        objective,
        matrix,
        np.asarray(right_hand_side, dtype=np.float64),
        bounds,
        layout,
    )


def _require_optimal(solution: Any, *, stage: str) -> None:
    if not bool(solution.success) or int(solution.status) != 0:
        raise RuntimeError(
            f"perfect-information {stage} solver did not report an optimal solution: "
            f"status={solution.status}, message={solution.message}"
        )


def solve_lexicographic_linear_program(
    returns: np.ndarray,
    *,
    transaction_cost_rate: np.ndarray,
    liquidation_cost_rate: np.ndarray,
    max_abs_weight: np.ndarray,
    max_gross: float,
    max_net_exposure: float | None,
    initial_weights: np.ndarray,
    minimum_period_net_return: float,
    objective_tolerance: float,
    feasibility_tolerance: float,
    solver_method: str,
) -> LinearProgramSolution:
    """Maximize the economic bound, then minimize turnover lexicographically."""

    optimize, sparse = _scipy_modules()
    objective, matrix, rhs, bounds, layout = _build_problem(
        returns,
        transaction_cost_rate=transaction_cost_rate,
        liquidation_cost_rate=liquidation_cost_rate,
        max_abs_weight=max_abs_weight,
        max_gross=max_gross,
        max_net_exposure=max_net_exposure,
        initial_weights=initial_weights,
        minimum_period_net_return=minimum_period_net_return,
        feasibility_tolerance=feasibility_tolerance,
    )
    primary = optimize.linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        bounds=bounds,
        method=solver_method,
        options={"presolve": True},
    )
    _require_optimal(primary, stage="primary")
    primary_objective = float(primary.fun)
    if not np.isfinite(primary_objective):
        raise RuntimeError(
            "perfect-information primary solver returned non-finite objective"
        )
    primary_vector = np.asarray(primary.x, dtype=np.float64)
    if (
        primary_vector.shape != (layout.variable_count,)
        or not np.isfinite(primary_vector).all()
    ):
        raise RuntimeError(
            "perfect-information primary solver returned invalid variables"
        )

    turnover_objective = np.zeros(layout.variable_count, dtype=np.float64)
    turnover_objective[layout.turnover_start :] = 1.0
    augmented_matrix = sparse.vstack(
        [matrix, sparse.csr_matrix(objective.reshape(1, -1))],
        format="csr",
    )
    augmented_rhs = np.concatenate(
        [rhs, np.asarray([primary_objective + objective_tolerance])]
    )
    secondary = optimize.linprog(
        turnover_objective,
        A_ub=augmented_matrix,
        b_ub=augmented_rhs,
        bounds=bounds,
        method=solver_method,
        options={"presolve": True},
    )
    _require_optimal(secondary, stage="secondary")
    vector = np.asarray(secondary.x, dtype=np.float64)
    if vector.shape != (layout.variable_count,) or not np.isfinite(vector).all():
        raise RuntimeError(
            "perfect-information secondary solver returned invalid variables"
        )

    weights = vector[: layout.weight_count].reshape(returns.shape)
    selected_objective = float(-objective @ vector)
    return LinearProgramSolution(
        target_weights=weights.copy(order="C"),
        linearized_upper_bound=float(-primary_objective),
        selected_linearized_objective=selected_objective,
        primary_status=int(primary.status),
        primary_message=str(primary.message),
        primary_iterations=int(getattr(primary, "nit", 0)),
        secondary_status=int(secondary.status),
        secondary_message=str(secondary.message),
        secondary_iterations=int(getattr(secondary, "nit", 0)),
    )


__all__ = ["LinearProgramSolution", "solve_lexicographic_linear_program"]
