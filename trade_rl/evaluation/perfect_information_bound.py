"""Perfect-information linear bound for research evaluation only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation._perfect_information_lp import (
    solve_lexicographic_linear_program,
)

PERFECT_INFORMATION_BOUND_SCHEMA = "perfect_information_linear_bound_v1"
_MAX_FLOAT_LOG = math.log(float(np.finfo(np.float64).max))


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _asset_tuple(
    name: str,
    value: object,
    *,
    n_assets: int,
) -> tuple[float, ...]:
    if isinstance(value, Real):
        result = (_finite_float(name, value),) * n_assets
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = tuple(_finite_float(name, item) for item in value)
        if len(result) != n_assets:
            raise ValueError(f"{name} must contain exactly n_assets values")
    else:
        raise ValueError(f"{name} must be a scalar or finite sequence")
    return result


@dataclass(frozen=True, slots=True, init=False)
class PerfectInformationBoundConfig:
    """Validated constraints for the linear perfect-information benchmark."""

    n_assets: int
    transaction_cost_rate: tuple[float, ...]
    liquidation_cost_rate: tuple[float, ...]
    max_abs_weight: tuple[float, ...]
    max_gross: float
    max_net_exposure: float | None
    initial_weights: tuple[float, ...]
    minimum_period_net_return: float
    lexicographic_objective_tolerance: float
    feasibility_tolerance: float
    solver_method: str
    schema_version: str

    def __init__(
        self,
        *,
        n_assets: int,
        transaction_cost_rate: float | Sequence[float] = 0.0,
        liquidation_cost_rate: float | Sequence[float] | None = None,
        max_abs_weight: float | Sequence[float] = 0.45,
        max_gross: float = 1.0,
        max_net_exposure: float | None = None,
        initial_weights: Sequence[float] | None = None,
        minimum_period_net_return: float = -0.999,
        lexicographic_objective_tolerance: float = 1e-10,
        feasibility_tolerance: float = 1e-8,
        solver_method: str = "highs",
    ) -> None:
        if isinstance(n_assets, bool) or not isinstance(n_assets, int) or n_assets <= 0:
            raise ValueError("n_assets must be a positive integer")
        transaction = _asset_tuple(
            "transaction_cost_rate", transaction_cost_rate, n_assets=n_assets
        )
        liquidation = _asset_tuple(
            "liquidation_cost_rate",
            transaction if liquidation_cost_rate is None else liquidation_cost_rate,
            n_assets=n_assets,
        )
        max_abs = _asset_tuple("max_abs_weight", max_abs_weight, n_assets=n_assets)
        if any(value < 0.0 for value in transaction):
            raise ValueError("transaction_cost_rate must be non-negative")
        if any(value < 0.0 for value in liquidation):
            raise ValueError("liquidation_cost_rate must be non-negative")
        if any(value <= 0.0 for value in max_abs):
            raise ValueError("max_abs_weight must be positive")

        gross = _finite_float("max_gross", max_gross)
        if gross <= 0.0:
            raise ValueError("max_gross must be positive")
        if max_net_exposure is None:
            net = None
        else:
            net = _finite_float("max_net_exposure", max_net_exposure)
            if net < 0.0 or net > gross:
                raise ValueError(
                    "max_net_exposure must be non-negative and no greater "
                    "than max_gross"
                )

        weights = (
            (0.0,) * n_assets
            if initial_weights is None
            else _asset_tuple("initial_weights", initial_weights, n_assets=n_assets)
        )
        tolerance = _finite_float("feasibility_tolerance", feasibility_tolerance)
        if tolerance <= 0.0:
            raise ValueError("feasibility_tolerance must be positive")
        if any(
            abs(weight) > limit + tolerance
            for weight, limit in zip(weights, max_abs, strict=True)
        ):
            raise ValueError("initial_weights exceed max_abs_weight")
        if sum(abs(weight) for weight in weights) > gross + tolerance:
            raise ValueError("initial_weights exceed max_gross")
        if net is not None and abs(sum(weights)) > net + tolerance:
            raise ValueError("initial_weights exceed max_net_exposure")

        minimum_return = _finite_float(
            "minimum_period_net_return", minimum_period_net_return
        )
        if minimum_return <= -1.0:
            raise ValueError("minimum_period_net_return must be greater than -1")
        objective_tolerance = _finite_float(
            "lexicographic_objective_tolerance",
            lexicographic_objective_tolerance,
        )
        if objective_tolerance < 0.0:
            raise ValueError("lexicographic_objective_tolerance must be non-negative")
        if solver_method != "highs":
            raise ValueError("solver_method must be 'highs'")

        object.__setattr__(self, "n_assets", n_assets)
        object.__setattr__(self, "transaction_cost_rate", transaction)
        object.__setattr__(self, "liquidation_cost_rate", liquidation)
        object.__setattr__(self, "max_abs_weight", max_abs)
        object.__setattr__(self, "max_gross", gross)
        object.__setattr__(self, "max_net_exposure", net)
        object.__setattr__(self, "initial_weights", weights)
        object.__setattr__(self, "minimum_period_net_return", minimum_return)
        object.__setattr__(
            self, "lexicographic_objective_tolerance", objective_tolerance
        )
        object.__setattr__(self, "feasibility_tolerance", tolerance)
        object.__setattr__(self, "solver_method", solver_method)
        object.__setattr__(self, "schema_version", PERFECT_INFORMATION_BOUND_SCHEMA)

    def digest_payload(self) -> dict[str, object]:
        return {
            "feasibility_tolerance": self.feasibility_tolerance,
            "initial_weights": self.initial_weights,
            "lexicographic_objective_tolerance": (
                self.lexicographic_objective_tolerance
            ),
            "liquidation_cost_rate": self.liquidation_cost_rate,
            "max_abs_weight": self.max_abs_weight,
            "max_gross": self.max_gross,
            "max_net_exposure": self.max_net_exposure,
            "minimum_period_net_return": self.minimum_period_net_return,
            "n_assets": self.n_assets,
            "schema_version": self.schema_version,
            "solver_method": self.solver_method,
            "transaction_cost_rate": self.transaction_cost_rate,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class PerfectInformationBoundResult:
    """Optimal linear bound and independently reconstructed replay evidence."""

    target_weights: np.ndarray
    absolute_weights: np.ndarray
    turnover: np.ndarray
    period_gross_returns: np.ndarray
    period_transaction_costs: np.ndarray
    period_net_returns: np.ndarray
    terminal_liquidation_cost: float
    linearized_log_upper_bound: float
    selected_path_linearized_objective: float
    replay_log_return: float
    replay_total_return: float | None
    primary_solver_status: int
    primary_solver_message: str
    primary_solver_iterations: int
    secondary_solver_status: int
    secondary_solver_message: str
    secondary_solver_iterations: int
    max_primal_violation: float
    problem_digest: str
    config_digest: str
    schema_version: str = PERFECT_INFORMATION_BOUND_SCHEMA

    def digest_payload(self) -> dict[str, object]:
        return {
            "absolute_weights": self.absolute_weights.tolist(),
            "config_digest": self.config_digest,
            "linearized_log_upper_bound": self.linearized_log_upper_bound,
            "max_primal_violation": self.max_primal_violation,
            "period_gross_returns": self.period_gross_returns.tolist(),
            "period_net_returns": self.period_net_returns.tolist(),
            "period_transaction_costs": self.period_transaction_costs.tolist(),
            "primary_solver_status": self.primary_solver_status,
            "problem_digest": self.problem_digest,
            "replay_log_return": self.replay_log_return,
            "replay_total_return": self.replay_total_return,
            "schema_version": self.schema_version,
            "secondary_solver_status": self.secondary_solver_status,
            "selected_path_linearized_objective": (
                self.selected_path_linearized_objective
            ),
            "target_weights": self.target_weights.tolist(),
            "terminal_liquidation_cost": self.terminal_liquidation_cost,
            "turnover": self.turnover.tolist(),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


def _validated_returns(returns: np.ndarray, *, n_assets: int) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != n_assets:
        raise ValueError("returns must be a non-empty (n_steps, n_assets) matrix")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if np.any(values <= -1.0):
        raise ValueError("returns must be greater than -1")
    return values.copy(order="C")


def _canonical_float(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy(order="C")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _turnover(weights: np.ndarray, initial_weights: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            np.abs(weights[0] - initial_weights),
            np.abs(np.diff(weights, axis=0)),
            np.abs(weights[-1]),
        ]
    )


def _constraint_violation(
    *,
    weights: np.ndarray,
    turnover: np.ndarray,
    period_net_returns: np.ndarray,
    terminal_cost: float,
    config: PerfectInformationBoundConfig,
) -> float:
    violations = [0.0]
    max_abs = np.asarray(config.max_abs_weight, dtype=np.float64)
    violations.append(float(np.max(np.abs(weights) - max_abs[None, :])))
    violations.append(float(np.max(np.abs(weights).sum(axis=1) - config.max_gross)))
    if config.max_net_exposure is not None:
        violations.append(
            float(np.max(np.abs(weights.sum(axis=1)) - config.max_net_exposure))
        )
    expected_turnover = _turnover(
        weights,
        np.asarray(config.initial_weights, dtype=np.float64),
    )
    violations.append(float(np.max(np.abs(turnover - expected_turnover))))
    violations.append(
        float(np.max(config.minimum_period_net_return - period_net_returns))
    )
    violations.append(terminal_cost - (1.0 - config.feasibility_tolerance))
    return max(violations)


def solve_perfect_information_bound(
    returns: np.ndarray,
    config: PerfectInformationBoundConfig,
) -> PerfectInformationBoundResult:
    """Solve the linearized perfect-information benchmark with HiGHS."""

    values = _validated_returns(returns, n_assets=config.n_assets)
    problem_digest = content_digest(
        {
            "config_digest": config.digest,
            "returns": values.tolist(),
            "schema_version": "perfect_information_linear_problem_v1",
        }
    )
    transaction = np.asarray(config.transaction_cost_rate, dtype=np.float64)
    liquidation = np.asarray(config.liquidation_cost_rate, dtype=np.float64)
    initial = np.asarray(config.initial_weights, dtype=np.float64)
    solution = solve_lexicographic_linear_program(
        values,
        transaction_cost_rate=transaction,
        liquidation_cost_rate=liquidation,
        max_abs_weight=np.asarray(config.max_abs_weight, dtype=np.float64),
        max_gross=config.max_gross,
        max_net_exposure=config.max_net_exposure,
        initial_weights=initial,
        minimum_period_net_return=config.minimum_period_net_return,
        objective_tolerance=config.lexicographic_objective_tolerance,
        feasibility_tolerance=config.feasibility_tolerance,
        solver_method=config.solver_method,
    )
    weights = np.asarray(solution.target_weights, dtype=np.float64).copy(order="C")
    if weights.shape != values.shape or not np.isfinite(weights).all():
        raise RuntimeError("perfect-information solver returned invalid target weights")
    weights[weights == 0.0] = 0.0
    upper_bound = _canonical_float(solution.linearized_upper_bound)
    selected_evidence = _canonical_float(solution.selected_linearized_objective)
    if not math.isfinite(upper_bound) or not math.isfinite(selected_evidence):
        raise RuntimeError(
            "perfect-information solver returned non-finite objective evidence"
        )
    turnover = _turnover(weights, initial)
    gross_returns = np.sum(values * weights, axis=1)
    transaction_costs = np.sum(turnover[:-1] * transaction[None, :], axis=1)
    net_returns = gross_returns - transaction_costs
    terminal_cost = _canonical_float(float(np.sum(turnover[-1] * liquidation)))
    if np.any(net_returns <= -1.0) or terminal_cost >= 1.0:
        raise RuntimeError(
            "perfect-information replay has a non-positive wealth factor"
        )

    selected_objective = _canonical_float(float(np.sum(net_returns) - terminal_cost))
    if not math.isclose(
        selected_objective,
        selected_evidence,
        rel_tol=0.0,
        abs_tol=config.feasibility_tolerance,
    ):
        raise RuntimeError("independent replay disagrees with the LP objective")
    if selected_objective < (
        upper_bound
        - config.lexicographic_objective_tolerance
        - config.feasibility_tolerance
    ):
        raise RuntimeError(
            "secondary solution violates the primary objective tolerance"
        )
    replay_log_return = _canonical_float(
        float(np.sum(np.log1p(net_returns)) + math.log1p(-terminal_cost))
    )
    if replay_log_return > upper_bound + config.feasibility_tolerance:
        raise RuntimeError("linearized upper bound is below exact replay log return")
    max_violation = _constraint_violation(
        weights=weights,
        turnover=turnover,
        period_net_returns=net_returns,
        terminal_cost=terminal_cost,
        config=config,
    )
    if max_violation > config.feasibility_tolerance:
        raise RuntimeError(
            "perfect-information solution violates declared constraints: "
            f"{max_violation}"
        )

    return PerfectInformationBoundResult(
        target_weights=_readonly(weights),
        absolute_weights=_readonly(np.abs(weights)),
        turnover=_readonly(turnover),
        period_gross_returns=_readonly(gross_returns),
        period_transaction_costs=_readonly(transaction_costs),
        period_net_returns=_readonly(net_returns),
        terminal_liquidation_cost=terminal_cost,
        linearized_log_upper_bound=upper_bound,
        selected_path_linearized_objective=selected_objective,
        replay_log_return=replay_log_return,
        replay_total_return=(
            None
            if replay_log_return > _MAX_FLOAT_LOG
            else _canonical_float(math.expm1(replay_log_return))
        ),
        primary_solver_status=solution.primary_status,
        primary_solver_message=solution.primary_message,
        primary_solver_iterations=solution.primary_iterations,
        secondary_solver_status=solution.secondary_status,
        secondary_solver_message=solution.secondary_message,
        secondary_solver_iterations=solution.secondary_iterations,
        max_primal_violation=max_violation,
        problem_digest=problem_digest,
        config_digest=config.digest,
    )


__all__ = [
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "solve_perfect_information_bound",
]
