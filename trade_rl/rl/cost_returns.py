"""Pure vectorized-environment returns and GAE for independent constraint costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CostReturnResult:
    """Independent cost advantages and value targets with shape [T, E, C]."""

    advantages: np.ndarray
    returns: np.ndarray

    def __post_init__(self) -> None:
        advantages = np.asarray(self.advantages, dtype=np.float64).copy()
        returns = np.asarray(self.returns, dtype=np.float64).copy()
        if advantages.shape != returns.shape or advantages.ndim != 3:
            raise ValueError(
                "cost return result arrays must share a three-dimensional shape"
            )
        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise ValueError("cost return result arrays must be finite")
        advantages.setflags(write=False)
        returns.setflags(write=False)
        object.__setattr__(self, "advantages", advantages)
        object.__setattr__(self, "returns", returns)


def _finite_float_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def compute_cost_returns_and_advantages(
    *,
    costs: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    terminal_values: np.ndarray,
    last_values: np.ndarray,
    gammas: np.ndarray,
    gae_lambdas: np.ndarray,
) -> CostReturnResult:
    """Compute independent GAE with explicit termination and truncation semantics.

    ``terminated[t, e]`` marks a true economic terminal transition and therefore
    uses no bootstrap value. ``truncated[t, e]`` marks a time-limit boundary and
    bootstraps from ``terminal_values[t, e]`` without carrying GAE into the reset
    episode. A non-terminal final rollout transition bootstraps from
    ``last_values``.
    """

    costs_array = _finite_float_array(costs, name="costs")
    if costs_array.ndim != 3:
        raise ValueError("costs must be three-dimensional [steps, envs, costs]")
    values_array = _finite_float_array(values, name="values")
    if values_array.shape != costs_array.shape:
        raise ValueError("costs and values must have the same shape")
    terminal_values_array = _finite_float_array(
        terminal_values,
        name="terminal_values",
    )
    if terminal_values_array.shape != costs_array.shape:
        raise ValueError("terminal value shape must match costs")

    steps, environments, cost_count = costs_array.shape
    terminated_array = np.asarray(terminated, dtype=np.bool_)
    truncated_array = np.asarray(truncated, dtype=np.bool_)
    if terminated_array.shape != (steps, environments) or truncated_array.shape != (
        steps,
        environments,
    ):
        raise ValueError("termination shape must match [steps, environments]")
    if np.any(terminated_array & truncated_array):
        raise ValueError("a transition cannot both terminate and truncate")

    last_values_array = _finite_float_array(
        last_values,
        name="last_values",
    )
    if last_values_array.shape != (environments, cost_count):
        raise ValueError("last value shape must match [environments, costs]")
    gammas_array = _finite_float_array(gammas, name="gammas").reshape(-1)
    lambdas_array = _finite_float_array(
        gae_lambdas,
        name="gae_lambdas",
    ).reshape(-1)
    if gammas_array.shape != (cost_count,) or lambdas_array.shape != (cost_count,):
        raise ValueError("gamma and lambda cost dimension must match costs")
    if np.any((gammas_array <= 0.0) | (gammas_array > 1.0)):
        raise ValueError("gammas must be within (0, 1]")
    if np.any((lambdas_array < 0.0) | (lambdas_array > 1.0)):
        raise ValueError("gae_lambdas must be within [0, 1]")

    advantages = np.zeros_like(costs_array, dtype=np.float64)
    last_advantages = np.zeros((environments, cost_count), dtype=np.float64)
    gamma_row = gammas_array.reshape(1, cost_count)
    lambda_row = lambdas_array.reshape(1, cost_count)

    for step in range(steps - 1, -1, -1):
        true_terminal = terminated_array[step]
        time_limit = truncated_array[step]
        if step == steps - 1:
            continuation_values = last_values_array
        else:
            continuation_values = values_array[step + 1]
        next_values = np.where(
            true_terminal[:, None],
            0.0,
            np.where(
                time_limit[:, None],
                terminal_values_array[step],
                continuation_values,
            ),
        )
        carry_advantage = ~(true_terminal | time_limit)
        delta = costs_array[step] + gamma_row * next_values - values_array[step]
        last_advantages = (
            delta + gamma_row * lambda_row * carry_advantage[:, None] * last_advantages
        )
        advantages[step] = last_advantages

    returns = advantages + values_array
    return CostReturnResult(advantages=advantages, returns=returns)


__all__ = [
    "CostReturnResult",
    "compute_cost_returns_and_advantages",
]
