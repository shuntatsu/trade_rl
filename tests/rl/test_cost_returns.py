from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.cost_returns import compute_cost_returns_and_advantages


def test_terminal_event_gamma_one_matches_monte_carlo_return() -> None:
    costs = np.array([[[0.0]], [[0.0]], [[1.0]]], dtype=np.float64)
    values = np.zeros_like(costs)

    result = compute_cost_returns_and_advantages(
        costs=costs,
        values=values,
        terminated=np.array([[False], [False], [True]]),
        truncated=np.zeros((3, 1), dtype=np.bool_),
        terminal_values=np.zeros_like(costs),
        last_values=np.zeros((1, 1), dtype=np.float64),
        gammas=np.array([1.0]),
        gae_lambdas=np.array([1.0]),
    )

    np.testing.assert_allclose(result.advantages[:, 0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.returns[:, 0, 0], [1.0, 1.0, 1.0])


def test_cost_lambda_changes_bias_variance_path_independently() -> None:
    costs = np.array(
        [
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    values = np.zeros_like(costs)

    result = compute_cost_returns_and_advantages(
        costs=costs,
        values=values,
        terminated=np.array([[False], [False], [True]]),
        truncated=np.zeros((3, 1), dtype=np.bool_),
        terminal_values=np.zeros_like(costs),
        last_values=np.zeros((1, 2), dtype=np.float64),
        gammas=np.array([1.0, 1.0]),
        gae_lambdas=np.array([0.5, 1.0]),
    )

    np.testing.assert_allclose(result.advantages[:, 0, 0], [0.25, 0.5, 1.0])
    np.testing.assert_allclose(result.advantages[:, 0, 1], [1.0, 1.0, 1.0])


def test_truncation_bootstraps_terminal_value_without_cross_episode_leakage() -> None:
    costs = np.zeros((3, 1, 1), dtype=np.float64)
    values = np.zeros_like(costs)
    terminal_values = np.zeros_like(costs)
    terminal_values[1, 0, 0] = 0.7

    result = compute_cost_returns_and_advantages(
        costs=costs,
        values=values,
        terminated=np.array([[False], [False], [True]]),
        truncated=np.array([[False], [True], [False]]),
        terminal_values=terminal_values,
        last_values=np.zeros((1, 1), dtype=np.float64),
        gammas=np.array([1.0]),
        gae_lambdas=np.array([1.0]),
    )

    np.testing.assert_allclose(result.returns[:, 0, 0], [0.7, 0.7, 0.0])


def test_vector_environments_keep_episode_boundaries_independent() -> None:
    costs = np.array(
        [
            [[0.0], [0.0]],
            [[1.0], [0.0]],
            [[0.0], [2.0]],
        ],
        dtype=np.float64,
    )
    values = np.zeros_like(costs)

    result = compute_cost_returns_and_advantages(
        costs=costs,
        values=values,
        terminated=np.array(
            [
                [False, False],
                [True, False],
                [False, True],
            ]
        ),
        truncated=np.zeros((3, 2), dtype=np.bool_),
        terminal_values=np.zeros_like(costs),
        last_values=np.zeros((2, 1), dtype=np.float64),
        gammas=np.array([1.0]),
        gae_lambdas=np.array([1.0]),
    )

    np.testing.assert_allclose(result.returns[:, 0, 0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(result.returns[:, 1, 0], [2.0, 2.0, 2.0])


def test_nonterminal_rollout_end_uses_last_cost_values() -> None:
    result = compute_cost_returns_and_advantages(
        costs=np.zeros((2, 1, 1), dtype=np.float64),
        values=np.zeros((2, 1, 1), dtype=np.float64),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        terminal_values=np.zeros((2, 1, 1), dtype=np.float64),
        last_values=np.array([[0.4]], dtype=np.float64),
        gammas=np.array([1.0]),
        gae_lambdas=np.array([1.0]),
    )

    np.testing.assert_allclose(result.returns[:, 0, 0], [0.4, 0.4])


@pytest.mark.parametrize(
    "override, message",
    [
        ({"costs": np.zeros((2, 1), dtype=np.float64)}, "three-dimensional"),
        (
            {"values": np.zeros((2, 2, 1), dtype=np.float64)},
            "same shape",
        ),
        (
            {"terminated": np.zeros((2, 2), dtype=np.bool_)},
            "termination shape",
        ),
        (
            {"gammas": np.array([1.0, 1.0])},
            "cost dimension",
        ),
        (
            {"costs": np.array([[[np.nan]], [[0.0]]])},
            "finite",
        ),
        (
            {
                "terminated": np.array([[True], [False]]),
                "truncated": np.array([[True], [False]]),
            },
            "both terminate and truncate",
        ),
    ],
)
def test_cost_return_calculation_fails_closed(
    override: dict[str, np.ndarray],
    message: str,
) -> None:
    values: dict[str, np.ndarray] = {
        "costs": np.zeros((2, 1, 1), dtype=np.float64),
        "values": np.zeros((2, 1, 1), dtype=np.float64),
        "terminated": np.zeros((2, 1), dtype=np.bool_),
        "truncated": np.zeros((2, 1), dtype=np.bool_),
        "terminal_values": np.zeros((2, 1, 1), dtype=np.float64),
        "last_values": np.zeros((1, 1), dtype=np.float64),
        "gammas": np.array([1.0]),
        "gae_lambdas": np.array([0.95]),
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        compute_cost_returns_and_advantages(**values)
