from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.evaluation.perfect_information_bound import (
    PerfectInformationBoundConfig,
    PerfectInformationBoundResult,
    solve_perfect_information_bound,
)


def test_monotonic_positive_returns_choose_maximum_long_weight() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[0.10], [0.10]], dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            max_abs_weight=0.4,
            max_gross=0.4,
            max_net_exposure=0.4,
        ),
    )

    assert isinstance(result, PerfectInformationBoundResult)
    np.testing.assert_allclose(result.target_weights, [[0.4], [0.4]], atol=1e-8)


def test_monotonic_negative_returns_choose_maximum_short_weight() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[-0.10], [-0.10]], dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            max_abs_weight=0.4,
            max_gross=0.4,
            max_net_exposure=0.4,
        ),
    )

    np.testing.assert_allclose(result.target_weights, [[-0.4], [-0.4]], atol=1e-8)


def test_flat_returns_with_costs_stay_in_cash() -> None:
    result = solve_perfect_information_bound(
        np.zeros((4, 1), dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            transaction_cost_rate=0.01,
            liquidation_cost_rate=0.01,
        ),
    )

    np.testing.assert_allclose(result.target_weights, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.turnover, 0.0, atol=1e-10)


def test_zero_cost_flat_market_avoids_extra_turnover() -> None:
    result = solve_perfect_information_bound(
        np.zeros((3, 1), dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            initial_weights=(0.3,),
            max_abs_weight=0.4,
            max_gross=0.4,
            max_net_exposure=0.4,
        ),
    )

    assert float(result.turnover.sum()) == pytest.approx(0.3, abs=1e-8)


def test_multi_asset_solution_respects_all_exposure_constraints() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[0.20, 0.10, -0.15], [0.10, 0.05, -0.10]], dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=3,
            max_abs_weight=(0.45, 0.30, 0.25),
            max_gross=0.60,
            max_net_exposure=0.20,
        ),
    )

    assert np.all(
        np.abs(result.target_weights) <= np.asarray([0.45, 0.30, 0.25]) + 1e-8
    )
    assert np.all(np.abs(result.target_weights).sum(axis=1) <= 0.60 + 1e-8)
    assert np.all(np.abs(result.target_weights.sum(axis=1)) <= 0.20 + 1e-8)


def test_large_cost_suppresses_profitable_direction_switching() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[0.02], [-0.02]], dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            transaction_cost_rate=0.03,
            liquidation_cost_rate=0.03,
            max_abs_weight=0.45,
            max_gross=0.45,
            max_net_exposure=0.45,
        ),
    )

    np.testing.assert_allclose(result.target_weights, 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "returns",
    [
        np.asarray([], dtype=np.float64),
        np.asarray([0.1, 0.2], dtype=np.float64),
        np.asarray([[0.1, 0.2]], dtype=np.float64),
        np.asarray([[math.nan]], dtype=np.float64),
        np.asarray([[-1.0]], dtype=np.float64),
    ],
)
def test_solver_rejects_invalid_return_matrices(returns: np.ndarray) -> None:
    with pytest.raises(ValueError, match="returns"):
        solve_perfect_information_bound(
            returns,
            PerfectInformationBoundConfig(n_assets=1),
        )


def test_solver_fails_closed_when_constraints_are_infeasible() -> None:
    with pytest.raises(RuntimeError, match="optimal"):
        solve_perfect_information_bound(
            np.zeros((2, 1), dtype=np.float64),
            PerfectInformationBoundConfig(
                n_assets=1,
                minimum_period_net_return=0.1,
            ),
        )


def test_result_reports_certified_upper_bound_and_exact_replay() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[0.08, -0.03], [0.02, 0.05], [-0.01, 0.04]], dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=2,
            transaction_cost_rate=(0.001, 0.002),
            liquidation_cost_rate=(0.001, 0.002),
            max_abs_weight=(0.4, 0.4),
            max_gross=0.6,
            max_net_exposure=0.4,
        ),
    )

    assert result.primary_solver_status == 0
    assert result.secondary_solver_status == 0
    assert result.max_primal_violation <= 1e-8
    assert (
        result.linearized_log_upper_bound + 1e-8
        >= result.selected_path_linearized_objective
    )
    assert result.linearized_log_upper_bound + 1e-8 >= result.replay_log_return
    assert result.replay_total_return == pytest.approx(
        math.expm1(result.replay_log_return)
    )


def _grid_linear_objective(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    transaction_cost: float,
    liquidation_cost: float,
) -> float:
    initial = np.zeros(weights.shape[1], dtype=np.float64)
    turnover = np.vstack(
        [
            np.abs(weights[0] - initial),
            np.abs(np.diff(weights, axis=0)),
            np.abs(weights[-1]),
        ]
    )
    return float(
        np.sum(returns * weights)
        - transaction_cost * turnover[:-1].sum()
        - liquidation_cost * turnover[-1].sum()
    )


def test_linear_program_dominates_tiny_brute_force_grid() -> None:
    returns = np.asarray([[0.08], [-0.03]], dtype=np.float64)
    transaction_cost = 0.01
    liquidation_cost = 0.005
    result = solve_perfect_information_bound(
        returns,
        PerfectInformationBoundConfig(
            n_assets=1,
            transaction_cost_rate=transaction_cost,
            liquidation_cost_rate=liquidation_cost,
            max_abs_weight=0.4,
            max_gross=0.4,
            max_net_exposure=0.4,
        ),
    )

    grid = (-0.4, 0.0, 0.4)
    enumerated = [
        _grid_linear_objective(
            np.asarray([[first], [second]], dtype=np.float64),
            returns,
            transaction_cost=transaction_cost,
            liquidation_cost=liquidation_cost,
        )
        for first in grid
        for second in grid
    ]
    assert result.linearized_log_upper_bound + 1e-9 >= max(enumerated)


def test_randomized_small_problems_preserve_constraints_and_bound() -> None:
    rng = np.random.default_rng(20260725)
    for _ in range(50):
        n_steps = int(rng.integers(1, 5))
        n_assets = int(rng.integers(1, 4))
        returns = rng.uniform(-0.20, 0.20, size=(n_steps, n_assets))
        max_abs = rng.uniform(0.15, 0.45, size=n_assets)
        max_gross = float(min(1.0, max(0.2, max_abs.sum() * 0.65)))
        max_net = float(rng.uniform(0.0, max_gross))
        transaction = rng.uniform(0.0, 0.01, size=n_assets)
        liquidation = rng.uniform(0.0, 0.01, size=n_assets)
        config = PerfectInformationBoundConfig(
            n_assets=n_assets,
            transaction_cost_rate=tuple(float(value) for value in transaction),
            liquidation_cost_rate=tuple(float(value) for value in liquidation),
            max_abs_weight=tuple(float(value) for value in max_abs),
            max_gross=max_gross,
            max_net_exposure=max_net,
        )

        result = solve_perfect_information_bound(returns, config)

        assert result.target_weights.shape == (n_steps, n_assets)
        assert result.turnover.shape == (n_steps + 1, n_assets)
        assert np.all(np.abs(result.target_weights) <= max_abs[None, :] + 1e-8)
        assert np.all(np.abs(result.target_weights).sum(axis=1) <= max_gross + 1e-8)
        assert np.all(np.abs(result.target_weights.sum(axis=1)) <= max_net + 1e-8)
        assert result.max_primal_violation <= config.feasibility_tolerance
        assert (
            result.selected_path_linearized_objective
            >= result.linearized_log_upper_bound
            - config.lexicographic_objective_tolerance
            - config.feasibility_tolerance
        )
        assert result.replay_log_return <= result.linearized_log_upper_bound + 1e-8
