from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from trade_rl.evaluation.perfect_information_bound import (
    PERFECT_INFORMATION_BOUND_SCHEMA,
    PerfectInformationBoundConfig,
    PerfectInformationBoundResult,
    solve_perfect_information_bound,
)


def test_perfect_information_bound_module_exists() -> None:
    assert importlib.util.find_spec(
        "trade_rl.evaluation.perfect_information_bound"
    ) is not None


def test_config_normalizes_scalar_asset_parameters() -> None:
    config = PerfectInformationBoundConfig(
        n_assets=3,
        transaction_cost_rate=0.001,
        liquidation_cost_rate=0.002,
        max_abs_weight=0.45,
    )

    assert PERFECT_INFORMATION_BOUND_SCHEMA == "perfect_information_linear_bound_v1"
    assert config.transaction_cost_rate == (0.001, 0.001, 0.001)
    assert config.liquidation_cost_rate == (0.002, 0.002, 0.002)
    assert config.max_abs_weight == (0.45, 0.45, 0.45)
    assert config.initial_weights == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_assets": 0}, "n_assets"),
        ({"n_assets": 1, "transaction_cost_rate": -0.1}, "transaction_cost_rate"),
        ({"n_assets": 1, "transaction_cost_rate": math.inf}, "transaction_cost_rate"),
        ({"n_assets": 1, "transaction_cost_rate": {}}, "transaction_cost_rate"),
        ({"n_assets": 1, "liquidation_cost_rate": -0.1}, "liquidation_cost_rate"),
        ({"n_assets": 1, "max_abs_weight": 0.0}, "max_abs_weight"),
        ({"n_assets": 2, "max_abs_weight": (0.4,)}, "max_abs_weight"),
        ({"n_assets": 1, "max_gross": 0.0}, "max_gross"),
        ({"n_assets": 1, "max_gross": True}, "max_gross"),
        ({"n_assets": 1, "max_net_exposure": -0.1}, "max_net_exposure"),
        (
            {"n_assets": 1, "minimum_period_net_return": -1.0},
            "minimum_period_net_return",
        ),
        (
            {"n_assets": 1, "lexicographic_objective_tolerance": -1.0},
            "lexicographic_objective_tolerance",
        ),
        ({"n_assets": 1, "feasibility_tolerance": 0.0}, "feasibility_tolerance"),
        ({"n_assets": 1, "solver_method": "simplex"}, "solver_method"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PerfectInformationBoundConfig(**kwargs)


def test_config_rejects_initial_weights_outside_constraints() -> None:
    with pytest.raises(ValueError, match="initial_weights"):
        PerfectInformationBoundConfig(
            n_assets=2,
            max_abs_weight=(0.4, 0.4),
            max_gross=0.5,
            max_net_exposure=0.5,
            initial_weights=(0.4, 0.4),
        )


def test_result_arrays_are_read_only() -> None:
    result = solve_perfect_information_bound(
        np.asarray([[0.02]], dtype=np.float64),
        PerfectInformationBoundConfig(n_assets=1),
    )

    for value in (
        result.target_weights,
        result.absolute_weights,
        result.turnover,
        result.period_gross_returns,
        result.period_transaction_costs,
        result.period_net_returns,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        result.target_weights[0, 0] = 0.0


def test_repeated_solves_are_deterministic_and_digest_stable() -> None:
    returns = np.asarray([[0.03, -0.01], [-0.02, 0.04]], dtype=np.float64)
    config = PerfectInformationBoundConfig(
        n_assets=2,
        transaction_cost_rate=0.001,
        max_abs_weight=0.4,
        max_gross=0.6,
        max_net_exposure=0.3,
    )

    first = solve_perfect_information_bound(returns, config)
    second = solve_perfect_information_bound(returns.copy(), config)

    np.testing.assert_array_equal(first.target_weights, second.target_weights)
    np.testing.assert_array_equal(first.turnover, second.turnover)
    assert first.digest == second.digest
    assert config.digest == PerfectInformationBoundConfig(
        n_assets=2,
        transaction_cost_rate=(0.001, 0.001),
        max_abs_weight=(0.4, 0.4),
        max_gross=0.6,
        max_net_exposure=0.3,
    ).digest


def test_result_digest_changes_when_problem_changes() -> None:
    config = PerfectInformationBoundConfig(n_assets=1)
    first = solve_perfect_information_bound(np.asarray([[0.01]]), config)
    second = solve_perfect_information_bound(np.asarray([[0.02]]), config)

    assert first.digest != second.digest


def test_evaluation_package_exports_perfect_information_bound_api() -> None:
    from trade_rl.evaluation import (
        PERFECT_INFORMATION_BOUND_SCHEMA as exported_schema,
    )
    from trade_rl.evaluation import (
        PerfectInformationBoundConfig as ExportedConfig,
    )
    from trade_rl.evaluation import (
        PerfectInformationBoundResult as ExportedResult,
    )
    from trade_rl.evaluation import (
        solve_perfect_information_bound as exported_solve,
    )

    assert exported_schema == PERFECT_INFORMATION_BOUND_SCHEMA
    assert ExportedConfig is PerfectInformationBoundConfig
    assert ExportedResult is PerfectInformationBoundResult
    assert exported_solve is solve_perfect_information_bound


def test_default_net_limit_is_redundant_with_gross_limit() -> None:
    config = PerfectInformationBoundConfig(n_assets=1, max_gross=0.4)

    assert config.max_net_exposure is None


def test_config_rejects_initial_weight_above_asset_limit() -> None:
    with pytest.raises(ValueError, match="initial_weights"):
        PerfectInformationBoundConfig(
            n_assets=1,
            max_abs_weight=0.2,
            initial_weights=(0.3,),
        )


def test_config_rejects_initial_weight_above_net_limit() -> None:
    with pytest.raises(ValueError, match="initial_weights"):
        PerfectInformationBoundConfig(
            n_assets=2,
            max_abs_weight=0.4,
            max_gross=0.8,
            max_net_exposure=0.1,
            initial_weights=(0.2, 0.0),
        )


def test_result_normalizes_signed_zero_for_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation import perfect_information_bound as module
    from trade_rl.evaluation._perfect_information_lp import LinearProgramSolution

    def solution(weight: float, objective: float) -> LinearProgramSolution:
        return LinearProgramSolution(
            target_weights=np.asarray([[weight]], dtype=np.float64),
            linearized_upper_bound=objective,
            selected_linearized_objective=objective,
            primary_status=0,
            primary_message="optimal",
            primary_iterations=0,
            secondary_status=0,
            secondary_message="optimal",
            secondary_iterations=0,
        )

    monkeypatch.setattr(
        module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: solution(-0.0, -0.0),
    )
    negative = solve_perfect_information_bound(
        np.asarray([[0.0]], dtype=np.float64),
        PerfectInformationBoundConfig(n_assets=1),
    )
    monkeypatch.setattr(
        module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: solution(0.0, 0.0),
    )
    positive = solve_perfect_information_bound(
        np.asarray([[0.0]], dtype=np.float64),
        PerfectInformationBoundConfig(n_assets=1),
    )

    assert math.copysign(1.0, negative.target_weights[0, 0]) == 1.0
    assert negative.digest == positive.digest


def test_large_log_return_omits_unrepresentable_simple_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation import perfect_information_bound as module
    from trade_rl.evaluation._perfect_information_lp import LinearProgramSolution

    steps = 2_000
    weights = np.full((steps, 1), 0.45, dtype=np.float64)
    objective = float(steps * 0.45)
    monkeypatch.setattr(
        module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: LinearProgramSolution(
            target_weights=weights,
            linearized_upper_bound=objective,
            selected_linearized_objective=objective,
            primary_status=0,
            primary_message="optimal",
            primary_iterations=0,
            secondary_status=0,
            secondary_message="optimal",
            secondary_iterations=0,
        ),
    )

    result = solve_perfect_information_bound(
        np.ones((steps, 1), dtype=np.float64),
        PerfectInformationBoundConfig(
            n_assets=1,
            max_abs_weight=0.45,
            max_gross=0.45,
        ),
    )

    assert result.replay_log_return > math.log(np.finfo(np.float64).max)
    assert result.replay_total_return is None
