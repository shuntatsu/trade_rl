from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.evaluation import _perfect_information_lp
from trade_rl.evaluation import perfect_information_bound as bound_module
from trade_rl.evaluation._perfect_information_lp import LinearProgramSolution
from trade_rl.evaluation.perfect_information_bound import (
    PerfectInformationBoundConfig,
    solve_perfect_information_bound,
)


def test_solver_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = _perfect_information_lp.importlib.import_module

    def missing_scipy(name: str) -> object:
        if name.startswith("scipy"):
            raise ImportError("scipy intentionally unavailable")
        return real_import(name)

    monkeypatch.setattr(
        _perfect_information_lp.importlib,
        "import_module",
        missing_scipy,
    )
    with pytest.raises(RuntimeError, match="optional 'oracle' dependency"):
        solve_perfect_information_bound(
            np.asarray([[0.01]], dtype=np.float64),
            PerfectInformationBoundConfig(n_assets=1),
        )


def _linear_solution(
    weights: np.ndarray,
    *,
    upper_bound: float,
    selected_objective: float,
) -> LinearProgramSolution:
    return LinearProgramSolution(
        target_weights=weights,
        linearized_upper_bound=upper_bound,
        selected_linearized_objective=selected_objective,
        primary_status=0,
        primary_message="optimal",
        primary_iterations=1,
        secondary_status=0,
        secondary_message="optimal",
        secondary_iterations=1,
    )


def test_replay_rejects_non_positive_wealth_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bound_module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: _linear_solution(
            np.asarray([[0.45]], dtype=np.float64),
            upper_bound=0.0,
            selected_objective=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="non-positive wealth factor"):
        solve_perfect_information_bound(
            np.asarray([[0.0]], dtype=np.float64),
            PerfectInformationBoundConfig(
                n_assets=1,
                transaction_cost_rate=3.0,
                max_gross=0.45,
                max_abs_weight=0.45,
            ),
        )


def test_replay_rejects_lp_objective_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bound_module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: _linear_solution(
            np.zeros((1, 1), dtype=np.float64),
            upper_bound=1.0,
            selected_objective=1.0,
        ),
    )
    with pytest.raises(RuntimeError, match="disagrees"):
        solve_perfect_information_bound(
            np.asarray([[0.0]], dtype=np.float64),
            PerfectInformationBoundConfig(n_assets=1),
        )


def test_replay_rejects_secondary_objective_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bound_module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: _linear_solution(
            np.zeros((1, 1), dtype=np.float64),
            upper_bound=1.0,
            selected_objective=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="primary objective tolerance"):
        solve_perfect_information_bound(
            np.asarray([[0.0]], dtype=np.float64),
            PerfectInformationBoundConfig(n_assets=1),
        )


def test_replay_rejects_upper_bound_below_log_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bound_module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: _linear_solution(
            np.asarray([[0.45]], dtype=np.float64),
            upper_bound=0.0,
            selected_objective=0.045,
        ),
    )
    with pytest.raises(RuntimeError, match="below exact replay"):
        solve_perfect_information_bound(
            np.asarray([[0.1]], dtype=np.float64),
            PerfectInformationBoundConfig(
                n_assets=1,
                lexicographic_objective_tolerance=0.1,
            ),
        )


def test_replay_rejects_constraint_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bound_module,
        "solve_lexicographic_linear_program",
        lambda *_args, **_kwargs: _linear_solution(
            np.asarray([[0.9]], dtype=np.float64),
            upper_bound=0.0,
            selected_objective=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="violates declared constraints"):
        solve_perfect_information_bound(
            np.asarray([[0.0]], dtype=np.float64),
            PerfectInformationBoundConfig(n_assets=1),
        )


def test_private_solver_rejects_invalid_primary_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sparse = __import__("scipy.sparse", fromlist=["sparse"])
    fake_optimize = SimpleNamespace(
        linprog=lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="optimal",
            x=np.zeros(3, dtype=np.float64),
            fun=0.0,
            nit=0,
        )
    )
    monkeypatch.setattr(
        _perfect_information_lp,
        "_scipy_modules",
        lambda: (fake_optimize, real_sparse),
    )
    with pytest.raises(RuntimeError, match="primary solver returned invalid"):
        _perfect_information_lp.solve_lexicographic_linear_program(
            np.zeros((1, 1), dtype=np.float64),
            transaction_cost_rate=np.zeros(1),
            liquidation_cost_rate=np.zeros(1),
            max_abs_weight=np.asarray([0.45]),
            max_gross=1.0,
            max_net_exposure=None,
            initial_weights=np.zeros(1),
            minimum_period_net_return=-0.999,
            objective_tolerance=0.0,
            feasibility_tolerance=1e-8,
            solver_method="highs",
        )


def test_private_solver_rejects_invalid_secondary_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sparse = __import__("scipy.sparse", fromlist=["sparse"])
    responses = iter(
        [
            SimpleNamespace(
                success=True,
                status=0,
                message="optimal",
                x=np.zeros(4, dtype=np.float64),
                fun=0.0,
                nit=0,
            ),
            SimpleNamespace(
                success=True,
                status=0,
                message="optimal",
                x=np.zeros(3, dtype=np.float64),
                fun=0.0,
                nit=0,
            ),
        ]
    )
    fake_optimize = SimpleNamespace(
        linprog=lambda *_args, **_kwargs: next(responses)
    )
    monkeypatch.setattr(
        _perfect_information_lp,
        "_scipy_modules",
        lambda: (fake_optimize, real_sparse),
    )
    with pytest.raises(RuntimeError, match="secondary solver returned invalid"):
        _perfect_information_lp.solve_lexicographic_linear_program(
            np.zeros((1, 1), dtype=np.float64),
            transaction_cost_rate=np.zeros(1),
            liquidation_cost_rate=np.zeros(1),
            max_abs_weight=np.asarray([0.45]),
            max_gross=1.0,
            max_net_exposure=None,
            initial_weights=np.zeros(1),
            minimum_period_net_return=-0.999,
            objective_tolerance=0.0,
            feasibility_tolerance=1e-8,
            solver_method="highs",
        )
