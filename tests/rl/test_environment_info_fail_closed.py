from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from trade_rl.rl.environment_info import (
    EnvironmentInfoBuilder,
    EnvironmentStepInfoRequest,
)


class _Dataset:
    periods_per_year = 365

    @staticmethod
    def elapsed_hours(start_index: int, end_index: int) -> float:
        del start_index, end_index
        return 0.0


def _builder(*, initial_capital: float | None = 100.0) -> EnvironmentInfoBuilder:
    return EnvironmentInfoBuilder(
        cast(Any, _Dataset()),
        cast(Any, object()),
        initial_capital=initial_capital,
    )


def _request(
    *,
    max_gross: float | None = 1.0,
    drawdown_budget: float | None = 0.1,
    hybrid_log_return: float = 0.0,
) -> EnvironmentStepInfoRequest:
    target = np.array([0.1], dtype=np.float64)
    return cast(
        EnvironmentStepInfoRequest,
        SimpleNamespace(
            submitted_target=target,
            executed_target=target,
            hybrid_risk=SimpleNamespace(
                proposal_weights=target,
                pretrade_weights=target,
                weights=target,
                max_gross=max_gross,
                drawdown_budget=drawdown_budget,
            ),
            hybrid=SimpleNamespace(
                portfolio_value=100.0,
                margin_deficit=0.0,
            ),
            hybrid_log_return=hybrid_log_return,
        ),
    )


def test_info_builder_rejects_invalid_initial_capital() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        _builder(initial_capital=0.0)


def test_info_builder_rejects_invalid_transition_duration() -> None:
    execution = cast(
        Any,
        SimpleNamespace(next_index=2, bars_advanced=2),
    )

    with pytest.raises(RuntimeError, match="transition duration"):
        _builder()._decision_hours(execution)  # noqa: SLF001


def test_info_builder_rejects_nonfinite_liquidation_metric() -> None:
    liquidation = SimpleNamespace(interval_cost=np.nan)

    with pytest.raises(RuntimeError, match="liquidation interval_cost"):
        EnvironmentInfoBuilder._liquidation_metric(  # noqa: SLF001
            liquidation,
            "interval_cost",
        )


def test_info_builder_rejects_invalid_target_vector() -> None:
    with pytest.raises(RuntimeError, match="policy target"):
        EnvironmentInfoBuilder._target_vector(  # noqa: SLF001
            np.array([np.nan]),
            field_name="policy target",
        )


def test_constraint_cost_derivation_requires_initial_capital() -> None:
    with pytest.raises(RuntimeError, match="initial capital"):
        _builder(initial_capital=None)._derived_constraint_costs(  # noqa: SLF001
            _request()
        )


def test_constraint_cost_derivation_requires_risk_limits() -> None:
    with pytest.raises(RuntimeError, match="constraint-limit metadata"):
        _builder()._derived_constraint_costs(  # noqa: SLF001
            _request(max_gross=None)
        )


def test_constraint_cost_derivation_rejects_invalid_previous_equity() -> None:
    with pytest.raises(RuntimeError, match="previous equity"):
        _builder()._derived_constraint_costs(  # noqa: SLF001
            _request(hybrid_log_return=float("inf"))
        )
