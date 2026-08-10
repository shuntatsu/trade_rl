from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.rl.environment_info import (
    EnvironmentInfoBuilder,
    EnvironmentStepInfoRequest,
)
from trade_rl.rl.rewards import RewardBreakdown, RewardConfig, RewardContext
from trade_rl.simulation.accounting import BookState


class _Dataset:
    periods_per_year = 8_760

    @staticmethod
    def elapsed_hours(start_index: int, end_index: int) -> float:
        return float(end_index - start_index)


class _InvalidDurationDataset:
    periods_per_year = 8_760

    @staticmethod
    def elapsed_hours(start_index: int, end_index: int) -> float:
        del start_index, end_index
        return 0.0


class _RewardTracker:
    config = RewardConfig()
    last_context_before = RewardContext(
        rolling_hybrid_log_growth=0.0,
        rolling_shadow_log_growth=0.0,
        baseline_shortfall=0.0,
        baseline_tolerance=0.015,
        baseline_penalty=0.0,
        hybrid_drawdown=0.0,
        drawdown_severity=0.0,
        history_bars=0,
    )
    last_context_after = last_context_before


def _reward() -> RewardBreakdown:
    return RewardBreakdown(
        absolute_log_growth=math.log(0.95),
        excess_log_growth=0.0,
        incremental_drawdown=0.05,
        rolling_baseline_underperformance=0.0,
        projection_distance=1.1,
        terminal_equity_shortfall=0.0,
        margin_deficit=0.0,
        absolute_component=math.log(0.95),
        excess_component=0.0,
        drawdown_penalty=0.0,
        baseline_penalty=0.0,
        projection_penalty=0.0,
        terminal_penalty=0.0,
        margin_penalty=0.0,
        unscaled_total=math.log(0.95),
        scaled_total=100.0 * math.log(0.95),
    )


def _book() -> BookState:
    hybrid = BookState.zero(2, 100.0, np.array([10.0, 20.0]))
    hybrid.cash = 95.0
    return hybrid


def _execution(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "next_index": 2,
        "bars_advanced": 2,
        "interval_cost": 0.5,
        "interval_funding": -0.1,
        "interval_borrow_cost": 0.05,
        "interval_gross_return": -0.04,
        "interval_net_return": -0.05,
        "filled_turnover": 0.2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _risk(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "projection_l1": 1.1,
        "proposal_weights": np.array([0.8, -0.6]),
        "pretrade_weights": np.array([0.5, -0.4]),
        "weights": np.array([0.4, -0.3]),
        "max_gross": 1.0,
        "drawdown_budget": 0.10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _request(**overrides: object) -> EnvironmentStepInfoRequest:
    values: dict[str, object] = {
        "action_delta_l1": 0.0,
        "raw_max_abs": 0.8,
        "saturated_count": 0,
        "composition": object(),
        "decision_step_index": 1,
        "hybrid_log_return": math.log(0.95),
        "shadow_log_return": 0.0,
        "emergency_deleverage": False,
        "execution_delay_warmup": False,
        "submitted_target": np.array([0.8, -0.6]),
        "executed_target": np.array([0.8, -0.6]),
        "hybrid": _book(),
        "shadow": _book(),
        "reward_breakdown": _reward(),
        "hybrid_execution": _execution(),
        "hybrid_risk": _risk(),
        "hybrid_terminated": False,
        "shadow_execution": _execution(
            interval_cost=0.0,
            interval_funding=0.0,
            interval_borrow_cost=0.0,
            interval_gross_return=0.0,
            interval_net_return=0.0,
            filled_turnover=0.0,
        ),
        "shadow_risk": SimpleNamespace(projection_l1=0.0),
        "shadow_terminated": False,
        "liquidation_complete": True,
        "liquidation_terminal": False,
        "termination_reason": None,
        "terminal_accounting_mode": "mark_to_market",
        "terminal_liquidation_cost": 0.0,
        "pending_target_discarded": False,
        "discarded_pending_target": None,
        "hybrid_liquidation": None,
        "shadow_liquidation": None,
    }
    values.update(overrides)
    return EnvironmentStepInfoRequest(**values)  # type: ignore[arg-type]


def test_step_info_derives_action_path_and_costs_from_causal_transition_state() -> None:
    info = EnvironmentInfoBuilder(
        _Dataset(),
        _RewardTracker(),
        initial_capital=100.0,
    ).step_info(_request())

    action_path = info["action_path"]
    np.testing.assert_allclose(action_path.policy_target, [0.8, -0.6])
    np.testing.assert_allclose(action_path.execution_intent_target, [0.8, -0.6])
    np.testing.assert_allclose(action_path.pretrade_target, [0.5, -0.4])
    np.testing.assert_allclose(action_path.feasible_target, [0.4, -0.3])
    np.testing.assert_allclose(action_path.submitted_order_target, [0.4, -0.3])
    np.testing.assert_allclose(action_path.filled_weight, [0.0, 0.0])
    assert info["action_path_policy_to_execution_intent_l1"] == 0.0
    assert info["action_path_policy_to_filled_l1"] == pytest.approx(1.4)

    costs = info["constraint_costs"]
    assert costs.drawdown_excess == pytest.approx(0.0)
    assert costs.gross_exposure_request_excess == pytest.approx(0.4)
    assert costs.daily_turnover == pytest.approx(2.4)
    assert costs.execution_cost_fraction == pytest.approx(0.0065)
    assert costs.funding_credit_fraction == pytest.approx(0.0)
    assert costs.drawdown_stop_event == 0.0
    assert costs.forced_liquidation_event == 0.0


@pytest.mark.parametrize("initial_capital", [0.0, np.nan])
def test_info_builder_rejects_invalid_initial_capital(initial_capital: float) -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        EnvironmentInfoBuilder(
            _Dataset(),
            _RewardTracker(),
            initial_capital=initial_capital,
        )


def test_decision_hours_rejects_non_positive_transition_duration() -> None:
    builder = EnvironmentInfoBuilder(
        _InvalidDurationDataset(),
        _RewardTracker(),
        initial_capital=100.0,
    )

    with pytest.raises(RuntimeError, match="transition duration"):
        builder._decision_hours(_execution())


def test_liquidation_metric_rejects_non_finite_values() -> None:
    with pytest.raises(RuntimeError, match="interval_cost"):
        EnvironmentInfoBuilder._liquidation_metric(
            SimpleNamespace(interval_cost=np.nan),
            "interval_cost",
        )


@pytest.mark.parametrize("target", [np.array([]), np.array([np.nan, 0.0])])
def test_target_vector_rejects_empty_or_non_finite_values(target: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="policy target"):
        EnvironmentInfoBuilder._target_vector(target, field_name="policy target")


def test_risk_pipeline_rejects_missing_stage_metadata() -> None:
    request = _request(hybrid_risk=_risk(proposal_weights=None))

    with pytest.raises(RuntimeError, match="action-path stage metadata"):
        EnvironmentInfoBuilder._risk_pipeline_targets(request)


def test_risk_pipeline_rejects_execution_intent_disagreement() -> None:
    request = _request(executed_target=np.array([0.2, -0.1]))

    with pytest.raises(RuntimeError, match="disagrees with risk proposal"):
        EnvironmentInfoBuilder._risk_pipeline_targets(request)


def test_constraint_derivation_requires_initial_capital() -> None:
    builder = EnvironmentInfoBuilder(_Dataset(), _RewardTracker())

    with pytest.raises(RuntimeError, match="configured initial capital"):
        builder._derived_constraint_costs(_request())


def test_constraint_derivation_requires_limit_metadata() -> None:
    builder = EnvironmentInfoBuilder(
        _Dataset(),
        _RewardTracker(),
        initial_capital=100.0,
    )
    request = _request(hybrid_risk=_risk(max_gross=None))

    with pytest.raises(RuntimeError, match="constraint-limit metadata"):
        builder._derived_constraint_costs(request)


def test_constraint_derivation_rejects_non_finite_previous_equity() -> None:
    builder = EnvironmentInfoBuilder(
        _Dataset(),
        _RewardTracker(),
        initial_capital=100.0,
    )
    request = _request(hybrid_log_return=np.nan)

    with pytest.raises(RuntimeError, match="previous equity"):
        builder._derived_constraint_costs(request)
