from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import (
    ActionPathDiagnostics,
    ConstraintCostRequest,
    calculate_constraint_costs,
)


def test_action_path_diagnostics_measure_every_maintained_stage() -> None:
    diagnostics = ActionPathDiagnostics.from_stages(
        policy_target=np.array([0.6, -0.4]),
        execution_intent_target=np.array([0.55, -0.4]),
        pretrade_target=np.array([0.5, -0.4]),
        feasible_target=np.array([0.45, -0.35]),
        submitted_order_target=np.array([0.45, -0.35]),
        filled_weight=np.array([0.4, -0.3]),
    )

    assert diagnostics.policy_to_execution_intent_l1 == pytest.approx(0.05)
    assert diagnostics.execution_intent_to_pretrade_l1 == pytest.approx(0.05)
    assert diagnostics.policy_to_pretrade_l1 == pytest.approx(0.1)
    assert diagnostics.pretrade_to_feasible_l1 == pytest.approx(0.1)
    assert diagnostics.feasible_to_submitted_l1 == pytest.approx(0.0)
    assert diagnostics.submitted_to_filled_l1 == pytest.approx(0.1)
    assert diagnostics.execution_intent_to_filled_l1 == pytest.approx(0.25)
    assert diagnostics.policy_to_filled_l1 == pytest.approx(0.3)
    assert diagnostics.policy_to_execution_intent_max_abs == pytest.approx(0.05)
    assert diagnostics.execution_intent_to_pretrade_max_abs == pytest.approx(0.05)
    assert diagnostics.policy_to_pretrade_max_abs == pytest.approx(0.1)
    assert diagnostics.pretrade_to_feasible_max_abs == pytest.approx(0.05)
    assert diagnostics.feasible_to_submitted_max_abs == pytest.approx(0.0)
    assert diagnostics.submitted_to_filled_max_abs == pytest.approx(0.05)
    assert diagnostics.policy_changed_by_execution_delay is True
    assert diagnostics.execution_intent_changed_by_pretrade is True
    assert diagnostics.policy_changed_by_pretrade is True
    assert diagnostics.pretrade_changed_by_feasibility is True
    assert diagnostics.feasible_changed_before_submission is False
    assert diagnostics.submission_changed_by_fill is True


def test_action_path_diagnostics_copy_and_freeze_input_vectors() -> None:
    policy = np.array([0.2, -0.1])
    execution_intent = np.array([0.1, -0.1])
    diagnostics = ActionPathDiagnostics.from_stages(
        policy_target=policy,
        execution_intent_target=execution_intent,
        pretrade_target=execution_intent,
        feasible_target=execution_intent,
        submitted_order_target=execution_intent,
        filled_weight=execution_intent,
    )

    policy[:] = 0.0
    execution_intent[:] = 0.0

    np.testing.assert_allclose(diagnostics.policy_target, np.array([0.2, -0.1]))
    np.testing.assert_allclose(
        diagnostics.execution_intent_target,
        np.array([0.1, -0.1]),
    )
    assert diagnostics.policy_target.flags.writeable is False
    assert diagnostics.execution_intent_target.flags.writeable is False
    with pytest.raises(ValueError):
        diagnostics.policy_target[0] = 1.0


def test_action_path_defaults_execution_intent_to_policy_for_legacy_callers() -> None:
    policy = np.array([0.2, -0.1])

    diagnostics = ActionPathDiagnostics.from_stages(
        policy_target=policy,
        pretrade_target=policy,
        feasible_target=policy,
        submitted_order_target=policy,
        filled_weight=policy,
    )

    np.testing.assert_allclose(diagnostics.execution_intent_target, policy)
    assert diagnostics.policy_to_execution_intent_l1 == 0.0
    assert diagnostics.policy_changed_by_execution_delay is False


@pytest.mark.parametrize(
    "stages",
    [
        {
            "policy_target": np.array([0.1, 0.2]),
            "execution_intent_target": np.array([0.1, 0.2]),
            "pretrade_target": np.array([0.1]),
            "feasible_target": np.array([0.1, 0.2]),
            "submitted_order_target": np.array([0.1, 0.2]),
            "filled_weight": np.array([0.1, 0.2]),
        },
        {
            "policy_target": np.array([0.1, 0.2]),
            "execution_intent_target": np.array([0.1]),
            "pretrade_target": np.array([0.1, 0.2]),
            "feasible_target": np.array([0.1, 0.2]),
            "submitted_order_target": np.array([0.1, 0.2]),
            "filled_weight": np.array([0.1, 0.2]),
        },
        {
            "policy_target": np.array([np.nan, 0.2]),
            "execution_intent_target": np.array([0.1, 0.2]),
            "pretrade_target": np.array([0.1, 0.2]),
            "feasible_target": np.array([0.1, 0.2]),
            "submitted_order_target": np.array([0.1, 0.2]),
            "filled_weight": np.array([0.1, 0.2]),
        },
    ],
)
def test_action_path_diagnostics_fail_closed(stages: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        ActionPathDiagnostics.from_stages(**stages)


def test_constraint_costs_use_explicit_units() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.8, -0.6]),
            max_gross=1.0,
            decision_hours=0.25,
            drawdown=0.14,
            drawdown_budget=0.10,
            margin_deficit=250.0,
            initial_capital=100_000.0,
            previous_equity=100_000.0,
            filled_turnover=0.125,
            interval_cost=20.0,
            interval_funding=-5.0,
            interval_borrow_cost=2.0,
            termination_reason="drawdown_stop",
            emergency_deleverage=True,
            liquidation_terminal=False,
            liquidation_complete=True,
        )
    )

    assert costs.drawdown_excess == pytest.approx(0.04)
    assert costs.drawdown_stop_event == pytest.approx(1.0)
    assert costs.margin_deficit_fraction == pytest.approx(0.0025)
    assert costs.forced_liquidation_event == pytest.approx(0.0)
    assert costs.gross_exposure_request_excess == pytest.approx(0.4)
    assert costs.daily_turnover == pytest.approx(12.0)
    assert costs.execution_cost_fraction == pytest.approx(0.00027)
    assert costs.funding_credit_fraction == pytest.approx(0.0)


@pytest.mark.parametrize(
    "termination_reason,emergency_deleverage,liquidation_complete",
    [
        ("minimum_equity", False, True),
        ("execution_cost_exhaustion", False, True),
        ("margin_call", False, True),
        ("liquidation", False, True),
        ("insolvency", False, True),
        ("drawdown_stop", True, False),
    ],
)
def test_constraint_costs_classify_forced_liquidation_events(
    termination_reason: str,
    emergency_deleverage: bool,
    liquidation_complete: bool,
) -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.0]),
            max_gross=1.0,
            decision_hours=1.0,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=0.0,
            initial_capital=100.0,
            previous_equity=100.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            interval_funding=0.0,
            interval_borrow_cost=0.0,
            termination_reason=termination_reason,
            emergency_deleverage=emergency_deleverage,
            liquidation_terminal=False,
            liquidation_complete=liquidation_complete,
        )
    )

    assert costs.forced_liquidation_event == pytest.approx(1.0)


def test_constraint_costs_exclude_routine_terminal_flattening() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.5]),
            max_gross=1.0,
            decision_hours=1.0,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=0.0,
            initial_capital=100.0,
            previous_equity=100.0,
            filled_turnover=0.5,
            interval_cost=1.0,
            interval_funding=0.0,
            interval_borrow_cost=0.0,
            termination_reason="forced_close",
            emergency_deleverage=False,
            liquidation_terminal=True,
            liquidation_complete=True,
        )
    )

    assert costs.forced_liquidation_event == pytest.approx(0.0)


def test_positive_funding_is_reported_as_credit_not_negative_cost() -> None:
    costs = calculate_constraint_costs(
        ConstraintCostRequest(
            policy_target=np.array([0.0]),
            max_gross=1.0,
            decision_hours=1.0,
            drawdown=0.0,
            drawdown_budget=0.1,
            margin_deficit=0.0,
            initial_capital=100.0,
            previous_equity=100.0,
            filled_turnover=0.0,
            interval_cost=-0.25,
            interval_funding=1.5,
            interval_borrow_cost=0.0,
            termination_reason=None,
            emergency_deleverage=False,
            liquidation_terminal=False,
            liquidation_complete=True,
        )
    )

    assert costs.execution_cost_fraction == pytest.approx(0.0)
    assert costs.funding_credit_fraction == pytest.approx(0.015)


@pytest.mark.parametrize(
    "request_update, message",
    [
        ({"max_gross": 0.0}, "max_gross"),
        ({"decision_hours": 0.0}, "decision_hours"),
        ({"drawdown": 1.1}, "drawdown"),
        ({"drawdown_budget": 1.1}, "drawdown_budget"),
        ({"margin_deficit": -1.0}, "margin_deficit"),
        ({"initial_capital": 0.0}, "initial_capital"),
        ({"previous_equity": 0.0}, "previous_equity"),
        ({"filled_turnover": -0.1}, "filled_turnover"),
        ({"interval_cost": np.nan}, "interval_cost"),
        ({"interval_funding": np.inf}, "interval_funding"),
        ({"interval_borrow_cost": -0.1}, "interval_borrow_cost"),
    ],
)
def test_constraint_cost_request_fails_closed(
    request_update: dict[str, float],
    message: str,
) -> None:
    values: dict[str, object] = {
        "policy_target": np.array([0.0]),
        "max_gross": 1.0,
        "decision_hours": 1.0,
        "drawdown": 0.0,
        "drawdown_budget": 0.1,
        "margin_deficit": 0.0,
        "initial_capital": 100.0,
        "previous_equity": 100.0,
        "filled_turnover": 0.0,
        "interval_cost": 0.0,
        "interval_funding": 0.0,
        "interval_borrow_cost": 0.0,
        "termination_reason": None,
        "emergency_deleverage": False,
        "liquidation_terminal": False,
        "liquidation_complete": True,
    }
    values.update(request_update)

    with pytest.raises(ValueError, match=message):
        ConstraintCostRequest(**values)  # type: ignore[arg-type]
