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


def test_step_info_derives_action_path_and_costs_from_causal_transition_state() -> None:
    hybrid = BookState.zero(2, 100.0, np.array([10.0, 20.0]))
    hybrid.cash = 95.0
    execution = SimpleNamespace(
        next_index=2,
        bars_advanced=2,
        interval_cost=0.5,
        interval_funding=-0.1,
        interval_borrow_cost=0.05,
        interval_gross_return=-0.04,
        interval_net_return=-0.05,
        filled_turnover=0.2,
    )
    risk = SimpleNamespace(
        projection_l1=1.1,
        proposal_weights=np.array([0.8, -0.6]),
        pretrade_weights=np.array([0.5, -0.4]),
        weights=np.array([0.4, -0.3]),
        max_gross=1.0,
        drawdown_budget=0.10,
    )
    shadow_execution = SimpleNamespace(
        next_index=2,
        bars_advanced=2,
        interval_cost=0.0,
        interval_funding=0.0,
        interval_borrow_cost=0.0,
        interval_gross_return=0.0,
        interval_net_return=0.0,
        filled_turnover=0.0,
    )

    info = EnvironmentInfoBuilder(
        _Dataset(),
        _RewardTracker(),
        initial_capital=100.0,
    ).step_info(
        EnvironmentStepInfoRequest(
            action_delta_l1=0.0,
            raw_max_abs=0.8,
            saturated_count=0,
            composition=object(),
            decision_step_index=1,
            hybrid_log_return=math.log(0.95),
            shadow_log_return=0.0,
            emergency_deleverage=False,
            execution_delay_warmup=False,
            submitted_target=np.array([0.8, -0.6]),
            executed_target=np.array([0.8, -0.6]),
            hybrid=hybrid,
            reward_breakdown=_reward(),
            hybrid_execution=execution,
            hybrid_risk=risk,
            hybrid_terminated=False,
            shadow_execution=shadow_execution,
            shadow_risk=SimpleNamespace(projection_l1=0.0),
            shadow_terminated=False,
            liquidation_complete=True,
            liquidation_terminal=False,
            termination_reason=None,
            terminal_accounting_mode="mark_to_market",
            terminal_liquidation_cost=0.0,
            pending_target_discarded=False,
            discarded_pending_target=None,
            hybrid_liquidation=None,
            shadow_liquidation=None,
        )
    )

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
