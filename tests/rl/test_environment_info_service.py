from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.rl.environment_info import (
    EnvironmentInfoBuilder,
    EnvironmentStepInfoRequest,
    EnvironmentTerminalInfoRequest,
)
from trade_rl.rl.rewards import RewardBreakdown, RewardConfig, RewardContext
from trade_rl.simulation.accounting import BookState


class _Dataset:
    periods_per_year = 365


class _RewardTracker:
    config = RewardConfig(baseline_underperformance_weight=0.1)
    last_context_before = RewardContext(
        rolling_hybrid_log_growth=0.01,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.01,
        baseline_tolerance=0.015,
        baseline_penalty=0.0,
        hybrid_drawdown=0.1,
        drawdown_severity=0.05,
        history_bars=3,
    )
    last_context_after = RewardContext(
        rolling_hybrid_log_growth=0.03,
        rolling_shadow_log_growth=0.025,
        baseline_shortfall=0.0,
        baseline_tolerance=0.015,
        baseline_penalty=0.0,
        hybrid_drawdown=0.08,
        drawdown_severity=0.03,
        history_bars=4,
    )


def _reward() -> RewardBreakdown:
    return RewardBreakdown(
        absolute_log_growth=0.02,
        excess_log_growth=0.01,
        incremental_drawdown=0.03,
        rolling_baseline_underperformance=0.0,
        projection_distance=0.4,
        terminal_equity_shortfall=0.0,
        margin_deficit=0.0,
        absolute_component=0.02,
        excess_component=0.0,
        drawdown_penalty=0.004,
        baseline_penalty=0.002,
        projection_penalty=0.0,
        terminal_penalty=0.0,
        margin_penalty=0.0,
        unscaled_total=0.014,
        scaled_total=1.4,
    )


def _book(*, cash: float = 100.0) -> BookState:
    book = BookState.zero(2, 100.0, np.array([10.0, 20.0]))
    book.cash = cash
    return book


def _execution() -> SimpleNamespace:
    return SimpleNamespace(
        bars_advanced=2,
        interval_cost=0.5,
        interval_funding=-0.1,
        interval_gross_return=0.03,
        interval_net_return=0.02,
    )


def _step_request(**overrides: object) -> EnvironmentStepInfoRequest:
    values: dict[str, object] = {
        "action_delta_l1": 0.3,
        "raw_max_abs": 1.2,
        "saturated_count": 1,
        "composition": SimpleNamespace(proposal=np.array([0.4, -0.2])),
        "decision_step_index": 4,
        "hybrid_log_return": 0.02,
        "shadow_log_return": 0.01,
        "emergency_deleverage": False,
        "execution_delay_warmup": True,
        "submitted_target": np.array([0.4, -0.2]),
        "executed_target": np.array([0.1, -0.1]),
        "hybrid": _book(cash=95.0),
        "reward_breakdown": _reward(),
        "hybrid_execution": _execution(),
        "hybrid_risk": SimpleNamespace(projection_l1=0.4),
        "hybrid_terminated": False,
        "shadow_execution": _execution(),
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


def test_step_info_preserves_complete_stable_key_set() -> None:
    builder = EnvironmentInfoBuilder(_Dataset(), _RewardTracker())

    info = builder.step_info(_step_request())

    assert set(info) == {
        "action_delta_l1",
        "action_raw_max_abs",
        "action_saturated_count",
        "bars_advanced",
        "composition",
        "decision_step_index",
        "excess_log_return",
        "emergency_deleverage",
        "execution_delay_warmup",
        "submitted_target",
        "executed_target",
        "drawdown_after",
        "portfolio_value_after",
        "reward_growth_raw",
        "reward_baseline_penalty_delta",
        "reward_baseline_penalty_weighted",
        "reward_drawdown_penalty_delta",
        "reward_drawdown_penalty_weighted",
        "reward_total_raw",
        "reward_total_scaled",
        "reward_context_before",
        "reward_context_after",
        "rolling_hybrid_log_growth",
        "rolling_baseline_log_growth",
        "rolling_growth_gap",
        "hybrid_execution",
        "hybrid_risk",
        "hybrid_terminated",
        "interval_cost",
        "interval_funding",
        "interval_gross_return",
        "interval_net_return",
        "liquidation_complete",
        "liquidation_terminal",
        "projection_distance_l1",
        "reward_breakdown",
        "shadow_execution",
        "shadow_interval_net_return",
        "shadow_risk",
        "shadow_terminated",
        "termination_reason",
        "terminal_accounting_mode",
        "terminal_liquidation_cost",
        "pending_target_discarded",
    }
    assert info["reward_baseline_penalty_delta"] == 0.02
    assert info["rolling_growth_gap"] == pytest.approx(0.005)


def test_step_info_copies_targets_and_adds_optional_fields() -> None:
    submitted = np.array([0.4, -0.2])
    executed = np.array([0.1, -0.1])
    discarded = np.array([0.2, 0.0])
    hybrid_liquidation = object()
    shadow_liquidation = object()
    builder = EnvironmentInfoBuilder(_Dataset(), _RewardTracker())

    info = builder.step_info(
        _step_request(
            submitted_target=submitted,
            executed_target=executed,
            discarded_pending_target=discarded,
            hybrid_liquidation=hybrid_liquidation,
            shadow_liquidation=shadow_liquidation,
        )
    )

    np.testing.assert_allclose(info["submitted_target"], submitted)
    np.testing.assert_allclose(info["executed_target"], executed)
    np.testing.assert_allclose(info["discarded_pending_target"], discarded)
    assert info["submitted_target"] is not submitted
    assert info["executed_target"] is not executed
    assert info["discarded_pending_target"] is not discarded
    assert info["hybrid_liquidation"] is hybrid_liquidation
    assert info["shadow_liquidation"] is shadow_liquidation


def test_terminal_info_builds_metrics_and_diagnostics() -> None:
    hybrid = _book()
    shadow = _book()
    hybrid.returns_history.extend([0.01, -0.005])
    shadow.returns_history.extend([0.005, -0.002])
    hybrid.turnover_total = 0.4
    shadow.turnover_total = 0.2
    hybrid.total_cost = 0.01
    shadow.total_cost = 0.005
    diagnostics = object()
    builder = EnvironmentInfoBuilder(_Dataset(), _RewardTracker())

    info = builder.terminal_info(
        EnvironmentTerminalInfoRequest(
            episode_hours=48.0,
            episode_seed=7,
            action_diagnostics=diagnostics,
            hybrid=hybrid,
            shadow=shadow,
            initial_state_mode="cash",
        )
    )

    assert info["episode_hours"] == 48.0
    assert info["episode_seed"] == 7
    assert info["action_diagnostics"] is diagnostics
    assert info["initial_state_mode"] == "cash"
    assert info["hybrid_metrics"].turnover_total == 0.4
    assert info["shadow_metrics"].turnover_total == 0.2
    assert info["excess_total_return"] == (
        info["hybrid_metrics"].total_return - info["shadow_metrics"].total_return
    )
