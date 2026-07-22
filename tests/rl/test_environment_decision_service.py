from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    BaselineResidualComposer,
    ResidualAction,
)
from trade_rl.rl.environment_decision import (
    EnvironmentDecisionPlanner,
    EnvironmentDecisionRequest,
)
from trade_rl.strategies.trend import TrendTargets


class _RegularDataset:
    regular_cadence = True

    @staticmethod
    def bars_for_hours(hours: float) -> int:
        return int(hours)

    @staticmethod
    def bars_until(start: int, hours: float, *, maximum_index: int) -> int:
        del start, hours, maximum_index
        raise AssertionError("regular cadence must not call bars_until")


class _IrregularDataset:
    regular_cadence = False

    @staticmethod
    def bars_for_hours(hours: float) -> int:
        del hours
        raise AssertionError("irregular cadence must not call bars_for_hours")

    @staticmethod
    def bars_until(start: int, hours: float, *, maximum_index: int) -> int:
        assert start == 3
        assert hours == 2.0
        assert maximum_index == 9
        return 4


def _trends() -> TrendTargets:
    return TrendTargets(
        fast=np.array([0.4, -0.1]),
        base=np.array([0.2, -0.1]),
        slow=np.array([0.1, -0.2]),
    )


def _target_planner(*, delay: int = 0) -> EnvironmentDecisionPlanner:
    return EnvironmentDecisionPlanner(
        _RegularDataset(),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=2,
        ),
        composer=BaselineResidualComposer(),
        max_gross=1.0,
        alpha_enabled=False,
        accept_legacy_actions=False,
        signal_delay_decisions=delay,
        decision_every=2,
        decision_hours=2.0,
    )


def _request(
    action: np.ndarray,
    *,
    pending_hybrid: np.ndarray | None = None,
    pending_shadow: np.ndarray | None = None,
) -> EnvironmentDecisionRequest:
    return EnvironmentDecisionRequest(
        action=action,
        trends=_trends(),
        alpha=np.zeros(2),
        factor_basis=np.empty((0, 2)),
        hybrid_weights=np.array([0.05, -0.05]),
        shadow_weights=np.array([0.1, -0.1]),
        pending_hybrid_target=pending_hybrid,
        pending_shadow_target=pending_shadow,
        current_index=3,
        end_index=9,
    )


def test_target_weight_plan_preserves_maintained_action_and_composes_target() -> None:
    plan = _target_planner().plan(_request(np.array([1.4, -0.25])))

    np.testing.assert_allclose(plan.maintained_action, np.array([1.0, -0.25]))
    np.testing.assert_allclose(plan.submitted_hybrid_target, np.array([0.8, -0.2]))
    np.testing.assert_allclose(plan.executed_hybrid_target, np.array([0.8, -0.2]))
    np.testing.assert_allclose(plan.submitted_shadow_target, _trends().base)
    assert plan.saturated_count == 1
    assert plan.raw_max_abs == pytest.approx(1.4)
    assert plan.execution_delay_warmup is False
    assert plan.bars == 2


def test_signal_delay_warmup_executes_current_weights_and_queues_submission() -> None:
    plan = _target_planner(delay=1).plan(_request(np.array([0.3, -0.2])))

    np.testing.assert_allclose(plan.executed_hybrid_target, np.array([0.05, -0.05]))
    np.testing.assert_allclose(plan.executed_shadow_target, np.array([0.1, -0.1]))
    np.testing.assert_allclose(plan.next_pending_hybrid_target, np.array([0.3, -0.2]))
    np.testing.assert_allclose(plan.next_pending_shadow_target, _trends().base)
    assert plan.execution_delay_warmup is True


def test_signal_delay_executes_previous_pending_targets() -> None:
    pending_hybrid = np.array([-0.2, 0.4])
    pending_shadow = np.array([0.05, -0.05])

    plan = _target_planner(delay=1).plan(
        _request(
            np.array([0.3, -0.2]),
            pending_hybrid=pending_hybrid,
            pending_shadow=pending_shadow,
        )
    )

    np.testing.assert_allclose(plan.executed_hybrid_target, pending_hybrid)
    np.testing.assert_allclose(plan.executed_shadow_target, pending_shadow)
    assert plan.execution_delay_warmup is False
    assert plan.executed_hybrid_target is not pending_hybrid
    assert plan.executed_shadow_target is not pending_shadow


def test_legacy_action_is_migrated_to_maintained_layout() -> None:
    planner = EnvironmentDecisionPlanner(
        _RegularDataset(),
        action_spec=ActionSpec(alpha_enabled=True),
        composer=BaselineResidualComposer(),
        max_gross=1.0,
        alpha_enabled=True,
        accept_legacy_actions=True,
        signal_delay_decisions=0,
        decision_every=2,
        decision_hours=2.0,
    )

    plan = planner.plan(
        EnvironmentDecisionRequest(
            action=np.array([0.5, 0.25]),
            trends=_trends(),
            alpha=np.array([0.3, -0.3]),
            factor_basis=np.empty((0, 2)),
            hybrid_weights=np.zeros(2),
            shadow_weights=np.zeros(2),
            pending_hybrid_target=None,
            pending_shadow_target=None,
            current_index=3,
            end_index=9,
        )
    )

    assert isinstance(plan.parsed_action, ResidualAction)
    np.testing.assert_allclose(plan.maintained_action, np.array([0.5, 0.0, 0.0, 0.25]))


def test_irregular_calendar_uses_dataset_bar_boundary() -> None:
    planner = EnvironmentDecisionPlanner(
        _IrregularDataset(),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=2,
        ),
        composer=BaselineResidualComposer(),
        max_gross=1.0,
        alpha_enabled=False,
        accept_legacy_actions=False,
        signal_delay_decisions=0,
        decision_every=None,
        decision_hours=2.0,
    )

    assert planner.decision_bar_count(current_index=3, end_index=9) == 4


def test_decision_bar_count_fails_after_episode_end() -> None:
    with pytest.raises(RuntimeError, match="step called after the episode ended"):
        _target_planner().decision_bar_count(current_index=9, end_index=9)
