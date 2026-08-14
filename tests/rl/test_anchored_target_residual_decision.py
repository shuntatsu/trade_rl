from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.actions import ActionMode, ActionSpec, BaselineResidualComposer
from trade_rl.rl.environment_decision import (
    EnvironmentDecisionPlanner,
    EnvironmentDecisionRequest,
)
from trade_rl.strategies.trend import TrendTargets


class _Calendar:
    regular_cadence = True

    def bars_for_hours(self, hours: float) -> int:
        assert hours == pytest.approx(1.0)
        return 1

    def bars_until(self, start: int, hours: float, *, maximum_index: int) -> int:
        del start, hours, maximum_index
        raise AssertionError("regular cadence must use bars_for_hours")


def _planner() -> EnvironmentDecisionPlanner:
    return EnvironmentDecisionPlanner(
        _Calendar(),
        action_spec=ActionSpec(
            mode=ActionMode.ANCHORED_TARGET_RESIDUAL,
            alpha_enabled=True,
            risk_tilt_enabled=False,
            n_factors=0,
            target_weight_count=1,
            residual_scale=0.1,
        ),
        composer=BaselineResidualComposer(),
        max_gross=1.0,
        alpha_enabled=True,
        accept_legacy_actions=False,
        signal_delay_decisions=1,
        decision_every=None,
        decision_hours=1.0,
    )


def _request(*, pending: np.ndarray | None) -> EnvironmentDecisionRequest:
    return EnvironmentDecisionRequest(
        action=np.asarray([0.0], dtype=np.float32),
        trends=TrendTargets(
            fast=np.asarray([0.2]),
            base=np.asarray([0.1]),
            slow=np.asarray([0.05]),
        ),
        alpha=np.asarray([0.4], dtype=np.float64),
        factor_basis=np.empty((0, 1), dtype=np.float64),
        hybrid_weights=np.asarray([0.0], dtype=np.float64),
        shadow_weights=np.asarray([0.0], dtype=np.float64),
        pending_hybrid_target=None if pending is None else pending.copy(),
        pending_shadow_target=None if pending is None else pending.copy(),
        current_index=10,
        end_index=20,
    )


def test_anchored_zero_action_submits_teacher_anchor_to_hybrid_and_shadow() -> None:
    plan = _planner().plan(_request(pending=None))

    assert plan.submitted_hybrid_target.tolist() == pytest.approx([0.4])
    assert plan.submitted_shadow_target.tolist() == pytest.approx([0.4])
    assert plan.executed_hybrid_target.tolist() == pytest.approx([0.0])
    assert plan.executed_shadow_target.tolist() == pytest.approx([0.0])
    assert plan.next_pending_hybrid_target.tolist() == pytest.approx([0.4])
    assert plan.next_pending_shadow_target.tolist() == pytest.approx([0.4])
    assert plan.execution_delay_warmup is True


def test_anchored_mode_keeps_existing_one_decision_execution_delay() -> None:
    pending = np.asarray([0.4], dtype=np.float64)
    plan = _planner().plan(_request(pending=pending))

    assert plan.executed_hybrid_target.tolist() == pytest.approx([0.4])
    assert plan.executed_shadow_target.tolist() == pytest.approx([0.4])
    assert plan.next_pending_hybrid_target.tolist() == pytest.approx([0.4])
    assert plan.next_pending_shadow_target.tolist() == pytest.approx([0.4])
    assert plan.execution_delay_warmup is False
