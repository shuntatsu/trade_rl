from __future__ import annotations

from dataclasses import replace

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


def _planner(delay: int = 1) -> EnvironmentDecisionPlanner:
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
        signal_delay_decisions=delay,
        decision_every=None,
        decision_hours=1.0,
    )


def _request(
    *,
    submitted_reduce_only: bool,
    pending_target: np.ndarray | None = None,
    pending_reduce_only: np.ndarray | None = None,
) -> EnvironmentDecisionRequest:
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
        pending_hybrid_target=None if pending_target is None else pending_target.copy(),
        pending_shadow_target=None if pending_target is None else pending_target.copy(),
        current_index=10,
        end_index=20,
        submitted_hybrid_reduce_only_mask=np.asarray(
            [submitted_reduce_only], dtype=np.bool_
        ),
        pending_hybrid_reduce_only_mask=(
            None if pending_reduce_only is None else pending_reduce_only.copy()
        ),
    )


def test_reduce_only_mask_is_delayed_with_the_same_hybrid_target() -> None:
    warmup = _planner().plan(
        _request(submitted_reduce_only=True, pending_target=None, pending_reduce_only=None)
    )

    assert warmup.executed_hybrid_reduce_only_mask.tolist() == [False]
    assert warmup.next_pending_hybrid_reduce_only_mask.tolist() == [True]

    delayed = _planner().plan(
        _request(
            submitted_reduce_only=False,
            pending_target=np.asarray([0.4], dtype=np.float64),
            pending_reduce_only=np.asarray([True], dtype=np.bool_),
        )
    )

    assert delayed.executed_hybrid_target.tolist() == pytest.approx([0.4])
    assert delayed.executed_hybrid_reduce_only_mask.tolist() == [True]
    assert delayed.next_pending_hybrid_reduce_only_mask.tolist() == [False]


def test_zero_delay_uses_current_submission_reduce_only_mask() -> None:
    plan = _planner(delay=0).plan(
        _request(submitted_reduce_only=True, pending_target=None, pending_reduce_only=None)
    )

    assert plan.executed_hybrid_reduce_only_mask.tolist() == [True]
    assert plan.next_pending_hybrid_reduce_only_mask is None


def test_reduce_only_mask_must_be_boolean_and_symbol_aligned() -> None:
    request = replace(
        _request(submitted_reduce_only=True),
        submitted_hybrid_reduce_only_mask=np.asarray([1], dtype=np.int64),
    )

    with pytest.raises((TypeError, ValueError), match="reduce.only"):
        _planner().plan(request)
