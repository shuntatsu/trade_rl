from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.lagrangian import DualUpdateReport
from trade_rl.rl.lagrangian_diagnostics import (
    build_constraint_correlation_diagnostics,
    build_dual_stability_diagnostics,
)


def _report(
    *,
    multiplier_before: float,
    multiplier_after: float,
    at_lower_bound: bool,
    at_upper_cap: bool,
    residual: float,
    rollout_count: int,
) -> DualUpdateReport:
    return DualUpdateReport(
        name="drawdown_excess",
        raw_estimate=0.1,
        ema_estimate=0.1,
        budget=0.1,
        multiplier_before=multiplier_before,
        multiplier_after=multiplier_after,
        updated=True,
        skip_reason=None,
        denominator=1,
        pending_numerator_before=0.0,
        pending_denominator_before=0,
        consumed_denominator=1,
        censored_episode_count=0,
        constraint_residual=residual,
        at_lower_bound=at_lower_bound,
        at_upper_cap=at_upper_cap,
        rollout_count=rollout_count,
        update_count=rollout_count,
    )


def test_effective_penalty_diagnostics_use_raw_cost_advantages() -> None:
    raw_cost_advantages = np.asarray(
        [[1.0, 10.0], [3.0, 30.0]],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [[-1.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    multipliers = np.asarray([2.0, 0.5], dtype=np.float64)
    reward_advantages = np.asarray([4.0, -2.0], dtype=np.float64)

    diagnostics = build_constraint_correlation_diagnostics(
        cost_names=("drawdown_excess", "execution_cost_fraction"),
        raw_costs=np.asarray([[0.1, 0.01], [0.2, 0.02]], dtype=np.float64),
        raw_cost_advantages=raw_cost_advantages,
        normalized_cost_advantages=normalized,
        multipliers=multipliers,
        reward_advantages=reward_advantages,
    )

    expected_contributions = raw_cost_advantages * multipliers[None, :]
    expected_penalty = expected_contributions.sum(axis=1)
    expected_ratio = np.linalg.norm(expected_penalty) / max(
        np.linalg.norm(reward_advantages),
        1e-12,
    )
    np.testing.assert_array_equal(
        diagnostics.penalty_contributions,
        expected_contributions,
    )
    np.testing.assert_array_equal(diagnostics.aggregate_penalty, expected_penalty)
    assert diagnostics.penalty_to_reward_l2_ratio == pytest.approx(expected_ratio)
    assert diagnostics.penalty_contributions.flags.writeable is False
    assert diagnostics.aggregate_penalty.flags.writeable is False


def test_normalized_cost_advantages_are_correlation_only() -> None:
    raw_cost_advantages = np.asarray(
        [[1.0, 100.0], [2.0, 20.0], [4.0, 5.0]],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    multipliers = np.asarray([3.0, 0.25], dtype=np.float64)

    diagnostics = build_constraint_correlation_diagnostics(
        cost_names=("drawdown_excess", "execution_cost_fraction"),
        raw_costs=np.asarray([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        raw_cost_advantages=raw_cost_advantages,
        normalized_cost_advantages=normalized,
        multipliers=multipliers,
        reward_advantages=np.asarray([1.0, 2.0, 3.0]),
    )

    np.testing.assert_array_equal(
        diagnostics.penalty_contributions,
        raw_cost_advantages * multipliers[None, :],
    )
    assert diagnostics.normalized_cost_advantage_correlation is not None
    np.testing.assert_allclose(
        diagnostics.normalized_cost_advantage_correlation,
        np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        rtol=0.0,
        atol=1e-12,
    )


def test_constant_inputs_produce_deterministic_zero_correlation_rows() -> None:
    diagnostics = build_constraint_correlation_diagnostics(
        cost_names=("drawdown_excess", "execution_cost_fraction"),
        raw_costs=np.asarray([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        raw_cost_advantages=np.asarray(
            [[1.0, 2.0], [1.0, 4.0], [1.0, 8.0]],
        ),
        normalized_cost_advantages=None,
        multipliers=np.asarray([0.0, 1.0]),
        reward_advantages=np.asarray([1.0, 1.0, 1.0]),
    )

    np.testing.assert_array_equal(
        diagnostics.raw_cost_correlation[0],
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        diagnostics.raw_cost_correlation[:, 0],
        np.zeros(2),
    )
    assert diagnostics.raw_cost_correlation[1, 1] == pytest.approx(1.0)
    assert diagnostics.normalized_cost_advantage_correlation is None
    assert diagnostics.raw_cost_correlation.flags.writeable is False


def test_stability_saturation_counts_upper_cap_only() -> None:
    multipliers = (0.0, 0.0, 10.0, 10.0, 5.0)
    residuals = (-1.0, -0.5, 1.0, 2.0, -0.1)
    history = tuple(
        {
            "drawdown_excess": _report(
                multiplier_before=(multipliers[index - 1] if index else 0.0),
                multiplier_after=value,
                at_lower_bound=value == 0.0,
                at_upper_cap=value == 10.0,
                residual=residuals[index],
                rollout_count=index + 1,
            )
        }
        for index, value in enumerate(multipliers)
    )

    diagnostics = build_dual_stability_diagnostics(
        cost_names=("drawdown_excess",),
        report_history=history,
    )
    constraint = diagnostics.constraints[0]

    assert constraint.saturation_fraction == pytest.approx(2.0 / 5.0)
    assert constraint.lower_bound_fraction == pytest.approx(2.0 / 5.0)
    assert constraint.longest_saturation_run == 2
    assert constraint.violation_area == pytest.approx(3.0)
    assert constraint.longest_satisfaction_run == 2
    assert constraint.post_satisfaction_overconstraint_count == 2


def test_stability_rejects_reordered_or_missing_reports() -> None:
    report = _report(
        multiplier_before=0.0,
        multiplier_after=0.0,
        at_lower_bound=True,
        at_upper_cap=False,
        residual=-0.1,
        rollout_count=1,
    )

    with pytest.raises(ValueError, match="constraint names"):
        build_dual_stability_diagnostics(
            cost_names=("drawdown_excess",),
            report_history=({"execution_cost_fraction": report},),
        )
