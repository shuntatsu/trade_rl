from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.lagrangian import (
    DualUpdateReport,
    canonical_lagrangian_schema,
)
from trade_rl.rl.lagrangian_diagnostics import (
    build_constraint_correlation_diagnostics,
    build_dual_stability_diagnostics,
)
from trade_rl.rl.lagrangian_evidence import build_lagrangian_rollout_evidence
from trade_rl.rl.lagrangian_probe import (
    CanonicalActionProbeEvidence,
    CanonicalActionSemantic,
)


_NAMES = ("drawdown_excess", "drawdown_stop_event")


def _schema():
    return canonical_lagrangian_schema(
        names=_NAMES,
        budgets=(0.2, 0.1),
        dual_learning_rates=(0.25, 0.5),
        ema_betas=(0.5, 0.9),
        initial_multipliers=(0.0, 0.0),
        max_multipliers=(5.0, 10.0),
        warmup_rollouts=(0, 0),
        update_interval_rollouts=(1, 1),
        minimum_completed_episodes=(1, 20),
    )


def _report(
    name: str,
    *,
    residual: float,
    multiplier_after: float,
    at_lower_bound: bool,
    at_upper_cap: bool,
    consumed_denominator: int,
    pending_denominator_before: int,
    censored_episode_count: int = 2,
) -> DualUpdateReport:
    return DualUpdateReport(
        name=name,
        raw_estimate=0.3,
        ema_estimate=0.25,
        budget=0.2 if name == "drawdown_excess" else 0.1,
        multiplier_before=0.0,
        multiplier_after=multiplier_after,
        updated=True,
        skip_reason=None,
        denominator=consumed_denominator,
        pending_numerator_before=0.4,
        pending_denominator_before=pending_denominator_before,
        consumed_denominator=consumed_denominator,
        censored_episode_count=censored_episode_count,
        constraint_residual=residual,
        at_lower_bound=at_lower_bound,
        at_upper_cap=at_upper_cap,
        rollout_count=3,
        update_count=2,
    )


def _reports(*, residual_shift: float = 0.0):
    return {
        "drawdown_excess": _report(
            "drawdown_excess",
            residual=0.05 + residual_shift,
            multiplier_after=0.25,
            at_lower_bound=False,
            at_upper_cap=False,
            consumed_denominator=2,
            pending_denominator_before=1,
        ),
        "drawdown_stop_event": _report(
            "drawdown_stop_event",
            residual=-0.02,
            multiplier_after=0.0,
            at_lower_bound=True,
            at_upper_cap=False,
            consumed_denominator=20,
            pending_denominator_before=19,
        ),
    }


def _probe(*, warning: bool = True) -> CanonicalActionProbeEvidence:
    return CanonicalActionProbeEvidence(
        action_semantic=CanonicalActionSemantic.TARGET_WEIGHT_CASH,
        action=np.zeros(2, dtype=np.float32),
        estimates={"drawdown_excess": 0.3, "drawdown_stop_event": 0.0},
        denominators={"drawdown_excess": 2, "drawdown_stop_event": 2},
        budgets={"drawdown_excess": 0.2, "drawdown_stop_event": 0.1},
        violated_costs=("drawdown_excess",) if warning else (),
        completed_episode_count=2,
        censored_episode_count=1,
        episode_count=2,
        max_steps_per_episode=16,
        warning=warning,
    )


def _correlation(*, multiplier_scale: float = 1.0):
    return build_constraint_correlation_diagnostics(
        cost_names=_NAMES,
        raw_costs=np.asarray([[0.1, 0.0], [0.3, 1.0], [0.2, 0.0]]),
        raw_cost_advantages=np.asarray([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]]),
        normalized_cost_advantages=np.asarray(
            [[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]],
        ),
        multipliers=np.asarray([0.25, 0.5]) * multiplier_scale,
        reward_advantages=np.asarray([4.0, -2.0, 1.0]),
    )


def _stability(reports: dict[str, DualUpdateReport]):
    return build_dual_stability_diagnostics(
        cost_names=_NAMES,
        report_history=(reports,),
    )


def _evidence(
    *,
    residual_shift: float = 0.0,
    multiplier_scale: float = 1.0,
    warning: bool = True,
):
    reports = _reports(residual_shift=residual_shift)
    return build_lagrangian_rollout_evidence(
        actor_composition_mode="raw_lagrangian_then_sb3_normalize_v1",
        schema=_schema(),
        correlation_diagnostics=_correlation(multiplier_scale=multiplier_scale),
        stability_diagnostics=_stability(reports),
        dual_reports=reports,
        probe_evidence=_probe(warning=warning),
        completed_episode_count=2,
        censored_episode_count=2,
    )


def test_lagrangian_evidence_records_raw_penalty_and_boundary_semantics() -> None:
    evidence = _evidence()
    payload = evidence.payload()

    assert payload["schema_version"] == "lagrangian_rollout_evidence_v1"
    assert payload["actor_composition_mode"] == (
        "raw_lagrangian_then_sb3_normalize_v1"
    )
    assert payload["cost_names"] == list(_NAMES)
    assert payload["raw_reward_advantage_statistics"]["l2_norm"] == pytest.approx(
        np.linalg.norm([4.0, -2.0, 1.0])
    )
    assert payload["penalty_to_reward_l2_ratio"] == pytest.approx(
        evidence.correlation_diagnostics.penalty_to_reward_l2_ratio
    )
    np.testing.assert_array_equal(
        payload["raw_cost_covariance"],
        evidence.correlation_diagnostics.raw_cost_covariance.tolist(),
    )
    np.testing.assert_array_equal(
        payload["raw_cost_correlation"],
        evidence.correlation_diagnostics.raw_cost_correlation.tolist(),
    )
    assert payload["normalized_cost_advantage_correlation"] is not None

    drawdown = payload["constraints"]["drawdown_excess"]
    assert drawdown["aggregation"] == "episode_time_area"
    assert drawdown["unit"] == "drawdown_excess_area_days"
    assert drawdown["pending_denominator_before"] == 1
    assert drawdown["consumed_denominator"] == 2
    assert drawdown["beta_effective"] == pytest.approx(0.5**2)
    assert drawdown["constraint_residual"] == pytest.approx(0.05)
    assert drawdown["at_lower_bound"] is False
    assert drawdown["at_upper_cap"] is False

    event = payload["constraints"]["drawdown_stop_event"]
    assert event["aggregation"] == "episode_event_rate"
    assert event["unit"] == "event_per_episode"
    assert event["minimum_completed_episodes"] == 20
    assert event["beta_effective"] == pytest.approx(0.9**20)
    assert event["at_lower_bound"] is True
    assert event["at_upper_cap"] is False

    assert payload["probe"]["semantic"] == "target_weight_cash"
    assert payload["probe"]["warning"] is True
    assert payload["probe"]["digest"] == evidence.probe_evidence.digest
    assert payload["probe"]["payload"] == evidence.probe_evidence.digest_payload()
    assert payload["completed_episode_count"] == 2
    assert payload["censored_episode_count"] == 2
    assert payload["digest"] == evidence.digest


def test_lagrangian_evidence_digest_changes_with_semantic_inputs() -> None:
    baseline = _evidence()
    variants = (
        _evidence(residual_shift=0.01),
        _evidence(multiplier_scale=2.0),
        _evidence(warning=False),
    )

    assert len({baseline.digest, *(variant.digest for variant in variants)}) == 4


def test_lagrangian_evidence_rejects_reordered_dual_reports() -> None:
    reports = _reports()
    reordered = {
        "drawdown_stop_event": reports["drawdown_stop_event"],
        "drawdown_excess": reports["drawdown_excess"],
    }

    with pytest.raises(ValueError, match="constraint order"):
        build_lagrangian_rollout_evidence(
            actor_composition_mode="raw_lagrangian_then_sb3_normalize_v1",
            schema=_schema(),
            correlation_diagnostics=_correlation(),
            stability_diagnostics=_stability(reports),
            dual_reports=reordered,
            probe_evidence=_probe(),
            completed_episode_count=2,
            censored_episode_count=2,
        )
