from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.constrained_policy_report import (
    ConstraintCostObservation,
    ConstraintFoldEvidence,
    ConstraintPolicyObservation,
    build_constrained_policy_report,
)
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES

_BUDGETS = {
    "drawdown_excess": 0.5,
    "drawdown_stop_event": 0.0,
    "margin_deficit_fraction": 0.5,
    "forced_liquidation_event": 0.0,
    "gross_exposure_request_excess": 0.5,
    "daily_turnover": 1.0,
    "execution_cost_fraction": 0.03,
}


def _digest(character: str) -> str:
    return character * 64


def _costs(
    *,
    drawdown_excess: float,
    daily_turnover: float = 0.4,
    completed_episode_denominator: int = 30,
    upper_cap_fraction: float = 0.0,
    lower_bound_fraction: float = 0.5,
    missing_ema: bool = False,
) -> tuple[ConstraintCostObservation, ...]:
    values = {
        "drawdown_excess": drawdown_excess,
        "drawdown_stop_event": 0.0,
        "margin_deficit_fraction": 0.05,
        "forced_liquidation_event": 0.0,
        "gross_exposure_request_excess": 0.1,
        "daily_turnover": daily_turnover,
        "execution_cost_fraction": 0.01,
    }
    return tuple(
        ConstraintCostObservation(
            name=name,
            observed_value=values[name],
            budget=_BUDGETS[name],
            completed_episode_denominator=completed_episode_denominator,
            censored_episode_count=2,
            minimum_completed_episodes=(
                20
                if name in {"drawdown_stop_event", "forced_liquidation_event"}
                else 1
            ),
            raw_estimate=values[name] * 0.9,
            ema_estimate=(
                None if missing_ema and name == "drawdown_excess" else values[name]
            ),
            multiplier_mean=0.25,
            multiplier_max=0.4,
            upper_cap_fraction=upper_cap_fraction,
            lower_bound_fraction=lower_bound_fraction,
            cost_critic_explained_variance=0.3,
            cost_critic_loss=0.05,
        )
        for name in CONSTRAINT_COST_NAMES
    )


def _seed_observation(
    *,
    scenario: str,
    seed: int,
    fold_index: int,
    constrained: bool,
    drawdown_excess: float,
    completed_episode_denominator: int = 30,
    upper_cap_fraction: float = 0.0,
    lower_bound_fraction: float = 0.5,
    missing_ema: bool = False,
) -> ConstraintPolicyObservation:
    policy_digest = _digest(chr(ord("a") + seed))
    return ConstraintPolicyObservation(
        scenario=scenario,
        seed=seed,
        policy_digest=policy_digest,
        expected_member_policy_digests=(policy_digest,),
        evaluated_member_policy_digests=(policy_digest,),
        costs=(
            _costs(
                drawdown_excess=drawdown_excess,
                completed_episode_denominator=completed_episode_denominator,
                upper_cap_fraction=upper_cap_fraction,
                lower_bound_fraction=lower_bound_fraction,
                missing_ema=missing_ema,
            )
            if constrained
            else None
        ),
        penalty_to_reward_l2_ratio=0.2 if constrained else None,
        raw_to_filled_distortion=0.02,
        total_return=0.01 + seed * 0.005 + fold_index * 0.001,
        maximum_drawdown=0.08 + seed * 0.01,
        turnover_per_day=0.4 + seed * 0.05,
        economic_cost_fraction=0.01,
    )


def _ensemble_observation(
    *,
    scenario: str,
    fold_index: int,
    constrained: bool,
    drawdown_excess: float,
    completed_episode_denominator: int = 30,
    upper_cap_fraction: float = 0.0,
    lower_bound_fraction: float = 0.5,
    identity_mismatch: bool = False,
    daily_turnover: float = 0.4,
    missing_ema: bool = False,
) -> ConstraintPolicyObservation:
    expected = (_digest("a"), _digest("b"))
    evaluated = (
        (_digest("a"), _digest("c")) if identity_mismatch else expected
    )
    return ConstraintPolicyObservation(
        scenario=scenario,
        seed=None,
        policy_digest=_digest("e"),
        expected_member_policy_digests=expected,
        evaluated_member_policy_digests=evaluated,
        costs=(
            _costs(
                drawdown_excess=drawdown_excess,
                daily_turnover=daily_turnover,
                completed_episode_denominator=completed_episode_denominator,
                upper_cap_fraction=upper_cap_fraction,
                lower_bound_fraction=lower_bound_fraction,
                missing_ema=missing_ema,
            )
            if constrained
            else None
        ),
        penalty_to_reward_l2_ratio=0.25 if constrained else None,
        raw_to_filled_distortion=0.03,
        total_return=0.02 + fold_index * 0.002,
        maximum_drawdown=0.09 + fold_index * 0.01,
        turnover_per_day=0.45,
        economic_cost_fraction=0.012,
    )


def _fold(
    fold_index: int,
    *,
    constrained: bool = True,
    omit_scenario: str | None = None,
    identity_mismatch: bool = False,
    insufficient_support: bool = False,
    budget_violation: bool = False,
    upper_cap_fraction: float = 0.0,
    lower_bound_fraction: float = 0.5,
    missing_ema: bool = False,
) -> ConstraintFoldEvidence:
    seeds: list[ConstraintPolicyObservation] = []
    ensembles: list[ConstraintPolicyObservation] = []
    for scenario, scenario_offset in (("nominal", 0.0), ("joint_2x", 0.02)):
        if scenario == omit_scenario:
            continue
        denominator = 19 if insufficient_support and scenario == "joint_2x" else 30
        seeds.extend(
            _seed_observation(
                scenario=scenario,
                seed=seed,
                fold_index=fold_index,
                constrained=constrained,
                drawdown_excess=(
                    0.10 + seed * 0.10 + fold_index * 0.05 + scenario_offset
                ),
                completed_episode_denominator=denominator,
                upper_cap_fraction=upper_cap_fraction,
                lower_bound_fraction=lower_bound_fraction,
                missing_ema=missing_ema and scenario == "joint_2x",
            )
            for seed in (0, 1)
        )
        ensembles.append(
            _ensemble_observation(
                scenario=scenario,
                fold_index=fold_index,
                constrained=constrained,
                drawdown_excess=0.15 + fold_index * 0.05 + scenario_offset,
                completed_episode_denominator=denominator,
                upper_cap_fraction=upper_cap_fraction,
                lower_bound_fraction=lower_bound_fraction,
                identity_mismatch=(
                    identity_mismatch and scenario == "joint_2x"
                ),
                daily_turnover=(
                    1.2 if budget_violation and scenario == "joint_2x" else 0.4
                ),
                missing_ema=missing_ema and scenario == "joint_2x",
            )
        )
    return ConstraintFoldEvidence(
        fold_index=fold_index,
        configuration=("constrained-growth" if constrained else "ppo-control"),
        constrained=constrained,
        seed_observations=tuple(seeds),
        ensemble_observations=tuple(ensembles),
    )


def test_constrained_report_aggregates_folds_without_curve_concatenation() -> None:
    report = build_constrained_policy_report((_fold(1), _fold(0)))

    assert report.configuration == "constrained-growth"
    assert report.constrained is True
    assert report.eligibility.eligible is True
    assert report.eligibility.reasons == ()
    assert tuple(summary.fold_index for summary in report.fold_summaries) == (
        0,
        0,
        1,
        1,
    )
    nominal = next(
        summary
        for summary in report.aggregate_summaries
        if summary.scenario == "nominal"
    )
    assert nominal.fold_count == 2
    assert nominal.seed_count == 4
    assert nominal.mean_total_return == pytest.approx(0.021)
    assert nominal.worst_seed_total_return == pytest.approx(0.01)
    assert nominal.worst_fold_total_return == pytest.approx(0.02)
    assert nominal.constraints is not None
    drawdown = nominal.constraints[0]
    assert drawdown.name == "drawdown_excess"
    assert drawdown.mean == pytest.approx(0.175)
    assert drawdown.worst_seed == pytest.approx(0.25)
    assert drawdown.worst_fold == pytest.approx(0.20)
    assert drawdown.completed_episode_denominator == 30
    assert drawdown.censored_episode_count == 2
    assert drawdown.upper_cap_fraction == pytest.approx(0.0)
    assert drawdown.lower_bound_fraction == pytest.approx(0.5)


def test_required_scenario_budget_violation_is_ineligible() -> None:
    report = build_constrained_policy_report((_fold(0, budget_violation=True),))

    assert report.eligibility.eligible is False
    assert any(
        reason
        == (
            "constraint_budget_exceeded:fold=0:scenario=joint_2x:"
            "cost=daily_turnover:scope=ensemble"
        )
        for reason in report.eligibility.reasons
    )


def test_rare_event_support_below_minimum_is_ineligible() -> None:
    report = build_constrained_policy_report((_fold(0, insufficient_support=True),))

    assert report.eligibility.eligible is False
    assert any(
        reason.startswith(
            "constraint_support_below_minimum:fold=0:scenario=joint_2x:"
            "cost=drawdown_stop_event:"
        )
        for reason in report.eligibility.reasons
    )


def test_member_identity_mismatch_and_missing_required_scenario_fail_closed() -> None:
    mismatch = build_constrained_policy_report((_fold(0, identity_mismatch=True),))
    missing = build_constrained_policy_report(
        (_fold(0, omit_scenario="joint_2x"),)
    )

    assert mismatch.eligibility.eligible is False
    assert (
        "member_identity_mismatch:fold=0:scenario=joint_2x"
        in mismatch.eligibility.reasons
    )
    assert missing.eligibility.eligible is False
    assert "missing_required_scenario:fold=0:scenario=joint_2x" in (
        missing.eligibility.reasons
    )


def test_missing_constrained_model_diagnostics_do_not_create_partial_average() -> None:
    report = build_constrained_policy_report((_fold(0, missing_ema=True),))

    assert report.eligibility.eligible is False
    assert any(
        reason.endswith("cost=drawdown_excess:field=ema_estimate")
        for reason in report.eligibility.reasons
    )
    joint = next(
        summary
        for summary in report.aggregate_summaries
        if summary.scenario == "joint_2x"
    )
    assert joint.constraints is not None
    assert joint.constraints[0].ema_estimate is None


def test_ordinary_ppo_preserves_absent_constraint_evidence() -> None:
    report = build_constrained_policy_report((_fold(0, constrained=False),))

    assert report.configuration == "ppo-control"
    assert report.constrained is False
    assert report.eligibility.eligible is True
    assert all(summary.constraints is None for summary in report.fold_summaries)
    assert all(summary.constraints is None for summary in report.aggregate_summaries)
    assert all(
        summary.penalty_to_reward_l2_ratio is None
        for summary in report.fold_summaries
    )

    ordinary = _ensemble_observation(
        scenario="nominal",
        fold_index=0,
        constrained=False,
        drawdown_excess=0.1,
    )
    with pytest.raises(ValueError, match="ordinary PPO cannot contain constraint evidence"):
        replace(ordinary, costs=_costs(drawdown_excess=0.1))


def test_lower_bound_occupancy_is_not_upper_cap_saturation() -> None:
    report = build_constrained_policy_report(
        (
            _fold(
                0,
                upper_cap_fraction=0.0,
                lower_bound_fraction=1.0,
            ),
        )
    )

    assert report.eligibility.eligible is True
    nominal = next(
        summary
        for summary in report.aggregate_summaries
        if summary.scenario == "nominal"
    )
    assert nominal.constraints is not None
    assert all(cost.upper_cap_fraction == 0.0 for cost in nominal.constraints)
    assert all(cost.lower_bound_fraction == 1.0 for cost in nominal.constraints)


def test_non_finite_observation_fails_during_construction() -> None:
    with pytest.raises(ValueError, match="observed_value must be finite"):
        replace(_costs(drawdown_excess=0.1)[0], observed_value=float("inf"))


def test_report_digest_is_invariant_to_fold_input_order() -> None:
    left = build_constrained_policy_report((_fold(0), _fold(1)))
    right = build_constrained_policy_report((_fold(1), _fold(0)))

    assert left.digest == right.digest
    assert left.digest_payload() == right.digest_payload()
