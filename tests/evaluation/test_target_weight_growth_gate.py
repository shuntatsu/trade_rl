from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.target_weight_growth_gate import (
    GrowthEvaluationCell,
    GrowthProfileComparisonCell,
    SoftConstraintEstimate,
    evaluate_target_weight_growth_gate,
    select_target_weight_growth_profile,
)


def _constraints(
    *,
    turnover: float = 0.50,
    cost: float = 0.01,
) -> tuple[SoftConstraintEstimate, ...]:
    return (
        SoftConstraintEstimate(
            name="daily_turnover",
            observed_value=turnover,
            budget=1.0,
        ),
        SoftConstraintEstimate(
            name="execution_cost_fraction",
            observed_value=cost,
            budget=0.03,
        ),
    )


def _cells() -> tuple[GrowthEvaluationCell, ...]:
    cells: list[GrowthEvaluationCell] = []
    scenarios = {
        "nominal": (0.06, 0.02),
        "joint_2x": (0.04, 0.02),
        "joint_3x": (0.01, 0.005),
    }
    for scenario, (selected, baseline) in scenarios.items():
        for fold_index in range(6):
            for seed in (0, 1, 2):
                cells.append(
                    GrowthEvaluationCell(
                        fold_index=fold_index,
                        seed=seed,
                        scenario=scenario,
                        selected_net_log_growth=(
                            selected + 0.001 * fold_index + 0.0001 * seed
                        ),
                        baseline_net_log_growth=baseline,
                        soft_constraints=_constraints(),
                    )
                )
    return tuple(cells)


def test_growth_production_gate_accepts_complete_stable_evidence() -> None:
    decision = evaluate_target_weight_growth_gate(
        _cells(),
        identity_verified=True,
    )

    assert decision.passed is True
    assert decision.reasons == ()
    assert decision.nominal_cell_count == 18
    assert decision.nominal_growth_median > 0.0
    assert decision.nominal_paired_median > 0.0
    assert decision.nominal_paired_lower_bound > 0.0
    assert decision.positive_fold_count == 6
    assert decision.nonnegative_seed_count == 3
    assert decision.positive_seed_count == 3
    assert decision.cost_2x_paired_median > 0.0
    assert decision.cost_3x_growth_median >= 0.0
    assert decision.catastrophic_event_count == 0
    assert all(summary.passed for summary in decision.soft_constraints)
    assert len(decision.digest) == 64


def test_growth_production_gate_fails_closed_on_missing_support() -> None:
    decision = evaluate_target_weight_growth_gate(
        _cells()[:-1],
        identity_verified=True,
    )

    assert decision.passed is False
    assert any(reason.startswith("support_mismatch:") for reason in decision.reasons)


def test_growth_production_gate_rejects_any_catastrophic_event() -> None:
    cells = list(_cells())
    cells[0] = replace(cells[0], forced_liquidation_count=1)

    decision = evaluate_target_weight_growth_gate(
        tuple(cells),
        identity_verified=True,
    )

    assert decision.passed is False
    assert decision.catastrophic_event_count == 1
    assert "catastrophic_event_detected" in decision.reasons


def test_growth_production_gate_rejects_soft_constraint_uncertainty() -> None:
    cells = tuple(
        replace(
            cell,
            soft_constraints=_constraints(turnover=1.01),
        )
        if cell.scenario == "nominal" and cell.fold_index == 5
        else cell
        for cell in _cells()
    )

    decision = evaluate_target_weight_growth_gate(
        cells,
        identity_verified=True,
    )

    assert decision.passed is False
    turnover = next(
        summary
        for summary in decision.soft_constraints
        if summary.name == "daily_turnover"
    )
    assert turnover.maximum_fold_estimate > turnover.budget
    assert turnover.passed is False
    assert "soft_constraint_budget_failed:daily_turnover" in decision.reasons


def test_growth_production_gate_requires_verified_identity() -> None:
    decision = evaluate_target_weight_growth_gate(
        _cells(),
        identity_verified=False,
    )

    assert decision.passed is False
    assert "identity_not_verified" in decision.reasons


def _comparison_cells(delta: float) -> tuple[GrowthProfileComparisonCell, ...]:
    return tuple(
        GrowthProfileComparisonCell(
            fold_index=fold_index,
            seed=seed,
            lagrangian_minus_ppo_net_log_growth=delta,
        )
        for fold_index in range(6)
        for seed in (0, 1, 2)
    )


def test_profile_selection_chooses_lagrangian_only_with_positive_ci() -> None:
    ppo = evaluate_target_weight_growth_gate(_cells(), identity_verified=True)
    lagrangian = evaluate_target_weight_growth_gate(_cells(), identity_verified=True)

    decision = select_target_weight_growth_profile(
        ppo=ppo,
        lagrangian=lagrangian,
        comparisons=_comparison_cells(0.01),
    )

    assert decision.selected_profile == "g1_lagrangian"
    assert decision.lagrangian_minus_ppo_lower_bound > 0.0


def test_profile_selection_prefers_ppo_when_growth_is_indistinguishable() -> None:
    ppo = evaluate_target_weight_growth_gate(_cells(), identity_verified=True)
    lagrangian = evaluate_target_weight_growth_gate(_cells(), identity_verified=True)

    decision = select_target_weight_growth_profile(
        ppo=ppo,
        lagrangian=lagrangian,
        comparisons=_comparison_cells(0.0),
    )

    assert decision.selected_profile == "g1_ppo"
    assert decision.reason == "growth_difference_not_significantly_positive"


def test_growth_cell_rejects_duplicate_constraint_names() -> None:
    duplicate = _constraints()[0]

    with pytest.raises(ValueError, match="soft constraint names must be unique"):
        GrowthEvaluationCell(
            fold_index=0,
            seed=0,
            scenario="nominal",
            selected_net_log_growth=0.1,
            baseline_net_log_growth=0.0,
            soft_constraints=(duplicate, duplicate),
        )
