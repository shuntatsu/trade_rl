from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
    AllocationComparisonScenario,
    AllocationValidity,
)
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def capability():
    from trade_rl.evaluation.allocation_comparison_evidence import (
        allocation_business_objective_digest,
        allocation_comparison_scenario_digest,
        allocation_economic_clock_digest,
        allocation_fold_plan_digest,
        allocation_policy_schedule_digest,
        build_continuous_allocation_comparison_evidence,
        continuous_account_profit_and_drawdown,
    )

    return (
        allocation_business_objective_digest,
        allocation_comparison_scenario_digest,
        allocation_economic_clock_digest,
        allocation_fold_plan_digest,
        allocation_policy_schedule_digest,
        build_continuous_allocation_comparison_evidence,
        continuous_account_profit_and_drawdown,
    )


def policy(digest, actions):
    values = iter(actions)
    return AllocationFoldPolicy(
        policy_digest=digest,
        recipe_digest="f" * 64,
        action=lambda _observation, _recipe: next(values),
    )


def run_result():
    first, second = env_for(6, 8), env_for(8, 10)
    folds = folds_for(first, second)
    policies = (
        AllocationFoldPolicy("1" * 64, first.recipe_digest, lambda _o, _r: 3),
        AllocationFoldPolicy("2" * 64, second.recipe_digest, lambda _o, _r: 0),
    )
    result = run_continuous_allocation_walk_forward(
        folds,
        (first, second),
        policies,
        reset_seed=7,
    )
    return first, second, folds, result


def contract_for(first, folds, *, scenario_name="base", economics=None, risk=None):
    (
        objective_digest,
        scenario_digest,
        clock_digest,
        fold_digest,
        _policy_digest,
        _build,
        _profit,
    ) = capability()
    actual_economics = (
        first.executor.execution_policy_digest if economics is None else economics
    )
    actual_risk = content_digest(first.risk_config) if risk is None else risk
    scenario = AllocationComparisonScenario(
        scenario_name,
        scenario_digest(
            name=scenario_name,
            dataset_id=first.dataset.dataset_id,
            forecast_context_digest=first.stream.digest,
            economics_digest=actual_economics,
            risk_digest=actual_risk,
        ),
        required=True,
    )
    return AllocationComparisonContract(
        dataset_id=first.dataset.dataset_id,
        objective_digest=objective_digest(first.bound.objective),
        clock_digest=clock_digest(first.bound.clock),
        forecast_context_digest=first.stream.digest,
        economics_digest=first.executor.execution_policy_digest,
        risk_digest=content_digest(first.risk_config),
        fold_plan_digest=fold_digest(folds),
        nonrl_recipe_digest=first.recipe_digest,
        residual_recipe_digest=first.recipe_digest,
        direct_recipe_digest=first.recipe_digest,
        account_mode=first.bound.objective.capital.account_mode,
        initial_capital=first.initial_capital,
        scenarios=(scenario,),
        rl_seeds=(0,),
        maximum_drawdown=first.bound.objective.maximum_drawdown,
    )


def test_profit_and_drawdown_compound_geometric_account_returns():
    *_, profit = capability()
    terminal, drawdown = profit(
        ReturnSeries((0.10, -0.10, 0.20), ReturnKind.DECISION_STEP, 8760)
    )
    assert terminal == pytest.approx((1.10 * 0.90 * 1.20) - 1.0)
    # wealth path: 1.0 -> 1.1 -> 0.99 -> 1.188
    assert drawdown == pytest.approx(0.10)
    assert terminal != pytest.approx(sum((0.10, -0.10, 0.20)))


def test_business_objective_profile_excludes_fold_window_recipe_and_scenario_economics():
    objective_digest, *_ = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    changed = replace(
        second.bound.objective,
        economics_digest="a" * 64,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
    )
    assert objective_digest(first.bound.objective) == objective_digest(changed)


def test_economic_clock_profile_excludes_training_only_rollout_and_gae():
    _, _, clock_digest, *_ = capability()
    first = env_for(6, 8)
    base = first.bound.clock
    changed = replace(base, rollout_steps=base.rollout_steps + 8, gae_lambda=1.0)
    assert clock_digest(base) == clock_digest(changed)
    assert clock_digest(base) != clock_digest(
        replace(
            base,
            decision_interval_seconds=base.decision_interval_seconds * 2,
            reward_interval_seconds=base.reward_interval_seconds * 2,
        )
    )


def test_fold_plan_digest_changes_on_any_declared_range_change():
    *_, fold_digest, _policy, _build, _profit = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    folds = folds_for(first, second)
    changed = replace(
        folds[0],
        train=replace(folds[0].train, start=1),
    )
    assert fold_digest(folds) != fold_digest((changed, folds[1]))


def test_actual_continuous_result_builds_canonical_direct_ppo_evidence():
    (
        _objective,
        _scenario,
        _clock,
        _fold,
        policy_schedule,
        build,
        _profit,
    ) = capability()
    first, second, folds, result = run_result()
    declaration = contract_for(first, folds)
    evidence = build(
        declaration,
        candidate=AllocationCandidateKind.DIRECT_PPO,
        scenario="base",
        seed=0,
        folds=folds,
        environments=(first, second),
        result=result,
        validity_evidence_digest=content_digest({"validity": "independent-oracle"}),
        validity=AllocationValidity.VALID,
    )

    assert evidence.contract_digest == declaration.digest
    assert evidence.recipe_digest == declaration.direct_recipe_digest
    assert evidence.policy_digest == policy_schedule(
        AllocationCandidateKind.DIRECT_PPO, 0, result.policy_digests
    )
    assert evidence.terminal_profit_rate == pytest.approx(
        second.book.portfolio_value / first.initial_capital - 1.0
    )
    assert evidence.max_drawdown == pytest.approx(second.book.max_drawdown)
    assert evidence.opening_state_digest == result.folds[0].opening_state_digest
    assert evidence.closing_state_digest == result.folds[-1].closing_state_digest
    assert evidence.coverage_complete is True
    assert evidence.termination_reason is None
    assert evidence.ledger_digest != evidence.execution_digest


def test_nonrl_evidence_has_no_policy_digest_but_keeps_candidate_recipe():
    *_, build, _profit = capability()
    first, second, folds, result = run_result()
    declaration = contract_for(first, folds)
    evidence = build(
        declaration,
        candidate=AllocationCandidateKind.NONRL,
        scenario="base",
        seed=None,
        folds=folds,
        environments=(first, second),
        result=result,
        validity_evidence_digest="a" * 64,
    )
    assert evidence.policy_digest is None
    assert evidence.seed is None
    assert evidence.recipe_digest == declaration.nonrl_recipe_digest


@pytest.mark.parametrize(
    "change,match",
    [
        ("dataset", "Dataset"),
        ("forecast", "forecast"),
        ("recipe", "recipe"),
        ("fold", "fold"),
        ("clock", "clock"),
        ("objective", "objective"),
        ("scenario", "scenario"),
    ],
)
def test_runtime_context_mismatch_fails_before_evidence_is_built(change, match):
    (
        _objective,
        scenario_digest,
        _clock,
        _fold,
        _policy,
        build,
        _profit,
    ) = capability()
    first, second, folds, result = run_result()
    declaration = contract_for(first, folds)
    environments = (first, second)
    if change == "dataset":
        declaration = replace(declaration, dataset_id="a" * 64)
    elif change == "forecast":
        declaration = replace(declaration, forecast_context_digest="a" * 64)
    elif change == "recipe":
        declaration = replace(declaration, direct_recipe_digest="a" * 64)
    elif change == "fold":
        declaration = replace(declaration, fold_plan_digest="a" * 64)
    elif change == "clock":
        declaration = replace(declaration, clock_digest="a" * 64)
    elif change == "objective":
        declaration = replace(declaration, objective_digest="a" * 64)
    else:
        bad = AllocationComparisonScenario(
            "base",
            scenario_digest(
                name="base",
                dataset_id=first.dataset.dataset_id,
                forecast_context_digest=first.stream.digest,
                economics_digest="a" * 64,
                risk_digest=content_digest(first.risk_config),
            ),
            required=True,
        )
        declaration = replace(declaration, scenarios=(bad,))

    with pytest.raises(ValueError, match=match):
        build(
            declaration,
            candidate=AllocationCandidateKind.DIRECT_PPO,
            scenario="base",
            seed=0,
            folds=folds,
            environments=environments,
            result=result,
            validity_evidence_digest="b" * 64,
        )


def test_rl_policy_schedule_digest_changes_when_one_fold_policy_changes():
    *_, schedule_digest, _build, _profit = capability()
    first = schedule_digest(
        AllocationCandidateKind.RESIDUAL_PPO, 7, ("1" * 64, "2" * 64)
    )
    second = schedule_digest(
        AllocationCandidateKind.RESIDUAL_PPO, 7, ("1" * 64, "3" * 64)
    )
    assert first != second


def test_builder_rejects_tampered_continuous_result_state_or_stitching():
    *_, build, _profit = capability()
    first, second, folds, result = run_result()
    declaration = contract_for(first, folds)

    changed_fold = replace(result.folds[-1], closing_state_digest="a" * 64)
    tampered_state = replace(result, folds=(result.folds[0], changed_fold))
    with pytest.raises(ValueError, match="stitched|state|canonical"):
        build(
            declaration,
            candidate=AllocationCandidateKind.DIRECT_PPO,
            scenario="base",
            seed=0,
            folds=folds,
            environments=(first, second),
            result=tampered_state,
            validity_evidence_digest="b" * 64,
        )

    changed_returns = ReturnSeries(
        tuple(
            value + (0.0001 if index == 0 else 0.0)
            for index, value in enumerate(result.stitched.returns.values)
        ),
        result.stitched.returns.kind,
        result.stitched.returns.periods_per_year,
    )
    tampered_stitched = replace(
        result,
        stitched=replace(result.stitched, returns=changed_returns),
    )
    with pytest.raises(ValueError, match="stitched|fold"):
        build(
            declaration,
            candidate=AllocationCandidateKind.DIRECT_PPO,
            scenario="base",
            seed=0,
            folds=folds,
            environments=(first, second),
            result=tampered_stitched,
            validity_evidence_digest="c" * 64,
        )
