from copy import deepcopy
from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_rl_env import parameters


def capability():
    from trade_rl.evaluation.allocation_scenario_identity import (
        allocation_candidate_recipe_digest,
        allocation_runtime_invariant_digest,
        validate_allocation_scenario_recipe,
    )

    return (
        allocation_candidate_recipe_digest,
        allocation_runtime_invariant_digest,
        validate_allocation_scenario_recipe,
    )


def test_candidate_identity_ignores_declared_execution_economics_change():
    candidate, invariant, validate = capability()
    base = env_for(6, 8)
    args = parameters(stop=10)
    stress = env_for(
        6,
        8,
        cost=replace(
            args["execution_cost"], fee_rate=args["execution_cost"].fee_rate * 2
        ),
    )
    assert base.recipe_digest != stress.recipe_digest
    assert candidate(base.recipe) == candidate(stress.recipe)
    assert invariant(base.recipe) == invariant(stress.recipe)
    validate(
        base.recipe,
        stress.recipe,
        base_economics_digest=base.executor.execution_policy_digest,
        base_risk_digest=base.bound.objective.risk_digest,
        scenario_economics_digest=stress.executor.execution_policy_digest,
        scenario_risk_digest=stress.bound.objective.risk_digest,
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("action", "mode"), "residual"),
        (("feature_names",), ["another_feature"]),
        (("allocator", "risk_aversion"), 9.0),
        (("feature_preprocessing",), "different_preprocessing"),
    ],
)
def test_candidate_identity_changes_when_decision_structure_changes(path, value):
    candidate, _, _ = capability()
    env = env_for(6, 8)
    changed = deepcopy(env.recipe)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    assert candidate(env.recipe) != candidate(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_capital", 2000.0),
        ("currency", "JPY"),
        ("decision_interval_seconds", 7200),
        ("economic_horizon_seconds", 14400),
        ("calendar_kind", "session_calendar"),
        ("execution_bar_hours", 2.0),
        ("insolvency_valuation", "other"),
    ],
)
def test_scenario_validation_rejects_non_economic_runtime_invariant_changes(
    field, value
):
    _, invariant, validate = capability()
    env = env_for(6, 8)
    changed = deepcopy(env.recipe)
    changed["runtime_profile"][field] = value
    assert invariant(env.recipe) != invariant(changed)
    with pytest.raises(ValueError, match="runtime invariant"):
        validate(
            env.recipe,
            changed,
            base_economics_digest=env.executor.execution_policy_digest,
            base_risk_digest=env.bound.objective.risk_digest,
            scenario_economics_digest=env.executor.execution_policy_digest,
            scenario_risk_digest=env.bound.objective.risk_digest,
        )


@pytest.mark.parametrize(
    "side", ["base_economics", "base_risk", "scenario_economics", "scenario_risk"]
)
def test_scenario_validation_rejects_declared_digest_mismatch(side):
    _, _, validate = capability()
    env = env_for(6, 8)
    kwargs = {
        "base_economics_digest": env.executor.execution_policy_digest,
        "base_risk_digest": env.bound.objective.risk_digest,
        "scenario_economics_digest": env.executor.execution_policy_digest,
        "scenario_risk_digest": env.bound.objective.risk_digest,
    }
    kwargs[side + "_digest" if not side.endswith("digest") else side] = "a" * 64
    # normalize the parametrized convenience key
    if side == "base_economics":
        kwargs["base_economics_digest"] = "a" * 64
    elif side == "base_risk":
        kwargs["base_risk_digest"] = "a" * 64
    elif side == "scenario_economics":
        kwargs["scenario_economics_digest"] = "a" * 64
    else:
        kwargs["scenario_risk_digest"] = "a" * 64
    with pytest.raises(ValueError, match="economics|risk"):
        validate(env.recipe, env.recipe, **kwargs)


def test_candidate_identity_does_not_mutate_recipe():
    candidate, invariant, _ = capability()
    env = env_for(6, 8)
    before = deepcopy(env.recipe)
    candidate(env.recipe)
    invariant(env.recipe)
    assert env.recipe == before
