from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
from tests.evaluation.test_allocation_policy_admission import artifact
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.robustness.walk_forward.sealed_test import (
    SealedTestLedger,
    build_sealed_test_access_record,
)


def capability():
    from trade_rl.evaluation.rl_allocation.sealed_policy_admission import (
        SealedAllocationFoldPolicyArtifact,
        run_sealed_artifact_bound_continuous_allocation_walk_forward,
        validate_sealed_allocation_fold_artifacts,
    )

    return (
        SealedAllocationFoldPolicyArtifact,
        validate_sealed_allocation_fold_artifacts,
        run_sealed_artifact_bound_continuous_allocation_walk_forward,
    )


def authorized(fold, env, declaration, plan, configuration):
    Binding, _, _ = capability()
    ledger = SealedTestLedger()
    access = ledger.authorize_once(
        experiment_plan_digest=plan,
        dataset_id=env.dataset.dataset_id,
        fold_index=fold.fold_index,
        test_range=fold.test,
        selected_configuration=configuration,
        selected_policy_digest=declaration.expected_digest,
    )
    return Binding(access=access, artifact=declaration)


def setup(tmp_path):
    first, second = env_for(6, 8), env_for(8, 10)
    folds = folds_for(first, second)
    plan = content_digest({"plan": "allocation-continuous-oos"})
    declarations = (
        artifact(first, folds[0].fold_index, tmp_path / "a", "a"),
        artifact(second, folds[1].fold_index, tmp_path / "b", "b"),
    )
    bindings = (
        authorized(folds[0], first, declarations[0], plan, "candidate-a"),
        authorized(folds[1], second, declarations[1], plan, "candidate-b"),
    )
    return first, second, folds, plan, declarations, bindings


def test_authorized_bindings_match_fold_dataset_range_and_selected_policy(tmp_path):
    _, validate, _ = capability()
    first, second, folds, plan, _declarations, bindings = setup(tmp_path)
    validate(folds, (first, second), bindings, experiment_plan_digest=plan)


@pytest.mark.parametrize(
    "change", ["plan", "dataset", "fold", "range", "policy", "digest"]
)
def test_authorization_mismatch_fails_before_policy_artifact_loader(
    monkeypatch, tmp_path, change
):
    Binding, validate, _ = capability()
    first, second, folds, plan, declarations, bindings = setup(tmp_path)
    access = bindings[1].access
    if change == "plan":
        supplied_plan = "f" * 64
        changed = access
    elif change == "dataset":
        supplied_plan = plan
        changed = replace(access, dataset_id="f" * 64)
    elif change == "fold":
        supplied_plan = plan
        changed = replace(access, fold_index=99)
    elif change == "range":
        supplied_plan = plan
        changed = replace(
            access,
            test_range=replace(access.test_range, start=access.test_range.start - 1),
        )
    elif change == "policy":
        supplied_plan = plan
        changed = replace(access, selected_policy_digest="f" * 64)
    else:
        supplied_plan = plan
        changed = replace(access, access_digest="f" * 64)
    changed_bindings = (bindings[0], Binding(changed, declarations[1]))

    import trade_rl.evaluation.rl_allocation.sealed_policy_admission as module

    monkeypatch.setattr(
        module,
        "admit_allocation_fold_policies",
        lambda *_args, **_kwargs: pytest.fail(
            "artifact loader must not run before sealed authorization validation"
        ),
    )
    with pytest.raises(ValueError):
        validate(
            folds,
            (first, second),
            changed_bindings,
            experiment_plan_digest=supplied_plan,
        )


def test_none_selected_policy_is_not_an_executable_outer_test_authorization(tmp_path):
    Binding, validate, _ = capability()
    first, second, folds, plan, declarations, bindings = setup(tmp_path)
    access = build_sealed_test_access_record(
        experiment_plan_digest=plan,
        dataset_id=second.dataset.dataset_id,
        fold_index=folds[1].fold_index,
        test_range=folds[1].test,
        selected_configuration="no-policy",
        selected_policy_digest=None,
    )
    values = (bindings[0], Binding(access, declarations[1]))
    with pytest.raises(ValueError, match="selected policy"):
        validate(folds, (first, second), values, experiment_plan_digest=plan)


def test_all_bundles_are_admitted_before_first_oos_reset(monkeypatch, tmp_path):
    _, _, run = capability()
    first, second, folds, plan, _declarations, bindings = setup(tmp_path)
    calls = []
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )

    first_actions = iter((3, 0))
    second_actions = iter((0, 2))
    policies = (
        AllocationFoldPolicy(
            bindings[0].artifact.expected_digest,
            first.recipe_digest,
            lambda _observation, _recipe: next(first_actions),
        ),
        AllocationFoldPolicy(
            bindings[1].artifact.expected_digest,
            second.recipe_digest,
            lambda _observation, _recipe: next(second_actions),
        ),
    )

    import trade_rl.evaluation.rl_allocation.sealed_policy_admission as module

    def admit(_folds, environments, artifacts):
        calls.append(tuple(value.expected_digest for value in artifacts))
        assert not hasattr(environments[0], "book")
        assert not hasattr(environments[1], "book")
        return policies

    monkeypatch.setattr(module, "admit_allocation_fold_policies", admit)
    result = run(
        folds,
        (first, second),
        bindings,
        experiment_plan_digest=plan,
        reset_seed=7,
    )
    assert calls == [
        (
            bindings[0].artifact.expected_digest,
            bindings[1].artifact.expected_digest,
        )
    ]
    assert result.policy_digests == tuple(
        value.access.selected_policy_digest for value in bindings
    )
    assert second.index == second.stop_index
