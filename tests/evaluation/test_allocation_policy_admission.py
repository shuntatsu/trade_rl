from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
from trade_rl.artifacts import content_digest


def capability():
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
        admit_allocation_fold_policies,
        run_artifact_bound_continuous_allocation_walk_forward,
    )

    return (
        AllocationFoldPolicyArtifact,
        admit_allocation_fold_policies,
        run_artifact_bound_continuous_allocation_walk_forward,
    )


def fake_manifest(recipe_digest, name):
    return {"recipe_digest": recipe_digest, "policy": name}


class FakePolicy:
    def __init__(self, recipe_digest, name, actions):
        self.recipe_digest = recipe_digest
        self.manifest = fake_manifest(recipe_digest, name)
        self.actions = iter(actions)
        self.calls = []

    def action(self, observation, *, runtime_recipe_digest):
        assert isinstance(observation, np.ndarray)
        self.calls.append(runtime_recipe_digest)
        if runtime_recipe_digest != self.recipe_digest:
            raise ValueError("recipe mismatch")
        return next(self.actions)


def artifact(env, fold_index, root, name):
    AllocationFoldPolicyArtifact, _, _ = capability()
    return AllocationFoldPolicyArtifact(
        fold_index=fold_index,
        bundle_root=Path(root),
        expected_digest=content_digest(fake_manifest(env.recipe_digest, name)),
        expected_recipe_digest=env.recipe_digest,
    )


def test_admission_loads_every_artifact_before_returning_runtime_policies(
    monkeypatch, tmp_path
):
    _, admit, _ = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    declarations = (
        artifact(first, 0, tmp_path / "a", "a"),
        artifact(second, 1, tmp_path / "b", "b"),
    )
    loaded = [
        FakePolicy(first.recipe_digest, "a", (3, 0)),
        FakePolicy(second.recipe_digest, "b", (0, 2)),
    ]
    calls = []

    def fake_load(root, *, expected_digest, expected_recipe_digest):
        calls.append((Path(root), expected_digest, expected_recipe_digest))
        return loaded[len(calls) - 1]

    import trade_rl.evaluation.rl_allocation.policy_admission as module

    monkeypatch.setattr(module, "load_allocation_policy", fake_load)
    policies = admit(folds_for(first, second), (first, second), declarations)

    assert tuple(policy.policy_digest for policy in policies) == tuple(
        declaration.expected_digest for declaration in declarations
    )
    assert tuple(policy.recipe_digest for policy in policies) == (
        first.recipe_digest,
        second.recipe_digest,
    )
    assert calls == [
        (
            declarations[0].bundle_root,
            declarations[0].expected_digest,
            first.recipe_digest,
        ),
        (
            declarations[1].bundle_root,
            declarations[1].expected_digest,
            second.recipe_digest,
        ),
    ]
    assert not hasattr(first, "book")
    assert not hasattr(second, "book")


def test_artifact_bound_runner_uses_keyword_only_policy_action_and_verified_digests(
    monkeypatch, tmp_path
):
    _, _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    declarations = (
        artifact(first, 0, tmp_path / "a", "a"),
        artifact(second, 1, tmp_path / "b", "b"),
    )
    loaded = [
        FakePolicy(first.recipe_digest, "a", (3, 0)),
        FakePolicy(second.recipe_digest, "b", (0, 2)),
    ]

    import trade_rl.evaluation.rl_allocation.policy_admission as module

    monkeypatch.setattr(
        module,
        "load_allocation_policy",
        lambda root, **_kwargs: loaded[
            0 if Path(root) == declarations[0].bundle_root else 1
        ],
    )
    result = run(
        folds_for(first, second),
        (first, second),
        declarations,
        reset_seed=7,
    )

    assert result.policy_digests == tuple(
        declaration.expected_digest for declaration in declarations
    )
    assert loaded[0].calls == [first.recipe_digest, first.recipe_digest]
    assert loaded[1].calls == [second.recipe_digest, second.recipe_digest]
    assert second.book.quantities[0] == 0.0


def test_invalid_second_artifact_fails_before_first_oos_reset(monkeypatch, tmp_path):
    _, _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    declarations = (
        artifact(first, 0, tmp_path / "good", "good"),
        artifact(second, 1, tmp_path / "bad", "bad"),
    )
    first_policy = FakePolicy(first.recipe_digest, "good", (3, 0))

    def fake_load(root, **_kwargs):
        if Path(root) == declarations[0].bundle_root:
            return first_policy
        raise ValueError("tampered allocation policy")

    import trade_rl.evaluation.rl_allocation.policy_admission as module

    monkeypatch.setattr(module, "load_allocation_policy", fake_load)
    with pytest.raises(ValueError, match="tampered"):
        run(folds_for(first, second), (first, second), declarations)
    assert not hasattr(first, "book")
    assert not hasattr(second, "book")
    assert first_policy.calls == []


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda first, second, values: (
                values[1],
                values[0],
            ),
            "fold",
        ),
        (
            lambda first, second, values: (
                values[0],
                type(values[1])(
                    fold_index=values[1].fold_index,
                    bundle_root=values[1].bundle_root,
                    expected_digest=values[1].expected_digest,
                    expected_recipe_digest="0" * 64,
                ),
            ),
            "recipe",
        ),
    ],
)
def test_admission_rejects_declaration_mismatch_before_loading(
    monkeypatch, tmp_path, mutator, match
):
    _, admit, _ = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    values = (
        artifact(first, 0, tmp_path / "a", "a"),
        artifact(second, 1, tmp_path / "b", "b"),
    )
    declarations = mutator(first, second, values)
    import trade_rl.evaluation.rl_allocation.policy_admission as module

    monkeypatch.setattr(
        module,
        "load_allocation_policy",
        lambda *_args, **_kwargs: pytest.fail("loader must not run"),
    )
    with pytest.raises(ValueError, match=match):
        admit(folds_for(first, second), (first, second), declarations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fold_index": True},
        {"fold_index": -1},
        {"expected_digest": "bad"},
        {"expected_recipe_digest": "bad"},
    ],
)
def test_artifact_declaration_rejects_malformed_identity(tmp_path, kwargs):
    AllocationFoldPolicyArtifact, _, _ = capability()
    values = dict(
        fold_index=0,
        bundle_root=tmp_path / "policy",
        expected_digest="0" * 64,
        expected_recipe_digest="1" * 64,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        AllocationFoldPolicyArtifact(**values)


@pytest.mark.parametrize("change", ["gap", "dataset", "account"])
def test_full_continuous_chain_is_rejected_before_any_artifact_deserialization(
    monkeypatch, tmp_path, change
):
    _, admit, _ = capability()
    first = env_for(6, 8)
    if change == "gap":
        second = env_for(9, 10)
    elif change == "dataset":
        from dataclasses import replace

        second = env_for(8, 10, dataset=replace(first.dataset))
    else:
        second = env_for(8, 10, account_id="different-account")
    declarations = (
        artifact(first, 0, tmp_path / "a", "a"),
        artifact(second, 1, tmp_path / "b", "b"),
    )

    import trade_rl.evaluation.rl_allocation.policy_admission as module

    monkeypatch.setattr(
        module,
        "load_allocation_policy",
        lambda *_args, **_kwargs: pytest.fail(
            "artifact deserializer must not run before continuous-chain preflight"
        ),
    )
    with pytest.raises(ValueError):
        admit(folds_for(first, second), (first, second), declarations)
    assert not hasattr(first, "book")
    assert not hasattr(second, "book")
