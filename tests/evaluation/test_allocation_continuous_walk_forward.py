[Reading 309 lines from start (total: 309 lines, 0 remaining)]

from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_continuation import book_economics, env_for
from tests.evaluation.test_allocation_rl_env import parameters
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.evaluation.robustness.walk_forward.stitching import StitchMode


def capability():
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
        run_continuous_allocation_walk_forward,
    )

    return AllocationFoldPolicy, run_continuous_allocation_walk_forward


def fold(index, start, stop):
    train_stop = start - 4
    return WalkForwardFold(
        fold_index=index,
        train=IndexRange(0, train_stop),
        checkpoint_validation=IndexRange(train_stop, train_stop + 1),
        configuration_selection=IndexRange(train_stop + 1, start),
        test=IndexRange(start, stop),
        purge_bars=0,
    )


def folds_for(*environments):
    return tuple(
        fold(index, env.start_index, env.stop_index)
        for index, env in enumerate(environments)
    )


def scripted(*actions):
    values = iter(actions)

    def action(_observation, _recipe_digest):
        return next(values)

    return action


def policy(env, name, *actions):
    AllocationFoldPolicy, _ = capability()
    return AllocationFoldPolicy(
        policy_digest=content_digest({"policy": name}),
        recipe_digest=env.recipe_digest,
        action=scripted(*actions),
    )


def test_runner_switches_policies_without_resetting_the_account_chain():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    first_policy = policy(first, "first", 3, 0)
    second_policy = policy(second, "second", 0, 2)

    result = run(
        folds_for(first, second),
        (first, second),
        (first_policy, second_policy),
        reset_seed=7,
    )

    assert result.policy_digests == (
        first_policy.policy_digest,
        second_policy.policy_digest,
    )
    assert len(result.folds) == 2
    assert result.folds[0].closing_state_digest == result.folds[1].opening_state_digest
    assert result.stitched.mode is StitchMode.CONTINUOUS_ACCOUNT
    assert result.stitched.boundaries == ((6, 8), (8, 10))
    assert result.stitched.gaps == ()
    assert second.book.quantities[0] == 0.0
    assert second.book.fill_count == 2


def test_runner_uses_canonical_interval_returns_and_fold_counter_deltas():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    result = run(
        folds_for(first, second),
        (first, second),
        (
            policy(first, "first", 3, 0),
            policy(second, "second", 0, 2),
        ),
    )
    left, right = result.folds
    assert len(left.returns.values) == len(right.returns.values) == 2
    assert left.returns.values[0] == pytest.approx(0.049)
    assert left.diagnostics.n_trades == 1
    assert left.diagnostics.total_cost == pytest.approx(1.0)
    assert right.diagnostics.n_trades == 1
    assert right.diagnostics.total_cost == pytest.approx(1.1)
    assert result.stitched.diagnostics.n_trades == 2
    assert result.stitched.diagnostics.total_cost == pytest.approx(2.1)


@pytest.mark.parametrize(
    "change,match",
    [
        ("gap", "contiguous"),
        ("recipe", "recipe"),
        ("dataset", "Dataset"),
    ],
)
def test_runner_preflight_rejects_invalid_fold_chain_before_first_reset(change, match):
    AllocationFoldPolicy, run = capability()
    first = env_for(6, 8)
    if change == "gap":
        second = env_for(9, 10)
    elif change == "dataset":
        second = env_for(8, 10, dataset=replace(first.dataset))
    else:
        second = env_for(8, 10)
    left = policy(first, "first", 3, 0)
    right = policy(second, "second", 0, 2)
    if change == "recipe":
        right = AllocationFoldPolicy(
            policy_digest=right.policy_digest,
            recipe_digest="0" * 64,
            action=right.action,
        )

    with pytest.raises(ValueError, match=match):
        run(folds_for(first, second), (first, second), (left, right))
    assert not hasattr(first, "book")
    assert not hasattr(second, "book")


def test_runner_rejects_reused_or_already_initialized_environments():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    first.reset()
    with pytest.raises(RuntimeError, match="fresh|initialized"):
        run(
            folds_for(first, second),
            (first, second),
            (policy(first, "first", 3, 0), policy(second, "second", 0, 2)),
        )


def test_policy_identity_is_closed_and_action_must_be_callable():
    AllocationFoldPolicy, _ = capability()
    with pytest.raises(ValueError, match="policy_digest"):
        AllocationFoldPolicy("bad", "0" * 64, lambda _o, _r: 0)
    with pytest.raises(ValueError, match="recipe_digest"):
        AllocationFoldPolicy("0" * 64, "bad", lambda _o, _r: 0)
    with pytest.raises(ValueError, match="callable"):
        AllocationFoldPolicy("0" * 64, "1" * 64, None)


def test_runner_fails_closed_on_economic_termination_instead_of_stitching():
    AllocationFoldPolicy, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)

    def terminate(_observation, _recipe_digest):
        first.book.terminate("drawdown_stop")
        return 0

    terminating = AllocationFoldPolicy(
        policy_digest=content_digest({"policy": "terminating"}),
        recipe_digest=first.recipe_digest,
        action=terminate,
    )
    with pytest.raises(ValueError, match="economic termination|live, solvent"):
        run(
            folds_for(first, second),
            (first, second),
            (terminating, policy(second, "second", 0, 2)),
        )
    assert first.book.termination_reason is not None
    assert not hasattr(second, "book")


def test_stitched_simple_returns_compound_to_actual_continuous_equity():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    result = run(
        folds_for(first, second),
        (first, second),
        (
            policy(first, "first", 3, 0),
            policy(second, "second", 0, 2),
        ),
    )
    assert compound_return(result.stitched.returns.values) == pytest.approx(
        second.book.portfolio_value / second.initial_capital - 1.0
    )


def test_runner_preserves_declared_fold_indices_in_evidence():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    declared = (fold(4, 6, 8), fold(9, 8, 10))
    result = run(
        declared,
        (first, second),
        (
            policy(first, "first", 3, 0),
            policy(second, "second", 0, 2),
        ),
    )
    assert result.stitched.fold_indices == (4, 9)
    assert tuple(value.fold_index for value in result.folds) == (4, 9)


def test_runner_rejects_fold_test_range_mismatch_before_reset():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    bad = (fold(0, 6, 8), fold(1, 7, 10))
    with pytest.raises(ValueError, match="fold test range|contiguous"):
        run(
            bad,
            (first, second),
            (
                policy(first, "first", 3, 0),
                policy(second, "second", 0, 2),
            ),
        )
    assert not hasattr(first, "book")
    assert not hasattr(second, "book")


@pytest.mark.parametrize("bad_action", [True, -1, 4, 1.5])
def test_runner_rejects_noncanonical_policy_action_codes(bad_action):
    AllocationFoldPolicy, run = capability()
    first = env_for(6, 8)
    declared = (fold(0, 6, 8),)
    bad = AllocationFoldPolicy(
        policy_digest=content_digest(
            {"policy": "bad-action", "value": str(bad_action)}
        ),
        recipe_digest=first.recipe_digest,
        action=lambda _o, _r: bad_action,
    )
    with pytest.raises(ValueError, match="one integer"):
        run(declared, (first,), (bad,))


def test_runner_preserves_stochastic_execution_path_against_uninterrupted_account():
    _, run = capability()
    base = parameters(stop=10)
    cost = replace(
        base["execution_cost"],
        slippage_std=0.01,
        random_seed=73,
        max_participation_rate=1.0,
    )
    whole = env_for(6, 10, cost=cost)
    first, second = env_for(6, 8, cost=cost), env_for(8, 10, cost=cost)

    whole.reset(seed=9)
    for action in (3, 1, 3, 1):
        whole.step(action)

    result = run(
        folds_for(first, second),
        (first, second),
        (
            policy(first, "first", 3, 1),
            policy(second, "second", 3, 1),
        ),
    )
    assert book_economics(second.book) == pytest.approx(book_economics(whole.book))
    assert compound_return(result.stitched.returns.values) == pytest.approx(
        whole.book.portfolio_value / whole.initial_capital - 1.0
    )


def test_runner_requires_fold_indices_to_increase_with_chronology():
    _, run = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    declared = (fold(9, 6, 8), fold(4, 8, 10))
    with pytest.raises(ValueError, match="indices.*increase"):
        run(
            declared,
            (first, second),
            (
                policy(first, "first", 3, 0),
                policy(second, "second", 0, 2),
            ),
        )
    assert not hasattr(first, "book")


@pytest.mark.parametrize("reset_seed", [True, -1, 1.5])
def test_runner_rejects_invalid_reset_seed_before_account_reset(reset_seed):
    _, run = capability()
    first = env_for(6, 8)
    with pytest.raises(ValueError, match="reset_seed"):
        run(
            (fold(0, 6, 8),),
            (first,),
            (policy(first, "first", 3, 0),),
            reset_seed=reset_seed,
        )
    assert not hasattr(first, "book")

[executed on device: DESKTOP-MTD23AI (0abc841c-0215-491a-8eb3-2ddd8c73feac)]