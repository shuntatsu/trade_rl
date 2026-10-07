"""One global native clock; segment boundaries only switch frozen policies."""

from dataclasses import replace
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_continuation import book_economics, env_for
from tests.evaluation.test_allocation_continuous_walk_forward import fold, folds_for
from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    ContinuousAllocationWalkForwardResult,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import IndexRange


def capability():
    name = "trade_rl.evaluation.rl_allocation.global_walk_forward"
    assert util.find_spec(name) is not None, "global native-clock consumer is absent"
    return import_module(name)


def v2_env(start=6, stop=10, *, cost=None):
    old = env_for(start, stop, cost=cost)
    args = parameters(stop=10) | {
        "dataset": old.dataset,
        "stream": old.stream,
        "bound": old.bound,
        "account_id": old.account_id,
        "start_index": start,
        "stop_index": stop,
        "execution_cost": old.execution_cost,
        "estimates": tuple(old._estimates.values()),
    }
    return AllocationTradingEnv(**opt_in(args))


def declared(env, action, name="policy"):
    return AllocationFoldPolicy(
        content_digest({"policy": name}), env.recipe_digest, action
    )


def test_global_h4_actor_clock_changes_actual_account_from_local_h2():
    module = capability()
    global_env = v2_env()
    global_times, local_times = [], []

    def rule(record):
        def action(observation, _recipe):
            record.append(float(observation[22]))
            return 3 if observation[22] >= 0.75 else 2

        return action

    result = module.run_global_allocation_walk_forward(
        (fold(0, 6, 8), fold(1, 8, 10)),
        global_env,
        (declared(global_env, rule(global_times)),) * 2,
    )
    local = (v2_env(6, 8), v2_env(8, 10))
    run_continuous_allocation_walk_forward(
        folds_for(*local), local, tuple(declared(e, rule(local_times)) for e in local)
    )
    assert global_times == [1.0, 0.75, 0.5, 0.25]
    assert local_times == [1.0, 0.5, 1.0, 0.5]
    assert global_env.book.cash == pytest.approx(1047.9)
    assert global_env.book.cash != local[-1].book.cash
    assert result.stitched.boundaries == ((6, 8), (8, 10))
    assert type(result) is module.GlobalAllocationWalkForwardResult
    assert not isinstance(result, ContinuousAllocationWalkForwardResult)


@pytest.mark.parametrize("boundary", [7, 9])
@pytest.mark.parametrize("partial", [False, True])
def test_unequal_segments_preserve_orders_counters_risk_rng_and_single_reset(
    boundary, partial, monkeypatch
):
    module = capability()
    cost = replace(
        parameters()["execution_cost"],
        slippage_std=0.01,
        random_seed=73,
        max_participation_rate=0.00001 if partial else 1.0,
    )
    direct, segmented = v2_env(cost=cost), v2_env(cost=cost)
    seen, boundary_digests, economics, resets = [], {}, {}, []
    action_iter = iter((3, 3, 3, 2))
    observation, _ = direct.reset(seed=7)
    for _ in range(4):
        action = next(action_iter)
        seen.append(observation.copy())
        observation, _, terminal, truncated, info = direct.step(action)
        assert not truncated
        economics[direct.index] = book_economics(direct.book)
        boundary_digests[direct.index] = allocation_state_digest(direct)
        assert terminal is (direct.index == 10)
    original = segmented.reset

    def reset(**kwargs):
        resets.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(segmented, "reset", reset)
    actions = iter((3, 3, 3, 2))
    actor_observations = []

    def action(observation, _recipe):
        actor_observations.append(observation.copy())
        assert book_economics(segmented.book) == economics.get(
            segmented.index, book_economics(segmented.book)
        )
        if segmented.index == boundary:
            assert not segmented._terminated
            assert allocation_state_digest(segmented) == boundary_digests[boundary]
            if partial:
                assert segmented.book.quantities[0] > 0
                assert segmented.order_book.active_orders
        return next(actions)

    result = module.run_global_allocation_walk_forward(
        (fold(0, 6, boundary), fold(1, boundary, 10)),
        segmented,
        (declared(segmented, action, "first"), declared(segmented, action, "second")),
        reset_seed=7,
    )
    assert resets == [{"seed": 7}]
    np.testing.assert_array_equal(actor_observations, seen)
    assert book_economics(segmented.book) == book_economics(direct.book)
    assert allocation_state_digest(segmented) == allocation_state_digest(direct)
    assert result.folds[0].closing_state_digest == boundary_digests[boundary]
    assert result.folds[1].opening_state_digest == boundary_digests[boundary]
    assert result.stitched.diagnostics.n_trades == direct.book.fill_count
    assert result.stitched.diagnostics.total_cost == direct.book.total_cost


def test_literal_fee_quantity_cash_and_risk_oracle():
    module = capability()
    env = v2_env()
    actions = iter((3, 0, 0, 2))
    observed = []

    def action(_observation, _recipe):
        observed.append((env.book.cash, env.book.quantities[0], env.book.total_cost))
        return next(actions)

    result = module.run_global_allocation_walk_forward(
        (fold(0, 6, 7), fold(1, 7, 10)), env, (declared(env, action),) * 2
    )
    assert observed[1] == (499.0, 5.0, 1.0)
    assert env.book.cash == pytest.approx(1047.9)
    assert env.book.quantities[0] == 0.0
    assert env.book.total_cost == pytest.approx(2.1)
    assert result.stitched.diagnostics.rebalance_events == 2
    assert env.risk.config == env.risk_config


class Alias(int):
    pass


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "duplicate",
        "reversed",
        "range_alias",
        "fold_alias",
        "purge_alias",
        "late_recipe",
        "H2",
        "seed_bool",
        "seed_alias",
        "roster_list",
    ],
)
def test_all_declared_segments_are_rejected_before_first_reset(change):
    module = capability()
    env = v2_env()
    folds = (fold(0, 6, 8), fold(1, 8, 10))
    policies = (declared(env, lambda *_: 0),) * 2
    seed = None
    if change == "gap":
        folds = (folds[0], fold(1, 9, 10))
    elif change == "duplicate":
        folds = (folds[0], replace(folds[1], fold_index=0))
    elif change == "reversed":
        folds = folds[::-1]
    elif change == "range_alias":
        folds = (folds[0], replace(folds[1], test=IndexRange(Alias(8), 10)))
    elif change == "fold_alias":
        folds = (folds[0], replace(folds[1], fold_index=Alias(1)))
    elif change == "purge_alias":
        folds = (folds[0], replace(folds[1], purge_bars=False))
    elif change == "late_recipe":
        policies = (policies[0], replace(policies[1], recipe_digest="f" * 64))
    elif change == "H2":
        policies = (declared(v2_env(6, 8), lambda *_: 0),) * 2
    elif change.startswith("seed"):
        seed = True if change == "seed_bool" else Alias(7)
    else:
        policies = list(policies)
    with pytest.raises((ValueError, RuntimeError)):
        module.run_global_allocation_walk_forward(folds, env, policies, reset_seed=seed)
    assert not hasattr(env, "book")


@pytest.mark.parametrize("action", [True, np.bool_(False), 4, 1.0, None])
def test_malformed_action_is_rejected_without_native_execution(action):
    module = capability()
    env = v2_env()
    with pytest.raises(ValueError, match="action"):
        module.run_global_allocation_walk_forward(
            (fold(0, 6, 10),), env, (declared(env, lambda *_: action),)
        )
    assert env.index == 6 and env.book.fill_count == 0


def test_late_runtime_mutation_is_rejected_before_next_action():
    module = capability()
    env = v2_env()

    def change(_obs, _recipe):
        env.bound = replace(
            env.bound, clock=replace(env.bound.clock, economic_horizon_seconds=7200)
        )
        return 0

    with pytest.raises(ValueError):
        module.run_global_allocation_walk_forward(
            (fold(0, 6, 7), fold(1, 7, 10)),
            env,
            (declared(env, lambda *_: 0), declared(env, change)),
        )
    assert env.index == 7


@pytest.mark.parametrize("change", ["fold", "policy"])
def test_late_roster_mutation_cannot_redefine_declared_segments(change):
    module = capability()
    env = v2_env()
    folds = (fold(0, 6, 7), fold(1, 7, 10))
    policies = (declared(env, lambda *_: 0), declared(env, lambda *_: 0))

    def mutate(_obs, _recipe):
        if change == "fold":
            object.__setattr__(folds[1].test, "start", 8)
        else:
            object.__setattr__(policies[1], "policy_digest", "f" * 64)
        return 0

    policies = (declared(env, mutate), policies[1])
    with pytest.raises(ValueError, match="roster"):
        module.run_global_allocation_walk_forward(folds, env, policies)
    assert env.index == 6


@pytest.mark.parametrize(
    "change", ["fold_bool", "range_scalar_alias", "range_type_alias", "equal_callable"]
)
def test_equal_comparing_roster_swaps_fail_before_native_execution(change):
    module = capability()
    env = v2_env()
    folds = (fold(0, 6, 7), fold(1, 7, 10))

    class RangeAlias(IndexRange):
        pass

    class EqualAction:
        def __init__(self, action):
            self.action = action

        def __call__(self, *_):
            return self.action

        def __eq__(self, _other):
            return True

    later = declared(env, EqualAction(0))

    def mutate(_obs, _recipe):
        if change == "fold_bool":
            object.__setattr__(folds[1], "fold_index", True)
        elif change == "range_scalar_alias":
            object.__setattr__(folds[1].test, "start", Alias(7))
        elif change == "range_type_alias":
            object.__setattr__(folds[1], "test", RangeAlias(7, 10))
        else:
            replacement = EqualAction(3)
            assert later.action == replacement
            assert later.action is not replacement
            object.__setattr__(later, "action", replacement)
        return 0

    with pytest.raises(ValueError, match="roster"):
        module.run_global_allocation_walk_forward(
            folds, env, (declared(env, mutate), later)
        )
    assert env.index == 6 and env.book.fill_count == 0


def test_malformed_range_claim_is_rejected_before_reset():
    module = capability()
    env = v2_env()
    folds = (fold(0, 6, 10),)
    object.__setattr__(folds[0].train, "stop", 0)
    with pytest.raises(ValueError):
        module.run_global_allocation_walk_forward(
            folds, env, (declared(env, lambda *_: 0),)
        )
    assert not hasattr(env, "book")


def test_native_economic_termination_cannot_publish_segment_result():
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args

    module = capability()
    env = AllocationTradingEnv(**frozen_args(stop=9, early=True))
    with pytest.raises(ValueError, match="economic termination"):
        module.run_global_allocation_walk_forward(
            (fold(0, 6, 7), fold(1, 7, 9)), env, (declared(env, lambda *_: 1),) * 2
        )
    assert env.index == 7
    assert env.book.cash == -501.0
    assert env.book.termination_reason is not None


def test_literal_partial_fill_pending_order_survives_active_boundary():
    module = capability()
    env = v2_env(
        cost=replace(parameters()["execution_cost"], max_participation_rate=0.00001)
    )

    def maintain(_obs, _recipe):
        if env.index == 7:
            assert env.book.cash == 899.8
            assert env.book.quantities[0] == 1.0
            assert env.book.total_cost == 0.2
            assert env.order_book.active_orders[0].remaining_quantity == 4.0
            assert not env._terminated
        return 3

    module.run_global_allocation_walk_forward(
        (fold(0, 6, 7), fold(1, 7, 10)), env, (declared(env, maintain),) * 2
    )


def test_legacy_none_admission_keeps_load_then_bind_error_order(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import trade_rl.evaluation.rl_allocation.global_walk_forward as owner
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration

    base, stress = native_pair()
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    calls = []
    monkeypatch.setattr(
        owner,
        "load_allocation_policy",
        lambda *a, **k: (calls.append(a), SimpleNamespace(manifest={}))[1],
    )
    with pytest.raises(ValueError, match="admitted identity"):
        owner.run_artifact_bound_global_allocation_walk_forward(
            folds, base, artifacts, collector=None
        )
    assert len(calls) == 1 and not hasattr(base, "book")


def test_legacy_none_bad_second_zip_loads_first_before_original_failure(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from shutil import copytree

    import trade_rl.evaluation.rl_allocation.global_walk_forward as owner
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration

    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    bad = tmp_path / "second"
    copytree(root, bad)
    (bad / "policy.zip").write_bytes(b"bad second")
    artifacts = (artifacts[0], replace(artifacts[1], bundle_root=bad))
    calls = []
    monkeypatch.setattr(
        archive, "_load_policy", lambda path: (calls.append(path), model)[1]
    )
    with pytest.raises(
        ValueError, match="allocation policy changed during verified copy"
    ):
        owner.run_artifact_bound_global_allocation_walk_forward(
            folds, base, artifacts, collector=None
        )
    assert len(calls) == 1 and not hasattr(base, "book")
