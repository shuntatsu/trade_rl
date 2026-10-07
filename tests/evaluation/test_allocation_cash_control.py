from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.evaluation.test_allocation_comparison_evidence import contract_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
from tests.evaluation.test_allocation_rl_env import parameters
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)


def capability():
    from trade_rl.evaluation.allocation_cash_control import (
        AllocationCashControlEvidence,
        allocation_cash_control_policy_digest,
        build_allocation_cash_control_evidence,
        run_continuous_allocation_cash_control,
    )

    return (
        AllocationCashControlEvidence,
        allocation_cash_control_policy_digest,
        build_allocation_cash_control_evidence,
        run_continuous_allocation_cash_control,
    )


def cash_fixture(*, annual_rate=0.876):
    base = parameters(stop=10)
    original = base["dataset"]
    dataset_id = content_digest(
        {
            "base_dataset_id": original.dataset_id,
            "cash_rate": annual_rate,
            "purpose": "allocation-cash-control-test",
        }
    )
    dataset = replace(
        original,
        dataset_id=dataset_id,
        cash_rate=np.full(len(original.timestamps), annual_rate, dtype=np.float64),
    )
    stream = replace(base["stream"], dataset_id=dataset_id)

    def env_for(start, stop):
        from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv

        args = parameters(stop=10)
        args["dataset"] = dataset
        args["stream"] = stream
        args["start_index"], args["stop_index"] = start, stop
        args["account_id"] = "cash-control-S0"
        args["estimates"] = tuple(
            estimate
            for estimate in args["estimates"]
            if dataset.timestamps[start]
            <= estimate.decision_time
            < dataset.timestamps[stop]
        )
        execution = MarketExecutor(
            dataset, args["execution_cost"], insolvency_valuation="retain_debt"
        ).execution_policy_digest
        objective = args["bound"].objective
        clock = replace(
            args["bound"].clock,
            economic_horizon_seconds=(stop - start) * 3600,
            rollout_steps=max(1, stop - start),
        )
        profile = AllocationRuntimeProfile(
            execution,
            content_digest(args["risk_config"]),
            objective.capital.initial_equities[0],
            objective.capital.currency,
            clock.decision_interval_seconds,
            clock.economic_horizon_seconds,
        )
        recipe = allocation_recipe_digest(
            args["action_contract"],
            ("signal",),
            allocator=args["allocator"],
            expected_horizon_seconds=3600,
            runtime_profile=profile,
        )

        def utc(index):
            stamp = dataset.timestamps[index].astype("datetime64[us]").astype(datetime)
            return stamp.replace(tzinfo=UTC)

        args["bound"] = BoundObjectiveClock(
            replace(
                objective,
                evaluation_start=utc(start),
                evaluation_stop_exclusive=utc(stop),
                economics_digest=execution,
                risk_digest=content_digest(args["risk_config"]),
                deployment_recipe_digest=recipe,
            ),
            clock,
        )
        return AllocationTradingEnv(**args)

    first, second = env_for(6, 8), env_for(8, 10)
    folds = folds_for(first, second)
    return first, second, folds


def test_cash_control_executes_actual_cash_interest_without_orders():
    _, policy_digest, _build, run = capability()
    first, second, folds = cash_fixture()
    result = run(folds, (first, second))

    expected = 1000.0 * (1.0 + 0.876 / 8760.0) ** 4
    assert second.book.cash == pytest.approx(expected)
    assert second.book.portfolio_value == pytest.approx(expected)
    assert second.book.quantities[0] == 0.0
    assert second.book.fill_count == 0
    assert second.book.turnover_total == 0.0
    assert second.book.total_cost == 0.0
    assert result.policy_digests == (
        policy_digest(contract_for(first, folds)),
        policy_digest(contract_for(first, folds)),
    )
    assert sum(result.stitched.returns.values) != 0.0


def test_cash_control_evidence_keeps_actual_economic_return_and_roundtrips():
    Evidence, policy_digest, build, run = capability()
    first, second, folds = cash_fixture()
    declaration = contract_for(first, folds)
    result = run(folds, (first, second))
    evidence = build(
        declaration,
        scenario="base",
        folds=folds,
        environments=(first, second),
        result=result,
        validity_evidence_digest=content_digest(
            {"validity": "cash-independent-oracle"}
        ),
    )

    expected = second.book.portfolio_value / first.initial_capital - 1.0
    assert evidence.terminal_profit_rate == pytest.approx(expected)
    assert evidence.terminal_profit_rate > 0.0
    assert evidence.control_policy_digest == policy_digest(declaration)
    assert evidence.carrier_recipe_digest == declaration.direct_recipe_digest
    assert evidence.max_drawdown == pytest.approx(second.book.max_drawdown)
    assert evidence.coverage_complete is True
    assert evidence.termination_reason is None
    restored = Evidence.from_payload(evidence.payload())
    assert canonical_json_bytes(restored.payload()) == canonical_json_bytes(
        evidence.payload()
    )
    assert restored.digest == evidence.digest


def test_cash_control_is_seedless_and_policy_digest_is_scenario_independent():
    _, policy_digest, _build, _run = capability()
    first, _, folds = cash_fixture()
    base = contract_for(first, folds)
    stressed = replace(
        base,
        scenarios=(
            base.scenarios[0],
            replace(
                base.scenarios[0],
                name="cost_2x",
                digest="a" * 64,
                required=False,
            ),
        ),
    )
    assert policy_digest(base) == policy_digest(stressed)


@pytest.mark.parametrize(
    "change,match",
    [
        ("recipe", "recipe"),
        ("dataset", "Dataset"),
        ("scenario", "scenario"),
    ],
)
def test_cash_control_context_mismatch_fails_closed(change, match):
    _, _policy, build, run = capability()
    first, second, folds = cash_fixture()
    declaration = contract_for(first, folds)
    result = run(folds, (first, second))
    if change == "recipe":
        declaration = replace(declaration, direct_recipe_digest="a" * 64)
    elif change == "dataset":
        declaration = replace(declaration, dataset_id="a" * 64)
    else:
        declaration = replace(
            declaration,
            scenarios=(replace(declaration.scenarios[0], digest="a" * 64),),
        )

    with pytest.raises(ValueError, match=match):
        build(
            declaration,
            scenario="base",
            folds=folds,
            environments=(first, second),
            result=result,
            validity_evidence_digest="b" * 64,
        )


def test_cash_control_rejects_a_result_with_any_trade_or_nonflat_final_book():
    _, _policy, build, _run = capability()
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
        run_continuous_allocation_walk_forward,
    )

    first, second, folds = cash_fixture()
    declaration = contract_for(first, folds)
    policies = (
        AllocationFoldPolicy("1" * 64, first.recipe_digest, lambda _o, _r: 3),
        AllocationFoldPolicy("2" * 64, second.recipe_digest, lambda _o, _r: 0),
    )
    result = run_continuous_allocation_walk_forward(
        folds, (first, second), policies, reset_seed=0
    )
    with pytest.raises(ValueError, match="cash control|flat|trade"):
        build(
            declaration,
            scenario="base",
            folds=folds,
            environments=(first, second),
            result=result,
            validity_evidence_digest="c" * 64,
        )


def global_cash_capability():
    from trade_rl.evaluation import allocation_cash_control as api

    assert hasattr(api, "run_global_allocation_cash_control"), (
        "global cash runner is missing"
    )
    assert hasattr(api, "GlobalAllocationCashControlResult"), (
        "global cash result is missing"
    )
    return api


@pytest.mark.parametrize("annual_rate", [0, 0.876])
@pytest.mark.parametrize("boundary", [7, 9])
def test_global_cash_uses_one_native_account_and_complete_financial_clock(
    annual_rate,
    boundary,
    monkeypatch,
):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )
    from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest

    api = global_cash_capability()
    arguments = dict(mode="direct", cash_rate=annual_rate)
    uninterrupted = global_control_fixture(**arguments)
    observation, _ = uninterrupted.reset(seed=0)
    states, observations = {}, []
    for _ in range(4):
        observations.append(observation.copy())
        observation, _, terminal, truncated, _ = uninterrupted.step(0)
        states[uninterrupted.index] = allocation_state_digest(uninterrupted)
        assert not truncated and terminal is (uninterrupted.index == 10)
    env = global_control_fixture(**arguments)
    resets, actions, actual_observations = [], [], []
    native_reset, native_step = env.reset, env.step

    def reset(**kwargs):
        resets.append(kwargs)
        result = native_reset(**kwargs)
        actual_observations.append(result[0].copy())
        return result

    def step(action):
        actions.append(action)
        result = native_step(action)
        assert allocation_state_digest(env) == states[env.index]
        if env.index < 10:
            actual_observations.append(result[0].copy())
        return result

    monkeypatch.setattr(env, "reset", reset)
    monkeypatch.setattr(env, "step", step)
    result = api.run_global_allocation_cash_control(global_control_folds(boundary), env)
    assert type(result) is api.GlobalAllocationCashControlResult
    assert actions == [0] * 4 and resets == [{"seed": 0}]
    np.testing.assert_array_equal(actual_observations, observations)
    assert [float(o[22]) for o in actual_observations] == [1, 0.75, 0.5, 0.25]
    expected = 1000 * (1 + annual_rate / 8760) ** 4
    assert env.book.cash == pytest.approx(expected)
    assert env.book.portfolio_value == pytest.approx(expected)
    assert env.book.exact_quantities == (0,)
    assert env.initial_capital == env.observation_schema.initial_capital == 1000
    assert not env.order_book.active_orders
    assert env.book.fill_count == env.book.turnover_total == env.book.total_cost == 0
    assert env.book.funding_pnl == env.book.borrow_cost == 0
    assert result.walk_forward.stitched.diagnostics.n_trades == 0
    assert result.walk_forward.folds[0].closing_state_digest == states[boundary]
    assert result.walk_forward.folds[1].opening_state_digest == states[boundary]
    assert result.walk_forward.policy_digests == (result.control_policy_digest,) * 2
    assert result.runtime_recipe_digest == env.recipe_digest
    if annual_rate == 0:
        assert env.book.cash == 1000


@pytest.mark.parametrize(
    "change", ["residual", "gap", "wrong_env", "used", "missing_cost", "late_cost"]
)
def test_global_cash_preflight_rejects_before_native_reset(change, monkeypatch):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = global_cash_capability()
    env = global_control_fixture(mode="residual" if change == "residual" else "direct")
    folds = global_control_folds(7)
    if change == "gap":
        from trade_rl.evaluation.robustness.walk_forward.folds import IndexRange

        folds = (folds[0], replace(folds[1], test=IndexRange(8, 10)))
    elif change == "used":
        env.reset()
    elif change == "missing_cost":
        env._estimates.pop(next(iter(env._estimates)))
    elif change == "late_cost":
        key = next(iter(env._estimates))
        object.__setattr__(
            env._estimates[key],
            "available_at",
            np.datetime64(key, "ns") + np.timedelta64(1, "h"),
        )
    monkeypatch.setattr(env, "reset", lambda **kw: pytest.fail("native reset happened"))
    with pytest.raises((ValueError, RuntimeError)):
        api.run_global_allocation_cash_control(
            folds, (env,) if change == "wrong_env" else env
        )


def test_global_cash_result_rejects_local_fold_substitutes_and_wrong_identity():
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = global_cash_capability()
    first, second, folds = cash_fixture()
    local = api.run_continuous_allocation_cash_control(folds, (first, second))
    with pytest.raises(ValueError, match="global"):
        api.GlobalAllocationCashControlResult(local, "a" * 64, "b" * 64, "c" * 64)
    env = global_control_fixture(mode="direct")
    result = api.run_global_allocation_cash_control(global_control_folds(7), env)
    with pytest.raises(ValueError, match="cash control policy"):
        replace(
            result,
            walk_forward=replace(result.walk_forward, policy_digests=("b" * 64,) * 2),
        )


def test_global_cash_propagates_native_failure_without_completed_result(monkeypatch):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = global_cash_capability()
    env = global_control_fixture(mode="direct")
    monkeypatch.setattr(
        env, "step", lambda *_: (_ for _ in ()).throw(RuntimeError("native failure"))
    )
    with pytest.raises(RuntimeError, match="native failure"):
        api.run_global_allocation_cash_control(global_control_folds(7), env)


@pytest.mark.parametrize("field", ["folds", "policy_digests"])
def test_global_cash_wrapper_rechecks_immutable_native_roster(field):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = global_cash_capability()
    env = global_control_fixture(mode="direct")
    result = api.run_global_allocation_cash_control(global_control_folds(7), env)
    object.__setattr__(
        result.walk_forward, field, list(getattr(result.walk_forward, field))
    )
    with pytest.raises(ValueError, match="immutable"):
        result.__post_init__()


def test_global_cash_checks_each_native_segment_trade_diagnostic(monkeypatch):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = global_cash_capability()
    native = api.run_global_allocation_walk_forward

    def inconsistent_segment(*args, **kwargs):
        result = native(*args, **kwargs)
        first, second = result.folds
        return replace(
            result,
            folds=(
                replace(first, diagnostics=replace(first.diagnostics, n_trades=1)),
                second,
            ),
        )

    monkeypatch.setattr(api, "run_global_allocation_walk_forward", inconsistent_segment)
    with pytest.raises(ValueError, match="cash control"):
        api.run_global_allocation_cash_control(
            global_control_folds(7), global_control_fixture(mode="direct")
        )
