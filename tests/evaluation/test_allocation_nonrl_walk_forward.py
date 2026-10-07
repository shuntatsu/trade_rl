"""Native fixed-input oracles: no fit, policy loader or market-data access."""

from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module

import numpy as np
import pytest

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    SimpleReturnTrainingSet,
)
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def capability():
    try:
        return import_module("trade_rl.evaluation.allocation_nonrl_walk_forward")
    except ModuleNotFoundError as error:
        if error.name == "trade_rl.evaluation.allocation_nonrl_walk_forward":
            pytest.fail("Fixed nonRL continuous-account consumer is missing")
        raise


def fixture(
    *,
    ranges=((6, 8), (8, 10)),
    signals=(1, -1, 1, -1),
    volume=100000,
    risk=None,
    mode="residual",
    fee=0.002,
    time_unit="ns",
    cost_changes=None,
):
    times = np.datetime64("2026-01-01", "ns") + np.arange(12) * np.timedelta64(1, "h")
    close = np.array([100] * 7 + [110, 110, 100, 100, 100], dtype=float)[:, None]
    opens = close.copy()
    opens[7] = 100
    features = np.zeros((12, 1, 1), dtype=np.float32)
    features[6:10, 0, 0] = signals
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=times.astype(f"datetime64[{time_unit}]"),
        features=features,
        global_features=np.zeros((12, 1), dtype=np.float32),
        open=opens,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        close=close,
        volume=np.full((12, 1), volume, dtype=float),
        volume_units=(VolumeUnit.BASE_ASSET,),
        funding_rate=np.zeros((12, 1)),
        tradable=np.ones((12, 1), dtype=np.bool_),
        feature_available=np.ones((12, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
    )
    trace = ForecastTrainingTrace(
        ("S0",),
        times[:1],
        times[1:2],
        times[:1],
        times[1:2],
        np.array([100.0]),
        np.array([100.0]),
    )
    training = SimpleReturnTrainingSet(
        CausalForecastTrainingSet(
            (0,),
            np.zeros((1, 1)),
            np.zeros(1),
            times[1:2],
            np.ones(1),
            times[5],
            1,
            trace,
            ("signal",),
        )
    )
    # Declared arithmetic coefficients, deliberately never produced by a fit.
    model = SimpleReturnRidgeModel(
        (0,), np.zeros(1), np.ones(1), np.array([0.1]), 0.0, 1, 1.0, 1, times[5], 0.0
    )
    vintage = SimpleReturnVintage(
        ForecastBlock(
            times[5], times[5] + np.timedelta64(15, "m"), times[6], times[10]
        ),
        training,
        model,
        ("S0",),
    )
    stream = FrozenSimpleReturnStream(
        dataset.dataset_id,
        (vintage,),
        tuple(
            SimpleReturnPacket(
                "S0",
                times[i],
                times[i],
                times[i],
                times[i + 1],
                3600,
                float(features[i, 0, 0]) * 0.1,
                0.0,
                vintage.digest,
                (float(features[i, 0, 0]),),
                float(close[i, 0]),
            )
            for i in range(6, 10)
        ),
    )
    costs = replace(
        ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0
    )
    costs = replace(costs, **(cost_changes or {}))
    risk = risk or PreTradeRiskConfig(
        max_gross=1.0, max_abs_weight=1.0, max_turnover=None
    )
    action = AllocationActionContract(mode=mode, scale=0.5)
    allocator = AfterCostTargetAllocator(
        lower_weight=0.0, upper_weight=0.5, risk_aversion=0.0
    )
    execution = MarketExecutor(
        dataset, costs, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    environments, folds = [], []
    for index, (start, stop) in enumerate(ranges):
        schema = AllocationObservationSchema(("signal",), 8, 1000.0, stop - start)
        profile = AllocationRuntimeProfile(
            execution, content_digest(risk), 1000.0, "USD", 3600, (stop - start) * 3600
        )
        recipe = allocation_recipe_payload_v2(
            action,
            ("signal",),
            allocator=allocator,
            expected_horizon_seconds=3600,
            runtime_profile=profile,
            observation_schema=schema,
        )
        objective = ObjectiveContract(
            CapitalContract("independent_symbol", "USD", (1000.0,)),
            datetime(2026, 1, 1, start, tzinfo=UTC),
            datetime(2026, 1, 1, stop, tzinfo=UTC),
            "marked_continuation",
            execution,
            content_digest(risk),
            content_digest(recipe),
        )
        clock = FinancialClockContract(
            3600, 3600, 3600, (stop - start) * 3600, 2, 1.0, 0.95, "equity_delta_v1"
        )
        environments.append(
            AllocationTradingEnv(
                dataset=dataset,
                stream=stream,
                estimates=tuple(
                    HorizonCostEstimates(
                        "S0",
                        times[i],
                        times[i],
                        times[i + 1],
                        "literal",
                        buy_cost=0.002,
                        sell_cost=0.002,
                    )
                    for i in range(start, stop)
                ),
                bound=BoundObjectiveClock(objective, clock),
                action_contract=action,
                allocator=allocator,
                execution_cost=costs,
                risk_config=risk,
                feature_indices=(0,),
                symbol_index=0,
                start_index=start,
                stop_index=stop,
                account_id="nonrl-S0",
                observation_schema=schema,
            )
        )
        folds.append(
            WalkForwardFold(
                index,
                IndexRange(0, 2),
                IndexRange(2, 3),
                IndexRange(3, start),
                IndexRange(start, stop),
                0,
            )
        )
    return tuple(folds), tuple(environments)


def test_direct_carrier_rejected_before_any_reset(monkeypatch):
    api = capability()
    folds, envs = fixture(mode="direct")
    monkeypatch.setattr(
        AllocationTradingEnv, "reset", lambda *a, **k: pytest.fail("reset happened")
    )
    with pytest.raises(ValueError, match="residual"):
        api.run_continuous_nonrl_allocation(folds, envs)


def facts(result, weights):
    book = result.book
    return (
        book.exact_quantities,
        book.cash,
        book.portfolio_value,
        book.peak_value,
        book.max_drawdown,
        book.margin_used,
        book.total_cost,
        book.fill_count,
        book.as_of_index,
        result.order_book,
        result.order_events,
        result.interval_net_return,
        tuple(weights),
    )


@pytest.mark.parametrize("control", ["ordinary", "risk_veto", "partial_hold"])
def test_actual_forecast_executor_parity_and_carried_pending_hold(control, monkeypatch):
    from trade_rl.evaluation.forecast_allocation import (
        execute_forecast_proposal,
        propose_forecast_target,
    )

    arguments = {}
    if control == "risk_veto":
        arguments["risk"] = PreTradeRiskConfig(
            max_gross=1.0, max_abs_weight=1.0, max_turnover=0.0
        )
    elif control == "partial_hold":
        arguments = dict(ranges=((6, 7), (7, 8)), volume=1, signals=(1, 0, 0, 0))
    api = capability()
    folds, envs = fixture(**arguments)
    _, (script,) = fixture(**(arguments | {"ranges": ((6, envs[-1].stop_index),)}))
    script.reset()
    expected = []
    for index in range(script.start_index, script.stop_index):
        causal = dict(
            account_id=script.account_id,
            stream=script.stream,
            estimates=script._estimates[
                int(script.dataset.timestamps[index].astype(np.int64))
            ],
            allocator=script.allocator,
            pretrade_risk=script.risk,
            symbol_index=0,
            start_index=index,
            expected_horizon_seconds=3600,
        )
        proposal = propose_forecast_target(
            script.executor, script.book, script.order_book, **causal
        )
        actual = execute_forecast_proposal(
            script.executor, script.book, script.order_book, proposal, **causal
        )
        script.book, script.order_book = (
            actual.execution.book,
            actual.execution.order_book,
        )
        expected.append(facts(actual.execution, actual.risk_target.weights))
    observed, actions = [], []
    native_step = AllocationTradingEnv.step

    def recorded_step(env, action):
        actions.append(action)
        value = native_step(env, action)
        observed.append(facts(value[-1]["execution"], value[-1]["risk_target"].weights))
        return value

    monkeypatch.setattr(AllocationTradingEnv, "step", recorded_step)
    result = api.run_continuous_nonrl_allocation(folds, envs)
    assert actions == [2] * len(expected)
    assert observed == expected  # Includes exact quantity, order IDs/events and fees.
    assert result.walk_forward.stitched.returns.values == tuple(
        row[11] for row in expected
    )
    assert (
        result.walk_forward.folds[0].closing_state_digest
        == result.walk_forward.folds[1].opening_state_digest
    )
    if control == "ordinary":
        assert observed[0][0] == (5,)
        assert observed[0][1:3] == (499.0, 1049.0)
        assert observed[0][6] == 1.0
        assert envs[0].book.cash == pytest.approx(1047.9)
        assert envs[0].book.total_cost == pytest.approx(2.1)
    elif control == "risk_veto":
        assert envs[-1].book.cash == 1000.0
        assert envs[-1].book.fill_count == 0
    else:
        assert envs[0].book.exact_quantities == (1,)
        assert len(envs[0].order_book.active_orders) == 1
        assert envs[-1].book.exact_quantities == (1,)
        assert not envs[-1].order_book.active_orders
        assert any(event.event_type == "cancelled" for event in observed[1][10])


@pytest.mark.parametrize(
    "mismatch",
    [
        "account",
        "forecast",
        "candidate",
        "gap",
        "used",
        "missing_cost",
        "missing_packet",
        "cost_horizon",
    ],
)
def test_whole_chain_incompatibility_rejected_before_first_reset(mismatch, monkeypatch):
    api = capability()
    folds, envs = fixture()
    if mismatch == "account":
        envs[1].account_id = "other"
    elif mismatch == "forecast":
        envs[1].stream = replace(envs[1].stream, dataset_id="e" * 64)
    elif mismatch == "candidate":
        envs[1].allocator = replace(envs[1].allocator, upper_weight=0.4)
    elif mismatch == "gap":
        folds = (folds[0], replace(folds[1], test=IndexRange(9, 10)))
    elif mismatch == "missing_cost":
        envs[1]._estimates.pop(next(iter(envs[1]._estimates)))
    elif mismatch == "missing_packet":
        stream = replace(envs[0].stream, packets=envs[0].stream.packets[:2])
        envs[0].stream = envs[1].stream = stream
    elif mismatch == "cost_horizon":
        key = next(iter(envs[1]._estimates))
        envs[1]._estimates[key] = replace(
            envs[1]._estimates[key],
            horizon_end=np.datetime64(key, "ns") + np.timedelta64(2, "h"),
        )
    else:
        envs[1].reset()
    monkeypatch.setattr(
        AllocationTradingEnv, "reset", lambda *a, **k: pytest.fail("reset happened")
    )
    with pytest.raises((ValueError, RuntimeError)):
        api.run_continuous_nonrl_allocation(folds, envs)
    assert not hasattr(envs[0], "book")


def comparison_contract(env, folds, *, stress=None):
    from trade_rl.evaluation.allocation_comparison import (
        AllocationComparisonContract,
        AllocationComparisonScenario,
    )
    from trade_rl.evaluation.allocation_comparison_evidence import (
        allocation_business_objective_digest,
        allocation_comparison_scenario_digest,
        allocation_economic_clock_digest,
        allocation_fold_plan_digest,
    )
    from trade_rl.evaluation.allocation_scenario_identity import (
        allocation_candidate_recipe_digest,
    )

    pin = allocation_candidate_recipe_digest(env.recipe)
    scenario_rows = (("base", env),) + ((("fee_stress", stress),) if stress else ())
    scenarios = tuple(
        AllocationComparisonScenario(
            name,
            allocation_comparison_scenario_digest(
                name=name,
                dataset_id=env.dataset.dataset_id,
                forecast_context_digest=env.stream.digest,
                economics_digest=actual.executor.execution_policy_digest,
                risk_digest=content_digest(actual.risk_config),
            ),
            True,
        )
        for name, actual in scenario_rows
    )
    return AllocationComparisonContract(
        env.dataset.dataset_id,
        allocation_business_objective_digest(env.bound.objective),
        allocation_economic_clock_digest(env.bound.clock),
        env.stream.digest,
        env.executor.execution_policy_digest,
        content_digest(env.risk_config),
        allocation_fold_plan_digest(folds),
        pin,
        pin,
        pin,
        "independent_symbol",
        1000.0,
        scenarios,
        (0, 1),
        0.2,
    )


def test_seedless_evidence_reuses_common_metrics_and_keeps_full_stress_runtime():
    from trade_rl.evaluation.allocation_comparison import AllocationCandidateKind
    from trade_rl.evaluation.allocation_comparison_evidence import (
        build_continuous_allocation_comparison_evidence,
    )

    api = capability()
    folds, base = fixture()
    _, stress = fixture(fee=0.004)
    contract = comparison_contract(base[0], folds, stress=stress[0])
    baseline = api.run_continuous_nonrl_allocation(folds, base)
    stressed = api.run_continuous_nonrl_allocation(folds, stress)
    assert baseline.candidate_recipe_digest == stressed.candidate_recipe_digest
    assert baseline.rule_digest == stressed.rule_digest
    assert baseline.rule_digest == content_digest(
        {
            "schema": "allocation_nonrl_rule_v1",
            "candidate_recipe_digest": baseline.candidate_recipe_digest,
            "action_semantics": "residual_exact_baseline_v1",
            "raw_action": 2,
        }
    )
    assert baseline.walk_forward.policy_digests == (baseline.rule_digest,) * 2
    assert baseline.runtime_recipe_digests != stressed.runtime_recipe_digests
    evidence = api.build_continuous_nonrl_allocation_evidence(
        contract,
        scenario="base",
        folds=folds,
        environments=base,
        result=baseline,
        validity_evidence_digest="a" * 64,
    )
    common = build_continuous_allocation_comparison_evidence(
        contract,
        candidate=AllocationCandidateKind.NONRL,
        scenario="base",
        seed=None,
        folds=folds,
        environments=base,
        result=baseline.walk_forward,
        validity_evidence_digest="a" * 64,
    )
    assert evidence == common
    assert evidence.seed is evidence.policy_digest is None
    stress_evidence = api.build_continuous_nonrl_allocation_evidence(
        contract,
        scenario="fee_stress",
        folds=folds,
        environments=stress,
        result=stressed,
        validity_evidence_digest="a" * 64,
    )
    assert evidence.execution_digest != stress_evidence.execution_digest
    with pytest.raises(ValueError, match="context"):
        api.build_continuous_nonrl_allocation_evidence(
            contract,
            scenario="fee_stress",
            folds=folds,
            environments=stress,
            result=baseline,
            validity_evidence_digest="a" * 64,
        )


def test_generic_or_wrong_action_result_cannot_be_accidentally_labelled_nonrl():
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
        run_continuous_allocation_walk_forward,
    )

    api = capability()
    folds, envs = fixture()
    contract = comparison_contract(envs[0], folds)
    generic = run_continuous_allocation_walk_forward(
        folds,
        envs,
        tuple(
            AllocationFoldPolicy("b" * 64, env.recipe_digest, lambda _o, _r: 0)
            for env in envs
        ),
    )
    with pytest.raises(ValueError, match="fixed allocator result"):
        api.build_continuous_nonrl_allocation_evidence(
            contract,
            scenario="base",
            folds=folds,
            environments=envs,
            result=generic,
            validity_evidence_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="fixed allocator rule"):
        api.ContinuousNonRLAllocationResult(
            generic,
            contract.nonrl_recipe_digest,
            envs[0].stream.digest,
            tuple(env.recipe_digest for env in envs),
        )


def test_execution_failure_propagates_without_completed_nonrl_result(monkeypatch):
    api = capability()
    folds, envs = fixture()
    monkeypatch.setattr(
        AllocationTradingEnv,
        "step",
        lambda *a: (_ for _ in ()).throw(RuntimeError("native failure")),
    )
    with pytest.raises(RuntimeError, match="native failure"):
        api.run_continuous_nonrl_allocation(folds, envs)


@pytest.mark.parametrize("unit", ["h", "us"])
def test_admitted_native_timestamp_units_preserve_nanosecond_cost_keys(unit):
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
        run_continuous_allocation_walk_forward,
    )

    folds, native = fixture(time_unit=unit)
    expected = run_continuous_allocation_walk_forward(
        folds,
        native,
        tuple(
            AllocationFoldPolicy("b" * 64, env.recipe_digest, lambda _o, _r: 2)
            for env in native
        ),
    )
    _, actual = fixture(time_unit=unit)
    result = capability().run_continuous_nonrl_allocation(folds, actual)
    assert result.walk_forward.stitched == expected.stitched


@pytest.mark.parametrize(
    "arguments",
    [
        {"cost_changes": {"order_latency_bars": 1}},
        {"cost_changes": {"order_type": "limit"}},
        {"risk": PreTradeRiskConfig(drawdown_start=0.2, drawdown_stop=0.2)},
    ],
)
def test_unsupported_native_allocation_profile_rejected_before_reset(
    arguments, monkeypatch
):
    folds, envs = fixture(**arguments)
    monkeypatch.setattr(
        AllocationTradingEnv, "reset", lambda *a, **k: pytest.fail("reset happened")
    )
    with pytest.raises(ValueError, match="MARKET|drawdown"):
        capability().run_continuous_nonrl_allocation(folds, envs)


def global_control_fixture(
    *,
    mode="residual",
    cash_rate=None,
    insolvent_short=False,
    zero_later_volume=False,
    **arguments,
):
    """Literal H4/C1000 native carrier; coefficients are declared, never fitted."""
    _, (original,) = fixture(ranges=((6, 10),), mode=mode, **arguments)
    dataset, allocator = original.dataset, original.allocator
    if cash_rate is not None:
        dataset = replace(dataset, cash_rate=np.full(12, cash_rate, dtype=np.float64))
    if zero_later_volume:
        volume = dataset.volume.copy()
        volume[8:] = 0
        dataset = replace(dataset, volume=volume)
    if insolvent_short:
        prices = dataset.close.copy()
        prices[7:] = 400
        opens = prices.copy()
        opens[7] = 100
        dataset = replace(
            dataset,
            close=prices,
            open=opens,
            high=np.maximum(opens, prices),
            low=np.minimum(opens, prices),
            mark_price=prices,
        )
        allocator = replace(allocator, lower_weight=-0.5, upper_weight=0)
    execution = MarketExecutor(
        dataset, original.execution_cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    profile = AllocationRuntimeProfile(
        execution, content_digest(original.risk_config), 1000.0, "USD", 3600, 14400
    )
    recipe = allocation_recipe_payload_v2(
        original.action_contract,
        ("signal",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        runtime_profile=profile,
        observation_schema=original.observation_schema,
    )
    bound = replace(
        original.bound,
        objective=replace(
            original.bound.objective,
            economics_digest=execution,
            deployment_recipe_digest=content_digest(recipe),
        ),
    )
    return AllocationTradingEnv(
        dataset,
        stream=replace(
            original.stream,
            packets=tuple(
                replace(packet, decision_close=float(dataset.close[index, 0]))
                for index, packet in enumerate(original.stream.packets, start=6)
            ),
        ),
        estimates=tuple(original._estimates.values()),
        bound=bound,
        action_contract=original.action_contract,
        allocator=allocator,
        execution_cost=original.execution_cost,
        risk_config=original.risk_config,
        feature_indices=(0,),
        symbol_index=0,
        start_index=6,
        stop_index=10,
        account_id=original.account_id,
        observation_schema=original.observation_schema,
    )


def global_control_folds(boundary):
    return tuple(
        WalkForwardFold(
            i,
            IndexRange(0, 2),
            IndexRange(2, 3),
            IndexRange(3, 6),
            IndexRange(start, stop),
            0,
        )
        for i, (start, stop) in enumerate(((6, boundary), (boundary, 10)))
    )


def global_nonrl_capability():
    api = capability()
    assert hasattr(api, "run_global_nonrl_allocation"), "global NONRL runner is missing"
    assert hasattr(api, "GlobalNonRLAllocationResult"), "global NONRL result is missing"
    return api


@pytest.mark.parametrize("boundary", [7, 9])
@pytest.mark.parametrize("control", ["ordinary", "risk_veto", "partial_no_fill"])
def test_global_nonrl_matches_native_forecast_and_unsegmented_state(
    boundary,
    control,
    monkeypatch,
):
    from trade_rl.evaluation.forecast_allocation import (
        execute_forecast_proposal,
        propose_forecast_target,
    )
    from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest

    api = global_nonrl_capability()
    arguments = {}
    if control == "risk_veto":
        arguments["risk"] = PreTradeRiskConfig(
            max_gross=1,
            max_abs_weight=1,
            max_turnover=0,
        )
    elif control == "partial_no_fill":
        arguments = dict(
            signals=(1, 1, 1, 1),
            zero_later_volume=True,
            cost_changes=dict(
                max_participation_rate=0.00001, random_seed=73, slippage_std=0.01
            ),
        )
    script = global_control_fixture(**arguments)
    script.reset()
    expected = []
    for index in range(6, 10):
        causal = dict(
            account_id=script.account_id,
            stream=script.stream,
            estimates=script._estimates[
                int(script.dataset.timestamps[index].astype(np.int64))
            ],
            allocator=script.allocator,
            pretrade_risk=script.risk,
            symbol_index=0,
            start_index=index,
            expected_horizon_seconds=3600,
        )
        proposal = propose_forecast_target(
            script.executor, script.book, script.order_book, **causal
        )
        actual = execute_forecast_proposal(
            script.executor,
            script.book,
            script.order_book,
            proposal,
            **causal,
        )
        script.book, script.order_book = (
            actual.execution.book,
            actual.execution.order_book,
        )
        expected.append(facts(actual.execution, actual.risk_target.weights))
    uninterrupted = global_control_fixture(**arguments)
    uninterrupted.reset()
    state = {}
    for _ in range(4):
        uninterrupted.step(2)
        state[uninterrupted.index] = allocation_state_digest(uninterrupted)
    env = global_control_fixture(**arguments)
    observations, observed, actions, resets = [], [], [], []
    native_reset, native_step = env.reset, env.step

    def reset(**kwargs):
        resets.append(kwargs)
        result = native_reset(**kwargs)
        observations.append(result[0].copy())
        return result

    def step(action):
        actions.append(action)
        result = native_step(action)
        observed.append(
            facts(result[-1]["execution"], result[-1]["risk_target"].weights)
        )
        assert allocation_state_digest(env) == state[env.index]
        if env.index < 10:
            observations.append(result[0].copy())
        return result

    monkeypatch.setattr(env, "reset", reset)
    monkeypatch.setattr(env, "step", step)
    result = api.run_global_nonrl_allocation(global_control_folds(boundary), env)
    assert type(result) is api.GlobalNonRLAllocationResult
    assert actions == [2] * 4
    assert resets == [{"seed": None}]
    assert observed == expected
    assert [float(o[22]) for o in observations] == [1, 0.75, 0.5, 0.25]
    assert env.initial_capital == env.observation_schema.initial_capital == 1000
    assert result.walk_forward.stitched.boundaries == ((6, boundary), (boundary, 10))
    assert result.walk_forward.folds[0].closing_state_digest == state[boundary]
    assert result.walk_forward.folds[1].opening_state_digest == state[boundary]
    assert result.walk_forward.stitched.returns.values == tuple(
        row[11] for row in expected
    )
    assert result.walk_forward.policy_digests == (result.rule_digest,) * 2
    assert result.runtime_recipe_digest == env.recipe_digest
    if control == "ordinary":
        assert [row[12] for row in observed] == [(0.5,), (0.0,), (0.5,), (0.0,)]
        assert observed[0][0] == (5,)
        assert observed[0][1:3] == (499, 1049)
        assert observed[0][6] == 1
        # Independent declared arithmetic: later sizing uses the current account
        # at decision close110; the third execution uses the next open100.
        cash2 = 499 + 5 * 110 * (1 - 0.002)
        quantity3 = 0.5 * cash2 / 110
        cash3 = cash2 - quantity3 * 100 * (1 + 0.002)
        cash4 = cash3 + quantity3 * 100 * (1 - 0.002)
        assert [row[0][0] for row in observed] == pytest.approx([5, 0, quantity3, 0])
        assert [row[1] for row in observed] == pytest.approx([499, cash2, cash3, cash4])
        assert [row[2] for row in observed] == pytest.approx(
            [1049, cash2, cash3 + quantity3 * 100, cash4]
        )
        assert env.book.total_cost == pytest.approx(
            1 + 1.1 + 2 * quantity3 * 100 * 0.002
        )
        assert env.book.peak_value == 1049
        assert env.book.max_drawdown == pytest.approx((1049 - cash4) / 1049)
    elif control == "risk_veto":
        assert env.book.fill_count == 0 and env.book.cash == 1000
    else:
        assert env.order_book.active_orders
        assert any(e.event_type == "no_fill" for row in observed[1:] for e in row[10])


@pytest.mark.parametrize(
    "change",
    ["direct", "gap", "bool_clock", "missing_cost", "late_cost", "latency", "drawdown"],
)
def test_global_nonrl_preflight_rejects_before_native_reset(change, monkeypatch):
    api = global_nonrl_capability()
    arguments = {}
    if change == "direct":
        arguments["mode"] = "direct"
    elif change == "latency":
        arguments["cost_changes"] = dict(order_latency_bars=1)
    elif change == "drawdown":
        arguments["risk"] = PreTradeRiskConfig(drawdown_start=0.2, drawdown_stop=0.2)
    env = global_control_fixture(**arguments)
    folds = global_control_folds(7)
    if change == "gap":
        folds = (folds[0], replace(folds[1], test=IndexRange(8, 10)))
    elif change == "bool_clock":
        object.__setattr__(folds[1].test, "stop", True)
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
    with pytest.raises(ValueError):
        api.run_global_nonrl_allocation(folds, env)
    assert not hasattr(env, "book")


def test_global_nonrl_economic_stop_preserves_actual_native_account_prefix(monkeypatch):
    api = global_nonrl_capability()
    env = global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True)
    actions, executions = [], []
    native_step = env.step

    def step(action):
        actions.append(action)
        result = native_step(action)
        executions.append(result[-1]["execution"])
        return result

    monkeypatch.setattr(env, "step", step)
    with pytest.raises(ValueError, match="economic termination"):
        api.run_global_nonrl_allocation(global_control_folds(7), env)
    assert actions == [2] and len(executions) == 1
    assert env.index == 7 < env.stop_index == 10
    assert env.book.portfolio_value == -501
    assert env.book.cash == -501
    assert env.book.termination_reason is not None
    assert env._terminated
    assert executions[0].book is env.book


def test_global_nonrl_result_rejects_local_fold_substitutes_and_wrong_rule():
    api = global_nonrl_capability()
    folds, envs = fixture()
    local = api.run_continuous_nonrl_allocation(folds, envs)
    with pytest.raises(ValueError, match="global"):
        api.GlobalNonRLAllocationResult(
            local.walk_forward,
            local.candidate_recipe_digest,
            local.forecast_context_digest,
            envs[0].recipe_digest,
        )
    env = global_control_fixture()
    result = api.run_global_nonrl_allocation(global_control_folds(7), env)
    with pytest.raises(ValueError, match="fixed allocator rule"):
        replace(
            result,
            walk_forward=replace(result.walk_forward, policy_digests=("b" * 64,) * 2),
        )
    with pytest.raises(ValueError, match="fixed allocator result"):
        api.build_continuous_nonrl_allocation_evidence(
            comparison_contract(envs[0], folds),
            scenario="base",
            folds=folds,
            environments=envs,
            result=result,
            validity_evidence_digest="a" * 64,
        )


@pytest.mark.parametrize("field", ["folds", "policy_digests"])
def test_global_nonrl_wrapper_rechecks_immutable_native_roster(field):
    api = global_nonrl_capability()
    env = global_control_fixture()
    result = api.run_global_nonrl_allocation(global_control_folds(7), env)
    object.__setattr__(
        result.walk_forward, field, list(getattr(result.walk_forward, field))
    )
    with pytest.raises(ValueError, match="immutable"):
        result.__post_init__()
