"""Learner-free receipt grammar and independent native arithmetic oracles."""

from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import CapitalContract, ObjectiveContract
from trade_rl.evaluation.rl_allocation.settled_binding import (
    BoundSettledAllocationClock,
)
from trade_rl.risk.pretrade import PreTradeRiskConfig
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
from trade_rl.strategies.rl.allocation_recipe_v4 import allocation_recipe_payload_v4
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)

_ROSTER = (
    ("cash_carry", 1, 2, "direct", (0,)),
    ("nonrl_residual_baseline", 2, 2, "residual", (2, 2)),
    ("direct_long", 1, 1, "direct", (3,)),
    ("direct_short", 1, 1, "direct", (1,)),
    ("partial_then_no_market", 1, 2, "direct", (3,)),
    ("reserve_drawdown_recovery", 1, 2, "direct", (3,)),
    ("signed_debt_end", 1, 2, "direct", (1,)),
    ("actor_post_return_capture_failure", 1, 1, "direct", (3,)),
    ("reserve_post_return_capture_failure", 1, 2, "direct", (3,)),
)


def _declarations(case):
    """Literal declared coefficients and sources; no solver or native call."""
    name, actor_steps, reserve_steps, mode, _ = case
    horizon = actor_steps + reserve_steps
    times = np.datetime64("2026-01-01", "ns") + np.arange(12) * np.timedelta64(1, "h")
    close = np.asarray(
        [100, 50, 50, 75, 75, 100, 100, 100, 100, 100, 100, 100], dtype=np.float64
    )[:, None]
    opens, volume, cash_rate = close.copy(), np.full((12, 1), 100000.0), np.zeros(12)
    if name == "cash_carry":
        cash_rate[:] = 8.76
    if name == "partial_then_no_market":
        volume[7:9, 0] = (1.0, 0.0)
    if name == "reserve_drawdown_recovery":
        close[8, 0], volume[8, 0] = 50.0, 0.0
    if name == "signed_debt_end":
        close[7, 0] = 400.0
    available = np.zeros((12, 1, 1), dtype=bool)
    available[: 6 + actor_steps] = True
    zeros, truth = np.zeros((12, 1)), np.ones((12, 1), dtype=bool)
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=("S0",),
        timestamps=times,
        features=available.astype(np.float32),
        feature_names=("constant",),
        feature_available=available,
        feature_staleness=np.where(available, 0, 1).astype(np.float32),
        global_features=np.zeros((12, 1), dtype=np.float32),
        global_feature_names=("regime",),
        periods_per_year=8760,
        open=opens,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        close=close,
        mark_price=close,
        index_price=close,
        volume=volume,
        volume_units=(VolumeUnit.BASE_ASSET,),
        funding_rate=zeros,
        funding_price_rate=zeros,
        funding_due=np.zeros((12, 1), dtype=bool),
        funding_event_count=np.zeros((12, 1), dtype=np.int64),
        tradable=truth,
        symbol_active=truth,
        information_available=truth,
        borrow_available=truth,
        buy_allowed=truth,
        sell_allowed=truth,
        fee_rate=zeros,
        maker_fee_rate=zeros,
        taker_fee_rate=zeros,
        spread_rate=zeros,
        max_participation_rate=np.ones((12, 1)),
        minimum_notional=zeros,
        lot_size=zeros,
        tick_size=zeros,
        borrow_rate=zeros,
        dividend=zeros,
        split_factor=np.ones((12, 1)),
        delisting_recovery=np.ones((12, 1)),
        cash_rate=cash_rate,
        contract_multipliers=np.ones(1),
        available_at=times[:, None],
    ).with_content_identity({"fixture": "settled-native-r2", "case": name})
    trace = ForecastTrainingTrace(
        row_symbols=("S0",) * 4,
        start_times=times[:4],
        end_times=times[1:5],
        start_available_at=times[:4],
        end_available_at=times[1:5],
        start_close=np.asarray([100, 50, 50, 75], dtype=float),
        end_close=np.asarray([50, 50, 75, 75], dtype=float),
    )
    training = SimpleReturnTrainingSet(
        CausalForecastTrainingSet(
            feature_indices=(0,),
            features=np.ones((4, 1)),
            labels=np.asarray(
                [
                    math.log(b / a)
                    for a, b in zip(trace.start_close, trace.end_close, strict=True)
                ]
            ),
            label_end_times=times[1:5],
            sample_weights=np.ones(4),
            fit_cutoff=times[5],
            horizon_hours=1,
            trace=trace,
            feature_names=("constant",),
        )
    )
    model = SimpleReturnRidgeModel(
        feature_indices=(0,),
        feature_mean=np.ones(1),
        feature_scale=np.ones(1),
        coefficients=np.zeros(1),
        intercept=1 / 16,
        horizon_hours=1,
        alpha=1.0,
        n_samples=4,
        fit_cutoff=times[5],
        fit_prefix_marginal_variance=1 / 8,
    )
    vintage = SimpleReturnVintage(
        ForecastBlock(
            times[5],
            times[5] + np.timedelta64(15, "m"),
            times[6],
            times[6 + actor_steps],
        ),
        training,
        model,
        ("S0",),
    )
    packets = tuple(
        SimpleReturnPacket(
            symbol="S0",
            as_of=times[i],
            source_available_at=times[i],
            forecast_available_at=times[i],
            horizon_end=times[i + 1],
            horizon_seconds=3600,
            expected_simple_return=1 / 16,
            fit_prefix_marginal_variance=1 / 8,
            vintage_digest=vintage.digest,
            feature_values=(1.0,),
            decision_close=100.0,
        )
        for i in range(6, 6 + actor_steps)
    )
    stream = FrozenSimpleReturnStream(dataset.dataset_id, (vintage,), packets)
    cost = replace(
        ExecutionCostConfig.zero(), fee_rate=1 / 128, max_participation_rate=1.0
    )
    risk = PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    action, allocator = (
        AllocationActionContract(mode, 0.5),
        AfterCostTargetAllocator(
            lower_weight=-0.5, upper_weight=0.5, risk_aversion=1.0, max_turnover=None
        ),
    )
    clock = SettledAllocationClockContract(
        actor_steps=actor_steps,
        reserve_steps=reserve_steps,
        execution_interval_seconds=3600,
        rollout_steps=2,
        gamma=1.0,
        gae_lambda=0.95,
    )
    schema = AllocationObservationSchema(("constant",), 8, 1000.0, horizon)
    economics = content_digest(
        {
            "schema_version": "insolvency_execution_policy_v1",
            "base_policy_digest": cost.execution_policy_digest,
            "insolvency_valuation": "retain_debt",
        }
    )
    profile = AllocationRuntimeProfile(
        economics, content_digest(risk), 1000.0, "USD", 3600, horizon * 3600
    )
    recipe = allocation_recipe_payload_v4(
        action,
        ("constant",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        runtime_profile=profile,
        settled_clock=clock,
        observation_schema=schema,
    )
    start = datetime(2026, 1, 1, 6, tzinfo=UTC)
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        start,
        start + timedelta(hours=horizon),
        "settled",
        economics,
        content_digest(risk),
        content_digest(recipe),
        maximum_drawdown=0.2,
    )
    estimates = tuple(
        HorizonCostEstimates(
            "S0",
            times[i],
            times[i],
            times[i + 1],
            "settled-native-r2-fee-1-over-128",
            buy_cost=1 / 128,
            sell_cost=1 / 128,
            exit_cost=1 / 128,
        )
        for i in range(6, 6 + actor_steps)
    )
    return dict(
        dataset=dataset,
        stream=stream,
        estimates=estimates,
        bound=BoundSettledAllocationClock(objective, clock),
        action_contract=action,
        allocator=allocator,
        execution_cost=cost,
        risk_config=risk,
        feature_indices=(0,),
        symbol_index=0,
        start_index=6,
        account_id="native-r2-" + name,
        observation_schema=schema,
        expected_horizon_seconds=3600,
    )


def test_missing_opening_stays_diagnostic_and_cannot_be_admitted():
    owner = _owner()
    context = owner.SettledNativeValidationContext(**_declarations(_ROSTER[0]))
    raw = _absent_capture()
    raw["context_digest"] = context.digest
    capture = owner.SettledNativeCapture.from_bytes(canonical_json_bytes(raw))
    result = owner.validate_settled_native_capture(context, capture, actor_intents=())
    assert not result.admitted
    assert result.payload()["counts"]["validated_native_rows"] == 0
    with pytest.raises(ValueError, match="admission"):
        owner.require_settled_native_admission(context, capture, actor_intents=())


def test_context_rejects_an_objective_for_another_economics():
    declared = _declarations(_ROSTER[0])
    bound = declared["bound"]
    declared["bound"] = replace(
        bound, objective=replace(bound.objective, economics_digest="c" * 64)
    )
    with pytest.raises(ValueError, match="objective"):
        _owner().SettledNativeValidationContext(**declared)


@pytest.mark.parametrize("field", ["risk_digest", "deployment_recipe_digest"])
def test_context_rejects_objective_hashes_for_other_declared_sources(field):
    declared = _declarations(_ROSTER[0])
    bound = declared["bound"]
    declared["bound"] = replace(
        bound, objective=replace(bound.objective, **{field: "c" * 64})
    )
    with pytest.raises(ValueError, match="objective"):
        _owner().SettledNativeValidationContext(**declared)


def test_context_rejects_objective_time_window_shifted_from_dataset():
    declared = _declarations(_ROSTER[0])
    bound = declared["bound"]
    declared["bound"] = replace(
        bound,
        objective=replace(
            bound.objective,
            evaluation_start=bound.objective.evaluation_start + timedelta(hours=1),
            evaluation_stop_exclusive=bound.objective.evaluation_stop_exclusive
            + timedelta(hours=1),
        ),
    )
    with pytest.raises(ValueError, match="endpoint"):
        _owner().SettledNativeValidationContext(**declared)


@pytest.mark.parametrize("change", ["volume_unit", "buy_denied", "borrow_denied"])
def test_other_volume_units_and_denied_permissions_are_diagnostic_only(change):
    declared = _declarations(_ROSTER[0])
    dataset = declared["dataset"]
    if change == "volume_unit":
        dataset = replace(
            dataset, identity_payload_json=None, volume_units=(VolumeUnit.CONTRACTS,)
        ).with_content_identity({"fixture": "pure-unsupported-volume-unit"})
    else:
        permission = np.zeros((12, 1), dtype=bool)
        dataset = replace(
            dataset,
            identity_payload_json=None,
            **{
                "buy_allowed"
                if change == "buy_denied"
                else "borrow_available": permission
            },
        ).with_content_identity(
            {"fixture": "pure-unsupported-permission", "case": change}
        )
    declared["dataset"] = dataset
    declared["stream"] = replace(declared["stream"], dataset_id=dataset.dataset_id)
    context = _owner().SettledNativeValidationContext(**declared)
    assert not _owner()._normal_profile(context)


def _owner():
    name = "trade_rl.evaluation.rl_allocation.settled_native_validation"
    assert importlib.util.find_spec(name) is not None, (
        "native-only receipt owner missing"
    )
    return importlib.import_module(name)


@pytest.mark.parametrize("action", [True, 1.0, -1, 4, "0"])
def test_actor_intent_requires_actual_integer_action(action):
    with pytest.raises(ValueError, match="actor intent"):
        _owner().SettledNativeActorIntent(ordinal=0, start_index=6, action=action)


@pytest.mark.parametrize("field", ["ordinal", "start_index"])
def test_actor_intent_clock_never_accepts_bool(field):
    arguments = dict(ordinal=0, start_index=6, action=0)
    arguments[field] = True
    with pytest.raises(ValueError, match="actor intent"):
        _owner().SettledNativeActorIntent(**arguments)


def _absent_capture():
    error = {"type_name": "EncodingFailure", "message": None}
    return {
        "schema": "allocation_settled_native_capture_v1",
        "context_digest": "a" * 64,
        "opening": None,
        "actor_calls": [],
        "returned_frontier": {
            "index": None,
            "time_ns": None,
            "native_account": None,
            "state_digest": None,
            "encoding_error": error,
        },
        "failure": {
            "origin": "capture",
            "actor_call_ordinal": None,
            "stage": "capture_encoding",
            "attempted_start_index": None,
            "error": error,
        },
    }


def test_capture_detaches_payload_and_preserves_exact_canonical_bytes():
    original = canonical_json_bytes(_absent_capture())
    capture = _owner().SettledNativeCapture.from_bytes(original)
    altered = capture.payload()
    altered["context_digest"] = "b" * 64
    assert capture.payload()["context_digest"] == "a" * 64
    assert capture.to_bytes() == original


@pytest.mark.parametrize(
    "change", ["extra", "no_failure", "partial_frontier", "bool_clock"]
)
def test_capture_rejects_fabricated_absence_or_undeclared_fields(change):
    value = _absent_capture()
    if change == "extra":
        value["admitted"] = True
    elif change == "no_failure":
        value["failure"] = None
    elif change == "partial_frontier":
        value["returned_frontier"]["index"] = 7
    else:
        value["failure"]["attempted_start_index"] = True
    with pytest.raises(ValueError):
        _owner().SettledNativeCapture.from_bytes(canonical_json_bytes(value))


def test_noncanonical_capture_does_not_acquire_an_identity():
    import json

    raw = json.dumps(_absent_capture(), indent=2).encode()
    with pytest.raises(ValueError, match="canonical"):
        _owner().SettledNativeCapture.from_bytes(raw)


@pytest.mark.parametrize(
    ("opened", "marked", "peak", "drawdown"),
    [
        (1099.21875, 999.21875, 1099.21875, 128 / 1407),
        (949.21875, 999.21875, 1000.0, 13 / 256),
    ],
)
def test_recovered_open_extrema_remain_in_drawdown(opened, marked, peak, drawdown):
    actual_peak, actual_dd = _owner()._normal_drawdown_sites(
        1000.0,
        0.0,
        opened,
        None,
        marked,
    )
    assert actual_peak == peak
    assert actual_dd == pytest.approx(drawdown, rel=0, abs=2e-16)


def test_fill_site_is_observed_before_recovered_final_equity():
    peak, drawdown = _owner()._normal_drawdown_sites(1000.0, 0.0, 1000.0, 750.0, 1000.0)
    assert peak == 1000.0
    assert drawdown == 0.25


def test_counters_use_actual_book_threshold_and_prior_equity():
    from fractions import Fraction

    before = {
        "equity": 1000.0,
        "fill_count": 2,
        "rebalance_events": 2,
        "turnover_total": 0.25,
    }
    after = {"fill_count": 3, "rebalance_events": 3, "turnover_total": 0.375}
    _owner()._counter_turnover_join(before, after, 125.0, Fraction(5, 4))
    after["turnover_total"] = 125.0 / 900.0 + 0.25
    with pytest.raises(ValueError, match="turnover"):
        _owner()._counter_turnover_join(before, after, 125.0, Fraction(5, 4))


def test_native_event_does_not_imply_thresholded_book_fill():
    from fractions import Fraction

    before = {
        "equity": 1000.0,
        "fill_count": 2,
        "rebalance_events": 2,
        "turnover_total": 0.25,
    }
    after = {"fill_count": 2, "rebalance_events": 2, "turnover_total": 0.25 + 1e-13}
    _owner()._counter_turnover_join(before, after, 1e-10, Fraction.from_float(1e-12))
    after["fill_count"] = 3
    with pytest.raises(ValueError, match="counter"):
        _owner()._counter_turnover_join(
            before, after, 1e-10, Fraction.from_float(1e-12)
        )


def _literal_order():
    from trade_rl.simulation.orders.model import (
        OrderIntent,
        OrderType,
        PendingOrder,
        TimeInForce,
    )

    return PendingOrder.from_intent(
        OrderIntent.create(
            dataset_id="a" * 64,
            target_identity="b" * 64,
            execution_policy_digest="c" * 64,
            symbol_index=0,
            requested_quantity=5.0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            limit_price=None,
            stop_price=None,
            submit_index=6,
            eligible_index=6,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1000.0,
        )
    )


def test_partial_pending_progress_has_one_eligibility_and_exact_cumulative_fill():
    result, transitions = _owner()._normal_pending_progress(
        _literal_order(), processing_index=7, filled_quantity=1.0, filled_notional=100.0
    )
    assert result.status.value == "partially_filled"
    assert result.remaining_quantity == 4.0
    assert result.exact_cumulative_filled_quantity == "1"
    assert result.cumulative_filled_notional == 100.0
    assert result.evidence_version == 2
    assert [t[2] for t in transitions] == ["eligible", "partial_fill"]
    unchanged, transitions = _owner()._normal_pending_progress(
        result, processing_index=8, filled_quantity=0.0, filled_notional=0.0
    )
    assert unchanged == result
    assert [t[2] for t in transitions] == ["no_fill"]


def test_full_pending_progress_cannot_be_a_partial_or_extra_eligible_event():
    result, transitions = _owner()._normal_pending_progress(
        _literal_order(), processing_index=7, filled_quantity=5.0, filled_notional=500.0
    )
    assert result.status.value == "filled"
    assert result.remaining_quantity == 0.0
    assert result.terminal_reason == "filled"
    assert result.evidence_version == 2
    assert [t[2] for t in transitions] == ["eligible", "filled"]


def test_source_denied_leverage_is_unsupported_diagnostic_not_malformed():
    from trade_rl.simulation.orders.model import OrderIntent, PendingOrder

    owner = _owner()
    declared = _declarations(_ROSTER[2])
    dataset = declared["dataset"]
    opens, high = dataset.open.copy(), dataset.high.copy()
    opens[7, 0] = high[7, 0] = 201.0
    dataset = replace(
        dataset, identity_payload_json=None, open=opens, high=high
    ).with_content_identity({"fixture": "pure-source-denied-leverage"})
    declared["dataset"], declared["stream"] = (
        dataset,
        replace(declared["stream"], dataset_id=dataset.dataset_id),
    )
    context = owner.SettledNativeValidationContext(**declared)
    values = _literal_order().intent.canonical_payload()
    values.pop("order_id")
    values.update(
        dataset_id=dataset.dataset_id, execution_policy_digest=context.economics_digest
    )
    order = PendingOrder.from_intent(OrderIntent.create(**values))
    before = {
        "book": {
            "quantities": [0.0],
            "cash": 1000.0,
            "equity": 1000.0,
            "insolvent": False,
            "exact_quantities": ["0"],
        }
    }
    row = {
        "processing_index": 7,
        "native": {
            "order_events": [],
            "execution": {
                "requested_notional": 500.0,
                "requested_notional_by_symbol": [500.0],
                "rejected_count": 1,
                "expired_count": 0,
            },
        },
    }
    with pytest.raises(owner._UnsupportedNormalProfile):
        owner._capacity_cost_join(context, row, before, (order,))


def test_failed_witness_counts_only_prefix_and_retains_adopted_tail():
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_account_facts,
    )
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.orders.model import OrderBookState

    owner = _owner()
    context = owner.SettledNativeValidationContext(**_declarations(_ROSTER[7]))
    account = native_allocation_account_facts(
        BookState(
            quantities=np.asarray([5.0]),
            cash=496.09375,
            mark_prices=np.asarray([100.0]),
            peak_value=1000.0,
            max_drawdown=1 / 256,
            total_cost=3.90625,
            fill_count=1,
            rebalance_events=1,
            turnover_total=0.5,
            as_of_index=7,
            as_of_dataset_id=context.dataset.dataset_id,
        ),
        OrderBookState(active_orders=(), terminal_orders=()),
    )
    frontier = dict(
        index=7,
        time_ns=int(context.dataset.timestamps[7].astype(np.int64)),
        native_account=account,
        state_digest=owner.native._state(context, 1, 7, account),
        encoding_error=None,
    )
    witness = dict(
        actor_steps_processed=0,
        physical_steps_processed=0,
        reserve_steps_processed=0,
        full_coverage=False,
        settled=False,
        eligible=False,
        maximum_drawdown=1 / 256,
        frontier_index=7,
        native_frontier=account,
    )
    owner._failed_witness(context, witness, frontier, [], episode=1)
    for changed in (
        witness | {"actor_steps_processed": 1},
        witness | {"eligible": True},
        witness | {"frontier_index": 6},
    ):
        with pytest.raises(ValueError):
            owner._failed_witness(context, changed, frontier, [], episode=1)


class _PostReturnFailure(BaseException):
    pass


def _capture_frontier(env):
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_account_facts,
    )

    try:
        account = native_allocation_account_facts(env.book, env.order_book)
        return dict(
            index=env.index,
            time_ns=int(env.dataset.timestamps[env.index].astype(np.int64)),
            native_account=account,
            state_digest=env._state_digest(env.index, env.book, env.order_book),
            encoding_error=None,
        )
    except BaseException as error:
        return dict(
            index=None,
            time_ns=None,
            native_account=None,
            state_digest=None,
            encoding_error={"type_name": type(error).__name__, "message": None},
        )


def _collect_case(case):
    """One fresh account, one reset; never continue or flatten after failure."""
    from trade_rl.evaluation.rl_allocation.settled_env import (
        SettledAllocationTradingEnv,
    )

    owner = _owner()
    declared = _declarations(case)
    context = owner.SettledNativeValidationContext(**declared)
    env = SettledAllocationTradingEnv(**declared)
    name, _, _, _, actions = case
    injected = (
        _PostReturnFailure(name) if "post_return_capture_failure" in name else None
    )
    stage = "actor_returned" if name.startswith("actor_") else "closing_returned"
    calls, intents, opening, failure, caught = [], [], None, None, None
    with pytest.MonkeyPatch.context() as patch:
        if injected is not None:
            original_note = env._note_stage

            def interrupt_after_adoption(index, current_stage):
                original_note(index, current_stage)
                if current_stage == stage:
                    raise injected

            patch.setattr(env, "_note_stage", interrupt_after_adoption)
        try:
            _, reset_info = env.reset(seed=0)
            frontier = _capture_frontier(env)
            opening = dict(
                episode_id=reset_info["episode_id"],
                index=frontier["index"],
                time_ns=frontier["time_ns"],
                reset_info=reset_info,
                native_account=frontier["native_account"],
                state_digest=frontier["state_digest"],
            )
        except BaseException as error:
            caught = error
            failure = dict(
                origin="reset",
                actor_call_ordinal=None,
                stage="reset_observation",
                attempted_start_index=6,
                error={"type_name": type(error).__name__, "message": None},
            )
        if failure is None:
            for ordinal, action in enumerate(actions):
                index = env.index
                intents.append(owner.SettledNativeActorIntent(ordinal, index, action))
                call = dict(
                    ordinal=ordinal,
                    start_index=index,
                    start_time_ns=int(env.dataset.timestamps[index].astype(np.int64)),
                    action=action,
                    action_type=type(action).__name__,
                    outcome="returned",
                    transition_hex=None,
                    failure_fragment_hex=None,
                )
                calls.append(call)
                try:
                    _, _, terminated, truncated, info = env.step(action)
                    assert truncated is False
                    call["transition_hex"] = info["settled_transition"].hex()
                except BaseException as error:
                    caught = error
                    call["outcome"] = "raised"
                    fragment = getattr(error, "settled_execution_receipt", None)
                    call["failure_fragment_hex"] = (
                        fragment.hex() if type(fragment) is bytes else None
                    )
                    failure = dict(
                        origin="native_call",
                        actor_call_ordinal=ordinal,
                        stage=env._failure_stage,
                        attempted_start_index=env._attempted_start_index,
                        error={"type_name": type(error).__name__, "message": None},
                    )
                    break
                if terminated:
                    break
        raw = dict(
            schema="allocation_settled_native_capture_v1",
            context_digest=context.digest,
            opening=opening,
            actor_calls=calls,
            returned_frontier=_capture_frontier(env),
            failure=failure,
        )
    return dict(
        context=context,
        capture=owner.SettledNativeCapture.from_bytes(canonical_json_bytes(raw)),
        actor_intents=tuple(intents),
        caught=caught,
        injected=injected,
        source_envelope_parity=context._source_envelope == env._source_envelope,
    )


@pytest.fixture(scope="module")
def native_batch(tmp_path_factory):
    """The fixed nine-account batch is shared by every native/mutant test."""
    import os
    import subprocess
    from hashlib import sha256
    from pathlib import Path
    from uuid import uuid4

    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    directory = (
        Path(os.environ["TRADE_RL_NATIVE_VALIDATION_OUTPUT"]) / str(uuid4())
        if "TRADE_RL_NATIVE_VALIDATION_OUTPUT" in os.environ
        else tmp_path_factory.mktemp("settled-native-batch")
    )
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        schema="settled_native_software_batch_v1",
        head=head,
        role="required_ci_replica"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "local_new_native_batch",
        github_run_id=os.environ.get("GITHUB_RUN_ID"),
        github_run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
        case_roster=_ROSTER,
        declared_native_bars=25,
        case_count=9,
        neural_fits=0,
        ridge_solver_calls=0,
        source_sha256=sha256(
            (
                root / "trade_rl/evaluation/rl_allocation/settled_native_validation.py"
            ).read_bytes()
        ).hexdigest(),
        test_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        status="started",
    )
    (directory / "invocation.json").write_bytes(canonical_json_bytes(metadata))
    cases = {}
    for case in _ROSTER:
        collected = _collect_case(case)
        cases[case[0]] = collected
        (directory / (case[0] + ".capture.json")).write_bytes(
            collected["capture"].to_bytes()
        )
    metadata["status"] = "captured"
    metadata["attempted_actor_calls"] = sum(
        len(c["actor_intents"]) for c in cases.values()
    )
    metadata["observed_native_advances"] = sum(
        c["capture"].payload()["returned_frontier"]["index"] - 6 for c in cases.values()
    )
    assert metadata["attempted_actor_calls"] <= 10
    assert metadata["observed_native_advances"] <= 25
    (directory / "invocation.json").write_bytes(canonical_json_bytes(metadata))
    return cases


def _verify(collected, raw=None, context=None):
    owner = _owner()
    capture = (
        collected["capture"]
        if raw is None
        else owner.SettledNativeCapture.from_bytes(canonical_json_bytes(raw))
    )
    return owner.validate_settled_native_capture(
        context or collected["context"],
        capture,
        actor_intents=collected["actor_intents"],
    )


def test_native_batch_literal_wealth_and_admission(native_batch):
    expected = {
        "cash_carry": (1003.003001, 0.0, 0, True, True),
        "nonrl_residual_baseline": (997.0703125, 2.9296875, 2, True, True),
        "direct_long": (992.1875, 7.8125, 2, True, True),
        "direct_short": (992.1875, 7.8125, 2, True, True),
        "partial_then_no_market": (998.4375, 1.5625, 2, True, True),
        "reserve_drawdown_recovery": (992.1875, 7.8125, 2, True, False),
        "signed_debt_end": (-503.90625, 3.90625, 1, False, False),
    }
    for name, (equity, cost, fills, full, admitted) in expected.items():
        collected = native_batch[name]
        assert collected["caught"] is None
        assert collected["source_envelope_parity"]
        report = _verify(collected).payload()
        book = report["returned_frontier"]["native_account"]["book"]
        assert book["equity"] == pytest.approx(equity, rel=0, abs=2e-12)
        assert book["total_cost"] == cost
        assert book["fill_count"] == fills
        assert report["full_structural_settlement"] is full
        assert report["admitted"] is admitted
    assert (
        _verify(native_batch["reserve_drawdown_recovery"]).payload()[
            "returned_frontier"
        ]["native_account"]["book"]["max_drawdown"]
        == 0.25390625
    )


@pytest.mark.parametrize(
    "name,index,prefix",
    [
        ("actor_post_return_capture_failure", 7, 0),
        ("reserve_post_return_capture_failure", 8, 1),
    ],
)
def test_native_batch_failure_keeps_original_exception_and_unvalidated_frontier(
    native_batch, name, index, prefix
):
    collected = native_batch[name]
    assert collected["caught"] is collected["injected"]
    raw = collected["capture"].payload()
    assert len(raw["actor_calls"]) == 1
    assert raw["actor_calls"][0]["transition_hex"] is None
    assert raw["returned_frontier"]["index"] == index
    report = _verify(collected).payload()
    assert report["counts"]["validated_native_rows"] == prefix
    assert report["frontier_coverage"] == "UNVALIDATED_TAIL"
    assert not report["full_structural_settlement"]
    assert not report["admitted"]


def _success_payloads(raw):
    import json

    assert raw["failure"] is None
    return [
        json.loads(bytes.fromhex(call["transition_hex"])) for call in raw["actor_calls"]
    ]


def _rechain_success(context, raw, payloads):
    """Rehash all duplicated states; never execute a native account."""
    owner = _owner()
    episode = raw["opening"]["episode_id"]
    before = raw["opening"]["native_account"]
    raw["context_digest"] = context.digest
    raw["opening"]["state_digest"] = owner.native._state(context, episode, 6, before)
    count, actors, reserves = 0, 0, 0
    for call, payload in zip(raw["actor_calls"], payloads, strict=True):
        previous_equity = before["book"]["equity"]
        payload["source_envelope_digest"] = context._source_digest
        for row in payload["rows"]:
            row["opening_equity"] = before["book"]["equity"]
            row["before_state_digest"] = owner.native._state(
                context, episode, row["start_index"], before
            )
            before = {
                "book": row["native"]["book"],
                "active_orders": row["native"]["active_orders"],
            }
            row["after_state_digest"] = owner.native._state(
                context, episode, row["processing_index"], before
            )
            count += 1
            actors += row["kind"] == "actor"
            reserves += row["kind"] == "reserve"
        full = count == context.bound.clock.physical_horizon_steps
        settled = (
            full
            and not before["active_orders"]
            and all(q == "0" for q in before["book"]["exact_quantities"])
        )
        payload["reward"] = (
            before["book"]["equity"] - previous_equity
        ) / context.initial_capital
        payload["settlement_witness"].update(
            actor_steps_processed=actors,
            physical_steps_processed=count,
            reserve_steps_processed=reserves,
            full_coverage=full,
            settled=settled,
            eligible=settled and before["book"]["max_drawdown"] <= 0.2,
            maximum_drawdown=before["book"]["max_drawdown"],
            frontier_index=before["book"]["as_of_index"],
            native_frontier=before,
        )
        call["transition_hex"] = canonical_json_bytes(payload).hex()
    index = before["book"]["as_of_index"]
    raw["returned_frontier"].update(
        index=index,
        native_account=before,
        state_digest=owner.native._state(context, episode, index, before),
    )
    return raw


def _constructed_open_recovery(collected, opened, *, entry_volume=1.0, requested=5.0):
    """Honest hand-arithmetic variant; no additional observed native origin."""
    from trade_rl.evaluation.rl_allocation.transition_facts import _json
    from trade_rl.simulation.orders.model import OrderIntent

    owner, old_context = _owner(), collected["context"]
    declared = _declarations(_ROSTER[4])
    dataset = declared["dataset"]
    opens, high, low, volume = (
        dataset.open.copy(),
        dataset.high.copy(),
        dataset.low.copy(),
        dataset.volume.copy(),
    )
    volume[7, 0] = entry_volume
    opens[8, 0], high[8, 0], low[8, 0] = opened, max(opened, 100.0), min(opened, 100.0)
    dataset = replace(
        dataset,
        identity_payload_json=None,
        open=opens,
        high=high,
        low=low,
        volume=volume,
    ).with_content_identity({"fixture": "constructed-native-DD", "open": opened})
    declared["dataset"] = dataset
    declared["stream"] = replace(declared["stream"], dataset_id=dataset.dataset_id)
    context = owner.SettledNativeValidationContext(**declared)
    raw = collected["capture"].payload()
    payloads = _success_payloads(raw)
    rows = payloads[0]["rows"]
    initial = raw["opening"]["native_account"]["book"]
    initial["as_of_dataset_id"] = dataset.dataset_id
    full_book = initial.copy()
    full_book.pop("equity")
    full_book[
        "returns_history"
    ] = []  # This variant uses the known fresh FIRST actor only.
    state = content_digest(
        dict(
            schema="independent_allocation_context_v1",
            account_id=context.account_id,
            dataset_id=dataset.dataset_id,
            execution_policy=context.economics_digest,
            risk_config=context.risk_config,
            symbol_index=0,
            decision_index=6,
            decision_time=str(dataset.timestamps[6]),
            book=full_book,
            active_orders=(),
            terminal_orders=(),
        )
    )
    decision = owner.native._decision(old_context, rows[0]["proposal"]["decision"])
    packet, vintage, estimate = (
        context.stream.packets[0],
        context.stream.vintages[0],
        context.estimates[0],
    )
    inputs = replace(
        decision.baseline.inputs,
        source_identity=content_digest(
            dict(
                schema="forecast_allocation_source_v1",
                dataset_id=dataset.dataset_id,
                packet=packet.digest,
                vintage=vintage.digest,
                estimates=estimate.payload(),
            )
        ),
    )
    account_context = replace(decision.baseline.context, state_digest=state)
    decision = replace(
        decision, baseline=decision.baseline.allocator.propose(inputs, account_context)
    )
    proposal = decision.propose(3)
    rows[0]["proposal"] = _json(proposal)
    target_identity = content_digest(
        {"decision": proposal.digest, "target": rows[0]["risk"]["weights"]}
    )
    old_intents = {}
    for row in rows:
        for order in row["accepted_orders"]:
            intent = OrderIntent.from_mapping(order["intent"])
            old_intents[intent.order_id] = intent
    intents = {}
    for old_id, intent in sorted(
        old_intents.items(), key=lambda item: item[1].submit_index
    ):
        values = intent.canonical_payload().copy()
        values.pop("order_id")
        values["dataset_id"] = dataset.dataset_id
        if intent.submit_index == 6:
            values["target_identity"] = target_identity
            values["requested_quantity"] = requested
        if values["replaced_order_id"] is not None:
            values["replaced_order_id"] = intents[values["replaced_order_id"]].order_id
        intents[old_id] = OrderIntent.create(**values)
    for row in rows:
        row["native"]["book"]["as_of_dataset_id"] = dataset.dataset_id
        for order in [*row["accepted_orders"], *row["native"]["active_orders"]]:
            new_intent = intents[order["intent"]["order_id"]]
            order["intent"] = new_intent.canonical_payload()
            if new_intent.submit_index == 6:
                order["remaining_quantity"] = (
                    requested - order["cumulative_filled_quantity"]
                )
        for event in row["native"]["order_events"]:
            old_id = event["order_id"]
            event["order_id"], event["dataset_id"] = (
                intents[old_id].order_id,
                dataset.dataset_id,
            )
            if intents[old_id].submit_index == 6:
                event["requested_quantity"] = requested
                event["remaining_quantity"] += requested - 5.0
            if event["replaced_order_id"] is not None:
                event["replaced_order_id"] = intents[
                    event["replaced_order_id"]
                ].order_id
        for reason in row["native"]["terminal_order_reasons"]:
            reason["order_id"] = intents[reason["order_id"]].order_id
    rows[1]["native"]["execution"]["capacity_evidence"][0][
        "capacity_reference_price"
    ] = opened
    for event in rows[1]["native"]["order_events"]:
        if event["event_type"] == "no_fill":
            event["path_points"] = [
                opened,
                min(opened, 100.0),
                max(opened, 100.0),
                100.0,
            ]
    peak = 1099.21875 if opened == 200.0 else 1000.0
    rows[1]["native"]["book"].update(
        peak_value=peak,
        max_drawdown=1.0 - 999.21875 / 1099.21875
        if opened == 200.0
        else 13 / 256
        if opened == 50.0
        else 1 / 1280,
    )
    rows[2]["native"]["book"].update(
        peak_value=peak,
        max_drawdown=1.0 - 998.4375 / 1099.21875
        if opened == 200.0
        else 13 / 256
        if opened == 50.0
        else 1 / 640,
    )
    entry = rows[0]["native"]["execution"]
    entry.update(
        requested_notional=requested * 100.0,
        requested_notional_by_symbol=[requested * 100.0],
        requested_turnover=requested / 10.0,
        unfilled_turnover=requested / 10.0 - 0.1,
        fill_ratio=1.0 / requested,
    )
    if entry_volume != 1.0:
        entry["capacity_evidence"][0].update(
            processing_volume=entry_volume,
            market_notional=entry_volume * 100.0,
            initial_capacity_notional=entry_volume * 100.0,
            remaining_capacity_notional=entry_volume * 100.0 - 100.0,
        )
        entry["max_participation"], entry["participation_by_symbol"] = (
            1.0 / entry_volume,
            [1.0 / entry_volume],
        )
        for event in rows[0]["native"]["order_events"]:
            if event["event_type"] == "partial_fill":
                event.update(
                    capacity_before=entry_volume * 100.0,
                    capacity_after=entry_volume * 100.0 - 100.0,
                    participation_rate=1.0 / entry_volume,
                )
    return context, _rechain_success(context, raw, payloads)


@pytest.mark.parametrize("change", ["underfill", "intent_size"])
def test_native_batch_coupled_capacity_and_intent_forgery_rejects_locally_coherent_bytes(
    native_batch, change
):
    collected = native_batch["partial_then_no_market"]
    context, honest = _constructed_open_recovery(collected, 100.0)
    assert _verify(collected, honest, context).admitted
    context, forged = _constructed_open_recovery(
        collected,
        100.0,
        entry_volume=2.0 if change == "underfill" else 1.0,
        requested=4.0 if change == "intent_size" else 5.0,
    )
    row = _success_payloads(forged)[0]["rows"][0]
    proposal = (
        _owner().native._decision(context, row["proposal"]["decision"]).propose(3)
    )
    identity = content_digest(
        {"decision": proposal.digest, "target": row["risk"]["weights"]}
    )
    # Old native local arithmetic remains coherent; the NEW source-derived join rejects it.
    _owner().native._native_sources(
        context, row, forged["opening"]["native_account"], target_identity=identity
    )
    with pytest.raises(ValueError, match="invalid successful"):
        _verify(collected, forged, context)


@pytest.mark.parametrize(
    "opened,reason", [(200.0, "peak update sites"), (50.0, "drawdown update sites")]
)
def test_native_batch_coupled_open_recovery_oracle_rejects_endpoint_only_dd(
    native_batch, opened, reason
):
    collected = native_batch["partial_then_no_market"]
    context, honest = _constructed_open_recovery(collected, opened)
    result = _verify(collected, honest, context).payload()
    assert result["admitted"]
    assert result["counts"]["validated_native_rows"] == 3
    assert result["returned_frontier"]["native_account"]["book"][
        "max_drawdown"
    ] == pytest.approx(129 / 1407 if opened == 200.0 else 13 / 256, rel=0, abs=2e-16)
    payloads = _success_payloads(honest)
    rows = payloads[0]["rows"]
    rows[1]["native"]["book"].update(peak_value=1000.0, max_drawdown=1 / 1280)
    rows[2]["native"]["book"].update(peak_value=1000.0, max_drawdown=1 / 640)
    dishonest = _rechain_success(context, honest, payloads)
    with pytest.raises(ValueError) as raised:
        _verify(collected, dishonest, context)
    assert reason in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "change", ["counter", "turnover", "pending_version", "rejected_count", "insolvent"]
)
def test_native_batch_rehashed_relational_forgery_is_rejected(native_batch, change):
    collected = native_batch["direct_long"]
    raw, context = collected["capture"].payload(), collected["context"]
    payloads = _success_payloads(raw)
    for row in payloads[0]["rows"]:
        if change == "counter":
            row["native"]["book"]["fill_count"] += 1
        elif change == "turnover":
            row["native"]["book"]["turnover_total"] += 0.125
        elif change == "pending_version":
            row["accepted_orders"][-1]["evidence_version"] += 1
        elif change == "rejected_count":
            row["native"]["execution"]["rejected_count"] = 1
        else:
            row["native"]["book"]["insolvent"] = True
    with pytest.raises(ValueError, match="invalid successful"):
        _verify(collected, _rechain_success(context, raw, payloads))


@pytest.mark.parametrize("change", ["missing", "malformed_witness", "lying_count"])
def test_native_batch_failed_fragment_never_creates_admission(native_batch, change):
    import json

    collected = native_batch["reserve_post_return_capture_failure"]
    raw = collected["capture"].payload()
    call = raw["actor_calls"][0]
    if change == "missing":
        call["failure_fragment_hex"] = None
        expected_prefix, limitation = 0, "MISSING_FRAGMENT"
    else:
        fragment = json.loads(bytes.fromhex(call["failure_fragment_hex"]))
        if change == "malformed_witness":
            fragment["settlement_witness"]["admitted"] = True
        else:
            fragment["settlement_witness"]["physical_steps_processed"] = 2
        call["failure_fragment_hex"] = canonical_json_bytes(fragment).hex()
        expected_prefix, limitation = 1, "MALFORMED_FRAGMENT"
    report = _verify(collected, raw).payload()
    assert report["counts"]["validated_native_rows"] == expected_prefix
    assert report["frontier_coverage"] == "UNVALIDATED_TAIL"
    assert limitation in report["limitations"]
    assert not report["admitted"]
