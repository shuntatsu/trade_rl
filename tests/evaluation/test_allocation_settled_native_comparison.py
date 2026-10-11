"""Learner-free receipt grammar and independent native arithmetic oracles."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction

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
from trade_rl.evaluation.rl_allocation.settled_native_validation import (
    SettledNativeActorIntent,
    SettledNativeCapture,
    SettledNativeValidationContext,
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

_GROUPS = (
    "interest_flat_continuous",
    "partial_pending_continuous",
    "profitable_recovered_but_DD_ineligible",
)
_METHODS = ("CASH", "NONRL")


def _comparison():
    from trade_rl.evaluation import allocation_settled_native_comparison

    return allocation_settled_native_comparison


def _declarations(group, method):
    """Literal declared coefficients and sources; no solver or native call."""
    name, actor_steps, reserve_steps = group, 2, 1
    mode = "direct" if method == "CASH" else "residual"
    mu = 0.0 if group == _GROUPS[0] else 3 / 16
    horizon = actor_steps + reserve_steps
    times = np.datetime64("2026-01-01", "ns") + np.arange(12) * np.timedelta64(1, "h")
    close = np.asarray(
        [100, 50, 50, 75, 75, 100, 100, 100, 100, 100, 100, 100], dtype=np.float64
    )[:, None]
    opens, volume, cash_rate = close.copy(), np.full((12, 1), 100000.0), np.zeros(12)
    if name == _GROUPS[0]:
        cash_rate[:] = 8.76
    if name == _GROUPS[1]:
        volume[7:9, 0] = (1.0, 0.0)
    if name == _GROUPS[2]:
        close[8:10, 0] = 125.0
        opens[8, 0], opens[9, 0], volume[8, 0] = 50.0, 125.0, 0.0
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
    ).with_content_identity({"fixture": "native-controls-r1", "group": name})
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
        intercept=mu,
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
            expected_simple_return=mu,
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
            "native-controls-r1-fee-1-over-128",
            buy_cost=1 / 128,
            sell_cost=1 / 128,
            exit_cost=1 / 128,
            cash_return=0.001 if group == _GROUPS[0] else 0.0,
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
        account_id="native-control-r1-" + name + "-" + method,
        observation_schema=schema,
        expected_horizon_seconds=3600,
    )


def _declared_groups():
    owner = _comparison()
    declarations = {
        (group, method): _declarations(group, method)
        for group in _GROUPS
        for method in _METHODS
    }
    contexts = {
        key: SettledNativeValidationContext(**value)
        for key, value in declarations.items()
    }
    groups = tuple(
        owner.SettledNativeControlGroup(
            group, contexts[group, "CASH"], contexts[group, "NONRL"]
        )
        for group in _GROUPS
    )
    return declarations, contexts, groups


def _frontier(env):
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_account_facts,
    )

    try:
        return dict(
            index=env.index,
            time_ns=int(env.dataset.timestamps[env.index].astype(np.int64)),
            native_account=native_allocation_account_facts(env.book, env.order_book),
            state_digest=env._state_digest(env.index, env.book, env.order_book),
            encoding_error=None,
        ), None
    except BaseException as error:
        return dict(
            index=None,
            time_ns=None,
            native_account=None,
            state_digest=None,
            encoding_error={"type_name": type(error).__name__, "message": None},
        ), error


def _collect_arm(declaration, context, method):
    """Only one new account and actual native capture; no prior collector calls."""
    from trade_rl.evaluation.rl_allocation.settled_env import (
        SettledAllocationTradingEnv,
    )

    env = SettledAllocationTradingEnv(**declaration)
    planned = tuple(
        SettledNativeActorIntent(i, 6 + i, 0 if method == "CASH" else 2)
        for i in range(2)
    )
    calls, opening, failure, caught = [], None, None, None
    try:
        _, reset_info = env.reset(seed=0)
    except BaseException as error:
        caught = error
        failure = dict(
            origin="reset",
            actor_call_ordinal=None,
            stage="reset_observation",
            attempted_start_index=6,
            error={"type_name": type(error).__name__, "message": None},
        )
    else:
        frontier, caught = _frontier(env)
        if caught is not None:
            failure = dict(
                origin="capture",
                actor_call_ordinal=None,
                stage="capture_encoding",
                attempted_start_index=6,
                error={"type_name": type(caught).__name__, "message": None},
            )
    if failure is None:
        opening = dict(
            episode_id=reset_info["episode_id"],
            index=frontier["index"],
            time_ns=frontier["time_ns"],
            reset_info=reset_info,
            native_account=frontier["native_account"],
            state_digest=frontier["state_digest"],
        )
        for intent in planned:
            assert env.index == intent.start_index
            call = dict(
                ordinal=intent.ordinal,
                start_index=env.index,
                start_time_ns=int(env.dataset.timestamps[env.index].astype(np.int64)),
                action=intent.action,
                action_type="int",
                outcome="returned",
                transition_hex=None,
                failure_fragment_hex=None,
            )
            calls.append(call)
            try:
                _, _, terminated, truncated, info = env.step(intent.action)
            except BaseException as error:
                caught = error
                call["outcome"] = "raised"
                fragment = getattr(error, "settled_execution_receipt", None)
                call["failure_fragment_hex"] = (
                    fragment.hex() if type(fragment) is bytes else None
                )
                failure = dict(
                    origin="native_call",
                    actor_call_ordinal=intent.ordinal,
                    stage=env._failure_stage,
                    attempted_start_index=env._attempted_start_index,
                    error={"type_name": type(error).__name__, "message": None},
                )
                break
            try:
                transition = info["settled_transition"]
                assert type(transition) is bytes
                call["transition_hex"] = transition.hex()
                assert not truncated
            except BaseException as error:
                caught = error
                failure = dict(
                    origin="capture",
                    actor_call_ordinal=intent.ordinal,
                    stage="capture_encoding",
                    attempted_start_index=intent.start_index,
                    error={"type_name": type(error).__name__, "message": None},
                )
                break
            if terminated:
                break
    frontier, frontier_error = _frontier(env)
    if frontier_error is not None and failure is None:
        caught = frontier_error
        failure = dict(
            origin="capture",
            actor_call_ordinal=calls[-1]["ordinal"] if calls else None,
            stage="capture_encoding",
            attempted_start_index=env.index,
            error={"type_name": type(frontier_error).__name__, "message": None},
        )
    raw = dict(
        schema="allocation_settled_native_capture_v1",
        context_digest=context.digest,
        opening=opening,
        actor_calls=calls,
        returned_frontier=frontier,
        failure=failure,
    )
    encoded = canonical_json_bytes(raw)
    try:
        capture = SettledNativeCapture.from_bytes(encoded)
    except BaseException as error:
        # Malformed capture cannot be promoted to a diagnostic success. Preserve
        # its exact raw bytes and original error for the batch owner, then stop.
        if caught is not None:
            caught.raw_capture_bytes = encoded
            raise caught from error
        raise
    return capture, planned, caught, context._source_envelope == env._source_envelope


@pytest.fixture(scope="module")
def matched_batch(tmp_path_factory):
    import os
    import subprocess
    from hashlib import sha256
    from pathlib import Path
    from uuid import uuid4

    owner = _comparison()
    declarations, contexts, groups = _declared_groups()
    plan = owner.declare_settled_native_control_comparison(groups)
    root = Path(__file__).resolve().parents[2]
    directory = (
        Path(os.environ["TRADE_RL_NATIVE_CONTROL_OUTPUT"]) / str(uuid4())
        if "TRADE_RL_NATIVE_CONTROL_OUTPUT" in os.environ
        else tmp_path_factory.mktemp("matched-native-controls")
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plan.json").write_bytes(plan.to_bytes())
    metadata = dict(
        schema="matched_native_control_batch_v1",
        head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        role="required_ci_replica"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "local_new_native_control_batch",
        github_run_id=os.environ.get("GITHUB_RUN_ID"),
        github_run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
        group_roster=_GROUPS,
        method_roster=_METHODS,
        account_count=6,
        declared_resets=6,
        declared_actor_calls=12,
        declared_native_bars=18,
        new_owner_neural_fits=0,
        new_owner_ridge_solver_calls=0,
        plan_sha256=sha256(plan.to_bytes()).hexdigest(),
        test_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "trade_rl/evaluation/allocation_settled_native_comparison.py",
                "trade_rl/evaluation/rl_allocation/settled_comparison_source.py",
            )
        },
        status="started",
    )
    (directory / "invocation.json").write_bytes(canonical_json_bytes(metadata))
    arms, exceptions, parities = [], [], []
    for group in _GROUPS:
        for method in _METHODS:
            try:
                capture, planned, caught, parity = _collect_arm(
                    declarations[group, method], contexts[group, method], method
                )
            except BaseException as error:
                encoded = getattr(error, "raw_capture_bytes", None)
                if type(encoded) is bytes:
                    (
                        directory / (group + "-" + method + ".invalid-capture.json")
                    ).write_bytes(encoded)
                metadata.update(
                    status="capture_failed", error_type=type(error).__name__
                )
                (directory / "invocation.json").write_bytes(
                    canonical_json_bytes(metadata)
                )
                raise
            (directory / (group + "-" + method + ".capture.json")).write_bytes(
                capture.to_bytes()
            )
            arms.append(
                owner.SettledNativeControlArm(
                    group, method, contexts[group, method], capture, planned
                )
            )
            exceptions.append(caught)
            parities.append(parity)
    metadata.update(
        status="captured",
        attempted_actor_calls=sum(
            len(arm.capture.payload()["actor_calls"]) for arm in arms
        ),
        observed_native_advances=sum(
            (arm.capture.payload()["returned_frontier"]["index"] or 6) - 6
            for arm in arms
        ),
    )
    assert (
        metadata["attempted_actor_calls"] <= 12
        and metadata["observed_native_advances"] <= 18
    )
    (directory / "invocation.json").write_bytes(canonical_json_bytes(metadata))
    report = owner.build_settled_native_control_comparison(plan, tuple(arms))
    (directory / "report.json").write_bytes(report.to_bytes())
    return dict(
        plan=plan,
        arms=tuple(arms),
        report=report,
        exceptions=exceptions,
        parities=parities,
        directory=directory,
    )


def test_declaration_rejects_duplicate_accounts_and_role_swaps():
    owner = _comparison()
    _, _, groups = _declared_groups()
    with pytest.raises(ValueError):
        owner.declare_settled_native_control_comparison((groups[0], groups[0]))
    with pytest.raises(ValueError):
        owner.declare_settled_native_control_comparison(
            (
                owner.SettledNativeControlGroup(
                    "swap", groups[0].nonrl_context, groups[0].cash_context
                ),
            )
        )


@pytest.mark.parametrize("array", ["global_features", "features"])
def test_opaque_dataset_id_cannot_hide_unselected_or_future_array_drift(array):
    owner = _comparison()
    _, _, groups = _declared_groups()
    context = groups[0].nonrl_context
    changed = getattr(context.dataset, array).copy()
    changed[10].flat[0] = 1.0
    object.__setattr__(context.dataset, array, changed)
    # Existing actor/native source selection omits these values; whole identity must still reject.
    context.validate_sources()
    with pytest.raises(ValueError):
        owner.declare_settled_native_control_comparison(groups)


def test_recomputed_identity_does_not_authenticate_a_changed_live_descriptor():
    owner = _comparison()
    declared = _declarations(_GROUPS[0], "NONRL")
    dataset = declared["dataset"]
    object.__setattr__(dataset, "global_feature_names", ("renamed",))
    assert dataset.identity_verified
    context = SettledNativeValidationContext(**declared)
    with pytest.raises(ValueError, match="live descriptor"):
        owner.declare_settled_native_control_comparison(
            (owner.SettledNativeControlGroup("descriptor", context, context),)
        )


def test_object_array_pointers_cannot_authorize_source_matching():
    owner = _comparison()
    declared = _declarations(_GROUPS[0], "CASH")
    dataset = declared["dataset"]
    object.__setattr__(
        dataset, "global_features", dataset.global_features.astype(object)
    )
    object.__setattr__(dataset, "dataset_id", dataset.recomputed_dataset_id())
    assert dataset.identity_verified
    declared["stream"] = replace(declared["stream"], dataset_id=dataset.dataset_id)
    context = SettledNativeValidationContext(**declared)
    with pytest.raises(ValueError, match="plain deterministic"):
        owner.declare_settled_native_control_comparison(
            (owner.SettledNativeControlGroup("objects", context, context),)
        )


def test_one_sided_valid_cost_belief_difference_is_not_a_matched_control():
    owner = _comparison()
    _, _, groups = _declared_groups()
    declared = _declarations(_GROUPS[0], "NONRL")
    declared["estimates"] = tuple(
        replace(value, cash_return=0.002) for value in declared["estimates"]
    )
    context = SettledNativeValidationContext(**declared)
    with pytest.raises(ValueError, match="fixed financial/source conditions"):
        owner.declare_settled_native_control_comparison(
            (replace(groups[0], nonrl_context=context),)
        )


def test_same_canonical_dataset_id_cannot_hide_changed_raw_dtype():
    owner = _comparison()
    _, _, groups = _declared_groups()
    context = groups[0].nonrl_context
    object.__setattr__(
        context.dataset,
        "global_features",
        context.dataset.global_features.astype(np.float64),
    )
    assert context.dataset.identity_verified
    context.validate_sources()
    with pytest.raises(ValueError, match="fixed financial/source conditions"):
        owner.declare_settled_native_control_comparison(groups)


def test_matched_native_literals_and_risk_gate(matched_batch):
    assert all(error is None for error in matched_batch["exceptions"])
    assert all(matched_batch["parities"])
    groups = matched_batch["report"].payload()["groups"]
    carry, partial, risk = groups
    for arm in carry["arms"]:
        assert arm["observed_terminal_equity"].hex() == "0x1.f5806256366d7p+9"
        assert (
            arm["observed_total_cost"]
            == arm["observed_fill_count"]
            == arm["observed_turnover_total"]
            == arm["observed_max_drawdown"]
            == 0
        )
        assert arm["verification"]["admitted"] is True
    assert (
        carry["primary_terminal_difference"]
        == carry["primary_return_difference"]
        == 0.0
    )
    cash, nonrl = partial["arms"]
    assert cash["observed_terminal_equity"] == 1000.0
    assert nonrl["observed_terminal_equity"] == 998.4375
    assert nonrl["observed_total_cost"] == 1.5625 and nonrl["observed_fill_count"] == 2
    assert nonrl["verification"]["admitted"] is True
    assert partial["primary_terminal_difference"] == -1.5625
    assert partial["primary_return_difference"] == -1 / 640
    cash, nonrl = risk["arms"]
    assert (
        cash["observed_terminal_equity"] == 1000.0
        and cash["verification"]["admitted"] is True
    )
    assert nonrl["observed_terminal_equity"] == 1116.2109375
    assert (
        nonrl["observed_total_cost"] == 8.7890625 and nonrl["observed_fill_count"] == 2
    )
    assert nonrl["observed_max_drawdown"] == 65 / 256
    assert nonrl["verification"]["full_structural_settlement"] is True
    assert nonrl["verification"]["admitted"] is False
    assert (
        risk["primary_terminal_difference"] is risk["primary_return_difference"] is None
    )
    # The rational mathematical cash oracle is retained separately from emitted IEEE order.
    assert Fraction(1000) * Fraction(1001, 1000) ** 3 == Fraction(1003003001, 1000000)


def test_native_pending_state_survives_the_second_actor_and_reserve(matched_batch):
    import json

    for position, requested in ((3, 1023 / 256), (5, -5 / 256)):
        arm = matched_batch["arms"][position]
        first, second = [
            json.loads(bytes.fromhex(call["transition_hex"]))
            for call in arm.capture.payload()["actor_calls"]
        ]
        assert (
            first["rows"][-1]["after_state_digest"]
            == second["rows"][0]["before_state_digest"]
        )
        assert [r["processing_index"] for r in first["rows"] + second["rows"]] == [
            7,
            8,
            9,
        ]
        assert (
            second["rows"][0]["accepted_orders"][0]["intent"]["requested_quantity"]
            == requested
        )
        assert second["rows"][0]["native"]["execution"]["filled_notional"] == 0.0
        assert second["rows"][-1]["kind"] == "reserve"


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "order", "wrong_action", "wrong_type"]
)
def test_complete_roster_and_full_planned_actions_are_required(matched_batch, change):
    owner = _comparison()
    arms = list(matched_batch["arms"])
    if change == "missing":
        arms.pop()
    elif change == "duplicate":
        arms[-1] = arms[0]
    elif change == "order":
        arms.reverse()
    elif change == "wrong_type":
        arms[0] = matched_batch["report"]
    else:
        arms[0] = replace(
            arms[0],
            planned_actor_intents=(
                SettledNativeActorIntent(0, 6, 2),
                SettledNativeActorIntent(1, 7, 2),
            ),
        )
    with pytest.raises(ValueError):
        owner.build_settled_native_control_comparison(
            matched_batch["plan"], tuple(arms)
        )


@pytest.mark.parametrize("stage", ["reset", "after_actor1"])
def test_constructed_failed_present_arm_keeps_plan_and_is_not_terminal(
    matched_batch, stage
):
    import json

    owner = _comparison()
    arms = list(matched_batch["arms"])
    arm = arms[3]
    raw = arm.capture.payload()
    raw["failure"] = dict(
        origin="capture",
        actor_call_ordinal=1,
        stage="capture_encoding",
        attempted_start_index=7,
        error={"type_name": "ConstructedDiagnostic", "message": None},
    )
    raw["actor_calls"] = raw["actor_calls"][:1]
    row = json.loads(bytes.fromhex(raw["actor_calls"][0]["transition_hex"]))["rows"][-1]
    raw["returned_frontier"] = dict(
        index=7,
        time_ns=int(arm.context.dataset.timestamps[7].astype(np.int64)),
        native_account={
            "book": row["native"]["book"],
            "active_orders": row["native"]["active_orders"],
        },
        state_digest=row["after_state_digest"],
        encoding_error=None,
    )
    if stage == "reset":
        raw["opening"], raw["actor_calls"] = None, []
        raw["failure"].update(
            origin="reset",
            actor_call_ordinal=None,
            stage="reset_observation",
            attempted_start_index=6,
        )
        raw["returned_frontier"] = dict(
            index=None,
            time_ns=None,
            native_account=None,
            state_digest=None,
            encoding_error={"type_name": "ConstructedDiagnostic", "message": None},
        )
    arms[3] = replace(
        arm, capture=SettledNativeCapture.from_bytes(canonical_json_bytes(raw))
    )
    report = owner.build_settled_native_control_comparison(
        matched_batch["plan"], tuple(arms)
    ).payload()
    result = report["groups"][1]
    diagnostic = result["arms"][1]
    assert len(report["groups"]) == 3 and all(
        len(g["arms"]) == 2 for g in report["groups"]
    )
    assert (
        diagnostic["observed_terminal_equity"]
        is diagnostic["observed_terminal_return"]
        is None
    )
    assert diagnostic["verification"]["admitted"] is False
    assert diagnostic["attempted_actor_intent_count"] == (0 if stage == "reset" else 1)
    assert (
        result["primary_terminal_difference"]
        is result["primary_return_difference"]
        is None
    )


def test_rescaled_fixed_capital_actor_reward_is_rejected(matched_batch):
    import json

    owner = _comparison()
    arms = list(matched_batch["arms"])
    raw = arms[3].capture.payload()
    transition = json.loads(bytes.fromhex(raw["actor_calls"][0]["transition_hex"]))
    transition["reward"] *= 2.0
    raw["actor_calls"][0]["transition_hex"] = canonical_json_bytes(transition).hex()
    arms[3] = replace(
        arms[3], capture=SettledNativeCapture.from_bytes(canonical_json_bytes(raw))
    )
    with pytest.raises(ValueError):
        owner.build_settled_native_control_comparison(
            matched_batch["plan"], tuple(arms)
        )


def test_detached_plan_and_report_payload_mutations_do_not_change_identity(
    matched_batch,
):
    plan, report = matched_batch["plan"], matched_batch["report"]
    before_plan, before_report = plan.to_bytes(), report.to_bytes()
    plan.payload()["groups"].clear()
    report.payload()["groups"][0]["arms"][0]["verification"]["admitted"] = False
    assert plan.to_bytes() == before_plan and report.to_bytes() == before_report


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_rehashed_nested_plan_content_cannot_authorize_a_comparison(
    matched_batch, change
):
    owner = _comparison()
    raw = matched_batch["plan"].payload()
    common = raw["groups"][0]["common_material"]
    if change == "extra":
        common["recipe"]["action"]["allow_profit"] = True
    else:
        del common["source_envelope"]["native_scope"]["settlement_stop_time_ns"]
    raw["groups"][0]["common_material_digest"] = content_digest(common)
    detached = owner.SettledNativeControlComparisonPlan(canonical_json_bytes(raw))
    with pytest.raises(ValueError, match="context/source changed"):
        owner.build_settled_native_control_comparison(detached, matched_batch["arms"])


@pytest.mark.parametrize("level", ["group", "arm", "verification"])
def test_report_grammar_rejects_extra_authority_flags(matched_batch, level):
    owner = _comparison()
    raw = matched_batch["report"].payload()
    target = raw["groups"][0]
    if level != "group":
        target = target["arms"][0]
    if level == "verification":
        target = target["verification"]
    target["approved_for_market"] = True
    with pytest.raises(ValueError):
        owner.SettledNativeControlComparisonReport(canonical_json_bytes(raw))


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_terminal_equity", "PASS"),
        ("attempted_actor_intent_count", -1),
        ("observed_frontier_index", True),
        ("observed_fill_count", 1.5),
    ],
)
def test_report_metrics_require_native_scalar_types(matched_batch, field, value):
    owner = _comparison()
    raw = matched_batch["report"].payload()
    raw["groups"][0]["arms"][0][field] = value
    with pytest.raises(ValueError):
        owner.SettledNativeControlComparisonReport(canonical_json_bytes(raw))


def test_constructed_failed_terminal_frontier_is_observed_but_not_admitted(
    matched_batch,
):
    owner = _comparison()
    arms = list(matched_batch["arms"])
    raw = arms[3].capture.payload()
    raw["failure"] = dict(
        origin="capture",
        actor_call_ordinal=1,
        stage="capture_encoding",
        attempted_start_index=7,
        error={"type_name": "ConstructedPostReturnDiagnostic", "message": None},
    )
    arms[3] = replace(
        arms[3], capture=SettledNativeCapture.from_bytes(canonical_json_bytes(raw))
    )
    group = owner.build_settled_native_control_comparison(
        matched_batch["plan"], tuple(arms)
    ).payload()["groups"][1]
    assert group["arms"][1]["observed_frontier_index"] == 9
    assert group["arms"][1]["observed_terminal_equity"] == 998.4375
    assert group["arms"][1]["verification"]["admitted"] is False
    assert (
        group["primary_terminal_difference"]
        is group["primary_return_difference"]
        is None
    )


def test_source_drift_during_pure_capture_validation_cannot_publish(
    matched_batch, monkeypatch
):
    owner = _comparison()
    original = owner.validate_settled_native_capture
    dataset = matched_batch["arms"][0].context.dataset
    prior = dataset.global_features

    def validate_then_drift(*args, **kwargs):
        result = original(*args, **kwargs)
        changed = prior.copy()
        changed[10, 0] = 1.0
        object.__setattr__(dataset, "global_features", changed)
        return result

    monkeypatch.setattr(owner, "validate_settled_native_capture", validate_then_drift)
    try:
        with pytest.raises(ValueError):
            owner.build_settled_native_control_comparison(
                matched_batch["plan"], matched_batch["arms"]
            )
    finally:
        object.__setattr__(dataset, "global_features", prior)
