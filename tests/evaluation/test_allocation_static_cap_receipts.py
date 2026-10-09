"""Exact versioned native-cap verification; closed synthetic software only."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.test_allocation_global_execution import capability, declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    fixture,
    global_control_fixture,
    global_control_folds,
)
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation import transition_facts
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)
from trade_rl.strategies.rl.allocation_fee_stress import retained_debt_economics_digest

FIXTURES = Path(__file__).with_name("fixtures")
V1_PINS = {
    "uncapped_success": (
        125496,
        "d4bdd10ae70086818af790a7f2d6799d7be72607818cd2db5dfbbb939691ec33",
        "completed",
    ),
    "capped_failure": (
        96422,
        "349dd25777452b5db6f0e683e3e5b9b014bb18cacd16e0471d0672055a3be5e8",
        "integrity_failure",
    ),
}


def historical(name):
    data = (FIXTURES / f"allocation_global_v1_{name}.json").read_bytes()
    length, digest, _ = V1_PINS[name]
    assert len(data) == length
    assert sha256(data).hexdigest() == digest
    return data, json.loads(data)


@pytest.mark.parametrize("name", V1_PINS)
def test_literal_v1_receipt_keeps_bytes_and_meaning_without_live_provenance(
    name, monkeypatch
):
    api = capability()

    def unavailable():
        pytest.fail("historical reader must not obtain current provenance")

    monkeypatch.setattr(api, "build_candidate_run_provenance", unavailable)
    original, raw = historical(name)
    restored = api.GlobalAllocationExecutionReceipt.from_payload(raw)
    assert restored._bytes == original
    assert restored.digest == V1_PINS[name][1]
    assert restored.payload["status"] == V1_PINS[name][2]
    assert restored.payload["schema"] == "allocation_global_execution_receipt_v1"
    if name == "capped_failure":
        row = restored.payload["rows"][0]
        assert row["native_validation_failure"] == {
            "error_type": "ValueError",
            "message": "transition facts differ from their native reconstruction",
        }
        assert (
            row["native"]["proposal"]["decision"]["baseline"]["allocator"][
                "upper_weight"
            ]
            == 0.4
        )
        assert (
            raw["plan"]["cell"]["runtime"]["recipe"]["allocator"]["upper_weight"] == 0.5
        )


@pytest.mark.parametrize(
    ("bad_clock", "message"),
    [
        (False, "transition facts differ from their native reconstruction"),
        (True, "transition native one-bar clock differs"),
    ],
)
def test_literal_v1_multiple_invalid_fields_keep_original_error_order(
    bad_clock, message
):
    _, raw = historical("capped_failure")
    native = deepcopy(raw["rows"][0]["native"])
    native["risk_config"]["max_abs_weight"] = True
    if bad_clock:
        native["processing_index"] = 99
    with pytest.raises(ValueError) as raised:
        transition_facts.validate_allocation_execution_facts(
            native,
            raw["plan"]["cell"]["runtime"]["recipe"],
            dataset_id=raw["plan"]["common"]["dataset_id"],
            action_code=2,
        )
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("risk_cap", "leverage", "expected_upper"),
    [(0.4, 1.0, 0.4), (1.0, 0.3, 0.3), (0.4, 0.3, 0.3)],
)
@pytest.mark.parametrize("kind", ["nonrl", "cash"])
def test_lawful_native_static_caps_complete(risk_cap, leverage, expected_upper, kind):
    api = capability()
    env = global_control_fixture(
        mode="direct" if kind == "cash" else "residual",
        risk=PreTradeRiskConfig(
            max_gross=1.0, max_abs_weight=risk_cap, max_turnover=None
        ),
        cost_changes={"max_leverage": leverage},
    )
    folds = global_control_folds(7)
    plan = declare(api, env, folds, kind=kind)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    receipt = result.receipt.payload
    assert receipt["schema"] == "allocation_global_execution_receipt_v2"
    assert receipt["status"] == "completed"
    assert receipt["reset_count"] == 1
    assert len(receipt["rows"]) == 4
    assert receipt["plan"] == plan.payload
    assert plan.payload["cell"]["runtime"]["recipe"]["allocator"]["upper_weight"] == 0.5
    for row in receipt["rows"]:
        assert row["native_validation_failure"] is None
        allocator = row["native"]["proposal"]["decision"]["baseline"]["allocator"]
        assert allocator["lower_weight"] == 0.0
        assert allocator["upper_weight"] == expected_upper
    if kind == "cash":
        assert env.book.fill_count == 0 and env.book.cash == 1000.0
    else:
        second = receipt["rows"][1]["native"]
        actual = second["proposal"]["decision"]["baseline"]["context"]["current_weight"]
        assert actual > expected_upper
        assert second["current_weights"] == [actual]


@pytest.fixture(scope="module")
def capped_cash_execution():
    api, folds = capability(), global_control_folds(7)
    env = global_control_fixture(
        mode="direct",
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=None),
    )
    plan = declare(api, env, folds, kind="cash")
    return plan, api.run_declared_global_allocation_execution(folds, env, plan)


def validate_v2(native, raw, *, action=2, cost=None, recipe=None):
    runtime = raw["plan"]["cell"]["runtime"]
    return transition_facts.validate_allocation_execution_facts_v2(
        native,
        runtime["recipe"] if recipe is None else recipe,
        dataset_id=raw["plan"]["common"]["dataset_id"],
        action_code=action,
        actual_cost_payload=runtime["cost_payload"] if cost is None else cost,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"upper_weight": 0.3},
        {"upper_weight": 0.5},
        {"lower_weight": 0.1},
        {"risk_aversion": 0.25},
        {"max_turnover": 0.3},
    ],
)
def test_coherently_rebuilt_cash_proposal_cannot_forge_effective_allocator(
    capped_cash_execution, change
):
    # HOLD remains exactly zero with no fills, so other valid native facts do
    # not mask omission of the complete allocator identity check.
    raw = capped_cash_execution[1].receipt.payload
    native = deepcopy(raw["rows"][0]["native"])
    original = native["proposal"]["decision"]
    baseline = original["baseline"]
    inputs = dict(baseline["inputs"])
    for name in ("decision_time", "available_at", "horizon_end"):
        inputs[name] = np.datetime64(inputs[name], "ns")
    context = dict(baseline["context"])
    context["decision_time"] = np.datetime64(context["decision_time"], "ns")
    allocator = AfterCostTargetAllocator(**(baseline["allocator"] | change))
    decision = AllocationDecision(
        baseline=allocator.propose(
            AllocationInputs(**inputs), AllocationContext(**context)
        ),
        action_contract=AllocationActionContract(**original["action_contract"]),
        **{
            name: tuple(original[name])
            if name in ("feature_names", "feature_values")
            else original[name]
            for name in (
                "feature_names",
                "feature_values",
                "max_drawdown",
                "initial_capital",
                "remaining_steps",
                "pending_gross",
                "pending_count",
            )
        },
    )
    proposal = decision.propose(0)
    assert proposal.target_weight == 0.0
    native["proposal"] = transition_facts._json(proposal)
    native["proposal_digest"] = proposal.digest
    native["decision_digest"] = decision.decision_digest
    with pytest.raises(ValueError, match="native reconstruction"):
        validate_v2(native, raw, action=0)


@pytest.mark.parametrize("field", ["fee_rate", "random_seed", "allow_short"])
def test_unchanged_cap_cost_drift_rejects_stale_economics(field):
    _, raw = historical("capped_failure")
    cost = deepcopy(raw["plan"]["cell"]["runtime"]["cost_payload"])
    cost[field] = {"fee_rate": 0.004, "random_seed": 77, "allow_short": False}[field]
    with pytest.raises(ValueError, match="actual cost differs"):
        validate_v2(raw["rows"][0]["native"], raw, cost=cost)


@pytest.mark.parametrize("pin_changed", [False, True])
def test_risk_content_or_pin_drift_remains_rejected(pin_changed):
    _, raw = historical("capped_failure")
    native = deepcopy(raw["rows"][0]["native"])
    recipe = deepcopy(raw["plan"]["cell"]["runtime"]["recipe"])
    if pin_changed:
        recipe["runtime_profile"]["risk_digest"] = "a" * 64
    else:
        native["risk_config"]["max_abs_weight"] = 0.3
    with pytest.raises(ValueError, match="risk config differs"):
        validate_v2(native, raw, recipe=recipe)


BAD_COST_CHANGES = [
    ("max_leverage", True),
    ("fee_rate", True),
    ("fee_rate", -0.1),
    ("random_seed", 1.0),
    ("order_latency_bars", True),
    ("allow_short", 1),
    ("schema_version", "execution_policy_v1"),
    ("trigger_volume_fractions", [1.0, True, 0.25, 0.0]),
    ("trigger_volume_fractions", [1.0, 0.5, 0.0]),
    ("margin_mode", "unsupported"),
    ("extra_cost", 0),
    ("missing_fee", None),
]


def malformed_cost(change):
    _, raw = historical("uncapped_success")
    cost = deepcopy(raw["plan"]["cell"]["runtime"]["cost_payload"])
    field, value = change
    if field == "missing_fee":
        del cost["fee_rate"]
    else:
        cost[field] = value
    return cost


@pytest.mark.parametrize("change", BAD_COST_CHANGES)
def test_complete_cost_validation_refuses_malformed_payload_despite_matching_hash(
    change,
):
    cost = malformed_cost(change)
    with pytest.raises(ValueError):
        transition_facts.validate_allocation_execution_cost_payload(
            cost, economics_digest=retained_debt_economics_digest(cost)
        )


def empty_historical_failure_with_cost(cost):
    from trade_rl.evaluation.allocation_nonrl_walk_forward import (
        allocation_nonrl_rule_digest,
    )
    from trade_rl.evaluation.allocation_scenario_identity import (
        allocation_candidate_recipe_digest,
    )

    _, raw = historical("capped_failure")
    raw.update(
        rows=[],
        segments=[],
        reset_count=0,
        opening_state_digest=None,
        closing_state_digest=None,
    )
    common, cell = raw["plan"]["common"], raw["plan"]["cell"]
    economics = retained_debt_economics_digest(cost)
    common["economics_digest"] = economics
    for recipe in (cell["original_recipe"], cell["runtime"]["recipe"]):
        recipe["runtime_profile"]["economics_digest"] = economics
    cell["runtime"]["cost_payload"] = cost
    cell["original_recipe_digest"] = content_digest(cell["original_recipe"])
    cell["original_candidate_digest"] = allocation_candidate_recipe_digest(
        cell["original_recipe"]
    )
    cell["runtime_recipe_digest"] = content_digest(cell["runtime"]["recipe"])
    cell["control_policy_digest"] = allocation_nonrl_rule_digest(
        cell["original_candidate_digest"]
    )
    raw["common_digest"], raw["cell_plan_digest"] = (
        content_digest(common),
        content_digest(cell),
    )
    return raw


@pytest.mark.parametrize(
    "change",
    [
        ("max_leverage", True),
        ("schema_version", "execution_policy_v1"),
        ("trigger_volume_fractions", [1.0, True, 0.25, 0.0]),
        ("missing_fee", None),
    ],
)
def test_empty_v2_receipt_cannot_bypass_complete_cost_validation(change):
    # The historical reader intentionally checks only the cost digest; do not
    # tighten its shared plan admission when adding new v2 schema semantics.
    raw = empty_historical_failure_with_cost(malformed_cost(change))
    api = capability()
    assert api.GlobalAllocationExecutionReceipt.from_payload(raw).payload["rows"] == []
    raw["schema"] = "allocation_global_execution_receipt_v2"
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionReceipt.from_payload(raw)


def test_valid_empty_v2_failure_is_readable_but_not_completed():
    _, original = historical("capped_failure")
    raw = empty_historical_failure_with_cost(
        original["plan"]["cell"]["runtime"]["cost_payload"]
    )
    raw["schema"] = "allocation_global_execution_receipt_v2"
    restored = capability().GlobalAllocationExecutionReceipt.from_payload(raw)
    assert restored.payload["status"] == "integrity_failure"
    assert restored.payload["rows"] == []
    assert restored.payload["segments"] == []


def test_complete_actual_cost_is_admitted_before_native_reset(monkeypatch):
    # Native config construction historically accepts numeric bools. Keep v1
    # plan semantics, but refuse a new v2 execution before touching its ledger.
    env = global_control_fixture(cost_changes={"fee_rate": True})
    api, folds = capability(), global_control_folds(7)
    plan = declare(api, env, folds)
    calls = []

    def forbidden_reset(*args, **kwargs):
        calls.append("reset")
        raise AssertionError("malformed costs reached native account reset")

    monkeypatch.setattr(type(env), "reset", forbidden_reset)
    with pytest.raises(ValueError):
        api.run_declared_global_allocation_execution(folds, env, plan)
    assert calls == []


def test_immutable_schema_dispatch_refuses_old_capped_failure_as_v2():
    _, raw = historical("capped_failure")
    raw["schema"] = "allocation_global_execution_receipt_v2"
    with pytest.raises(
        ValueError, match="native rejection requires actual reader failure"
    ):
        capability().GlobalAllocationExecutionReceipt.from_payload(raw)


def test_immutable_schema_dispatch_refuses_capped_v2_success_as_v1(
    capped_cash_execution,
):
    raw = capped_cash_execution[1].receipt.payload
    raw["schema"] = "allocation_global_execution_receipt_v1"
    with pytest.raises(ValueError, match="native reconstruction"):
        capability().GlobalAllocationExecutionReceipt.from_payload(raw)


@pytest.mark.parametrize("schema", ["allocation_global_execution_receipt_v3", {}, []])
def test_unknown_receipt_version_is_refused_with_historical_error(schema):
    _, raw = historical("capped_failure")
    raw["schema"] = schema
    with pytest.raises(ValueError, match="unknown global execution receipt schema"):
        capability().GlobalAllocationExecutionReceipt.from_payload(raw)


def test_rewritten_v1_rejection_is_refused():
    _, raw = historical("capped_failure")
    raw["rows"][0]["native_validation_failure"]["message"] = "different rejection"
    with pytest.raises(ValueError, match="differs from original reader"):
        capability().GlobalAllocationExecutionReceipt.from_payload(raw)


def carrier_with_allocator(original, allocator=None, *, flat_prices=False):
    dataset, stream = original.dataset, original.stream
    if flat_prices:
        prices = np.full_like(dataset.close, 100.0)
        dataset = replace(
            dataset,
            open=prices,
            high=prices,
            low=prices,
            close=prices,
            mark_price=prices,
        )
        stream = replace(
            stream,
            packets=tuple(
                replace(packet, decision_close=100.0) for packet in stream.packets
            ),
        )
    allocator = original.allocator if allocator is None else allocator
    recipe = deepcopy(original.recipe)
    recipe["allocator"] = asdict(allocator)
    bound = replace(
        original.bound,
        objective=replace(
            original.bound.objective, deployment_recipe_digest=content_digest(recipe)
        ),
    )
    return AllocationTradingEnv(
        dataset,
        stream=stream,
        estimates=tuple(original._estimates.values()),
        bound=bound,
        action_contract=original.action_contract,
        allocator=allocator,
        execution_cost=original.execution_cost,
        risk_config=original.risk_config,
        feature_indices=original.feature_indices,
        symbol_index=original.symbol_index,
        start_index=original.start_index,
        stop_index=original.stop_index,
        account_id=original.account_id,
        observation_schema=original.observation_schema,
    )


@pytest.mark.parametrize(
    ("lower", "upper", "signal", "expected_lower", "expected_upper"),
    [(-0.5, 0.2, -1, -0.4, 0.2), (-0.2, 0.5, 1, -0.2, 0.4), (0.4, 0.4, 1, 0.4, 0.4)],
)
def test_asymmetric_short_and_singleton_bounds_follow_native_intersection(
    lower, upper, signal, expected_lower, expected_upper
):
    env = global_control_fixture(
        signals=(signal,) * 4,
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=None),
    )
    env = carrier_with_allocator(
        env,
        replace(env.allocator, lower_weight=lower, upper_weight=upper),
        flat_prices=True,
    )
    api, folds = capability(), global_control_folds(7)
    observed = api.run_declared_global_allocation_execution(
        folds, env, declare(api, env, folds)
    )
    first = observed.receipt.payload["rows"][0]["native"]
    actual = first["proposal"]["decision"]["baseline"]["allocator"]
    assert (actual["lower_weight"], actual["upper_weight"]) == (
        expected_lower,
        expected_upper,
    )
    assert first["book"]["exact_quantities"] == ["-4" if signal == -1 else "4"]


@pytest.mark.parametrize("turnover", [False, True])
def test_infeasible_cap_or_allocator_turnover_keeps_original_failure(turnover):
    env = global_control_fixture(
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=None)
    )
    allocator = replace(
        env.allocator,
        lower_weight=0.3 if turnover else 0.45,
        max_turnover=0.1 if turnover else None,
    )
    env = carrier_with_allocator(env, allocator)
    api, folds = capability(), global_control_folds(7)
    with pytest.raises(ValueError) as raised:
        api.run_declared_global_allocation_execution(
            folds, env, declare(api, env, folds)
        )
    raw = raised.value.global_execution_receipt.payload
    assert raw["schema"] == "allocation_global_execution_receipt_v2"
    assert raw["status"] == "integrity_failure"
    assert all(row["native"] is None for row in raw["rows"])
    assert env.book.fill_count == 0


@pytest.mark.parametrize(
    ("risk_cap", "leverage", "quantity", "final_equity"),
    [(0.4, 1.0, 4, 998.4), (1.0, 0.3, 3, 998.8)],
)
def test_capped_native_account_closes_with_independent_literal_fees(
    risk_cap, leverage, quantity, final_equity
):
    from trade_rl.evaluation import allocation_terminal_execution as terminal

    folds, (original,) = fixture(
        ranges=((6, 7),),
        signals=(1, 0, 0, 0),
        fee=0.002,
        risk=PreTradeRiskConfig(
            max_gross=1.0, max_abs_weight=risk_cap, max_turnover=None
        ),
        cost_changes={"max_leverage": leverage},
    )
    env = carrier_with_allocator(original, flat_prices=True)
    native_plan = declare(capability(), env, folds)
    plan = terminal.declare_terminal_allocation_execution(
        folds, env, native_plan, settlement_stop_index=8
    )
    result = terminal.run_terminal_allocation_execution(folds, env, plan)
    raw = result.payload
    prefix = raw["native_receipt"]
    book = prefix["rows"][0]["native"]["book"]
    assert prefix["schema"] == "allocation_global_execution_receipt_v2"
    assert book["exact_quantities"] == [str(quantity)]
    assert book["cash"] == pytest.approx(
        1000.0 - quantity * 100.0 - quantity * 100.0 * 0.002
    )
    assert raw["settlement_complete"] and raw["risk_eligible"]
    assert raw["final_book"]["equity"] == pytest.approx(final_equity)
    assert raw["final_book"]["exact_quantities"] == ["0"]
    assert raw["settled_profit_rate"] == pytest.approx((final_equity - 1000.0) / 1000.0)
    assert raw["closing_opening_state_digest"] == prefix["closing_state_digest"]


def test_default_risk_capped_prefix_can_reach_genuinely_unsettled_close():
    from tests.evaluation.test_allocation_terminal_execution import carrier
    from trade_rl.evaluation import allocation_terminal_execution as terminal

    folds, env = carrier(risk=PreTradeRiskConfig(), partial=True)
    native_plan = declare(capability(), env, folds)
    plan = terminal.declare_terminal_allocation_execution(
        folds, env, native_plan, settlement_stop_index=8
    )
    raw = terminal.run_terminal_allocation_execution(folds, env, plan).payload
    assert raw["native_receipt"]["schema"] == "allocation_global_execution_receipt_v2"
    assert raw["native_receipt"]["rows"][0]["native"]["book"]["exact_quantities"] == [
        "4"
    ]
    assert not raw["settlement_complete"] and not raw["risk_eligible"]
    assert raw["settled_profit_rate"] is None
    # This fixture explicitly permits full participation: volume2 closes2 of
    # the four native units. Carry is charged after each bar's fills: entry
    # funding 4*100*.001=.4, then residual funding 2*110*.001=.22.
    assert env.execution_cost.max_participation_rate == 1.0
    assert env.dataset.volume[8, 0] == 2.0
    assert raw["final_book"]["exact_quantities"] == ["2"]
    assert raw["final_book"]["funding_pnl"] == pytest.approx(-0.62)
    # 1000 - 400 - 4 - .4 + 220 - 2.2 - .22; remaining units mark at110.
    assert raw["final_book"]["cash"] == pytest.approx(813.18)
    assert raw["final_book"]["equity"] == pytest.approx(1033.18)
    assert raw["final_active_orders"]


def test_capped_proposal_preserves_single_actual_drawdown_projection(monkeypatch):
    from trade_rl.risk.pretrade import PreTradeRisk

    env = global_control_fixture(
        signals=(1,) * 4,
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=None),
    )
    env = carrier_with_allocator(env)
    prices = env.dataset.close.copy()
    prices[9:] = 70.0
    env.dataset = replace(
        env.dataset,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        mark_price=prices,
    )
    env.stream = replace(
        env.stream,
        packets=tuple(
            replace(packet, decision_close=float(prices[index, 0]))
            for index, packet in enumerate(env.stream.packets, start=6)
        ),
    )
    env = carrier_with_allocator(env)
    calls, original = [], PreTradeRisk.constrain

    def spy(instance, *args, **kwargs):
        if instance is env.risk:
            calls.append((args[0].copy(), kwargs["current"].copy(), kwargs["drawdown"]))
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(PreTradeRisk, "constrain", spy)
    api, folds = capability(), global_control_folds(7)
    raw = api.run_declared_global_allocation_execution(
        folds, env, declare(api, env, folds)
    ).receipt.payload
    assert len(calls) == 4
    native = raw["rows"][-1]["native"]
    drawdown = raw["rows"][-2]["native"]["book"]["max_drawdown"]
    assert 0.1 < drawdown < 0.2
    assert calls[-1][2] == drawdown == native["proposal"]["decision"]["max_drawdown"]
    assert calls[-1][0].tolist() == [0.4]
    assert calls[-1][1].tolist() == native["current_weights"]
    assert native["proposal"]["target_weight"] == 0.4
    assert native["risk"]["risk_scale"] == pytest.approx((0.2 - drawdown) / 0.1)
    assert native["risk"]["weights"] == pytest.approx([0.4 * (0.2 - drawdown) / 0.1])


def test_capped_baseline_keeps_hard_risk_turnover_separate():
    env = global_control_fixture(
        signals=(1,) * 4,
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=0.1),
    )
    api, folds = capability(), global_control_folds(7)
    raw = api.run_declared_global_allocation_execution(
        folds, env, declare(api, env, folds)
    ).receipt.payload
    first = raw["rows"][0]["native"]
    assert first["proposal"]["target_weight"] == 0.4
    assert first["risk"]["weights"] == pytest.approx([0.1])
    assert first["book"]["exact_quantities"] == ["1"]


@pytest.mark.parametrize("flat_prices", [True, False])
def test_flat_utility_preserves_actual_hold_or_nearest_feasible_cap(flat_prices):
    env = global_control_fixture(
        fee=0.0,
        signals=(1, 0, 0, 0),
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=None),
    )
    env._estimates = {
        clock: replace(value, buy_cost=0.0, sell_cost=0.0)
        for clock, value in env._estimates.items()
    }
    env = carrier_with_allocator(env, flat_prices=flat_prices)
    api, folds = capability(), global_control_folds(7)
    raw = api.run_declared_global_allocation_execution(
        folds, env, declare(api, env, folds)
    ).receipt.payload
    native = raw["rows"][1]["native"]
    proposal = native["proposal"]
    baseline = proposal["decision"]["baseline"]
    assert baseline["objective_value"] == 0.0
    actual = 0.4 if flat_prices else 440.0 / 1040.0
    assert baseline["context"]["current_weight"] == pytest.approx(actual)
    assert native["current_weights"] == pytest.approx([actual])
    assert baseline["target_weight"] == 0.4
    assert proposal["is_hold"] is flat_prices


@pytest.mark.parametrize("kind", ["cash", "nonrl"])
def test_capped_v2_reaches_existing_exact_comparison_evidence(
    kind,
    capped_cash_execution,
):
    from tests.evaluation.test_allocation_global_comparison_evidence import (
        build,
        contract_for_plan,
    )
    from tests.evaluation.test_allocation_global_comparison_evidence import (
        capability as evidence_api,
    )

    if kind == "cash":
        plan, observed = capped_cash_execution
    else:
        env = global_control_fixture(
            risk=PreTradeRiskConfig(
                max_gross=1.0, max_abs_weight=0.4, max_turnover=None
            )
        )
        api, folds = capability(), global_control_folds(7)
        plan = declare(api, env, folds)
        observed = api.run_declared_global_allocation_execution(folds, env, plan)
    contract = contract_for_plan(plan)
    evidence = build(evidence_api(), contract, plan, observed)
    book = observed.receipt.payload["rows"][-1]["native"]["book"]
    assert evidence.terminal_profit_rate == pytest.approx(
        (book["equity"] - 1000) / 1000
    )
    assert evidence.max_drawdown == book["max_drawdown"]
    if kind == "cash":
        assert evidence.terminal_profit_rate == 0.0
        assert evidence.max_drawdown == 0.0
    assert evidence.coverage_complete
