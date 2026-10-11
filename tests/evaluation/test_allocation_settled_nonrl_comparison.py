"""Standalone two-arm controls with hand-solved native settlement examples."""

from dataclasses import replace
from importlib import import_module

import pytest

from tests.evaluation.test_allocation_settled_forecast_comparison import group_fixture
from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
)
from trade_rl.evaluation.allocation_terminal_execution import _terminal_recipe


def capability():
    api = import_module("trade_rl.evaluation.allocation_settled_forecast_comparison")
    for name in (
        "TerminalAllocationNonRLControls",
        "SettledNonRLComparisonPlan",
        "SettledNonRLComparisonReport",
        "build_settled_nonrl_comparison",
    ):
        assert hasattr(api, name), "standalone NONRL/CASH comparison is missing"
    return api


def pair(api, account="A", scenario="base", *, partial=False):
    folds, envs, _, declared = group_fixture(account, scenario, partial=partial)
    return (
        folds,
        (envs[0], envs[2]),
        api.TerminalAllocationNonRLControls(declared[0], declared[2]),
    )


def prepared(api, *, partial=False, accounts=("A",), scenarios=("base",)):
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    groups, runs = [], []
    for account in accounts:
        for scenario in scenarios:
            folds, envs, group = pair(api, account, scenario, partial=partial)
            groups.append(group)
            runs.append((folds, envs, group))
    plan = api.SettledNonRLComparisonPlan(tuple(groups), accounts, scenarios)
    # The entire matrix exists before any native account starts.
    records = tuple(
        runner.run_terminal_allocation_execution(folds, env, arm).payload
        for folds, envs, group in runs
        for env, arm in zip(envs, group.plans, strict=True)
    )
    return plan, records


def test_literal_two_arm_net_profit_keeps_actual_trading_charges():
    api = capability()
    plan, records = prepared(api)
    report = api.build_settled_nonrl_comparison(plan, records)
    group = report.payload["groups"][0]
    # q=32/19 at price100, entry+exit fee .002: cost64/95.
    assert group["roles"]["nonrl"]["final_equity"] == pytest.approx(94936 / 95)
    assert group["roles"]["nonrl"]["total_cost"] == pytest.approx(64 / 95)
    assert group["roles"]["cash"]["final_equity"] == 1000
    assert group["roles"]["cash"]["total_cost"] == 0
    assert group["nonrl_minus_cash"] == pytest.approx(-64 / 95000)
    assert report.payload["comparison_complete"]
    assert len(report.receipts) == 2
    assert report.payload["claim_scope"] == (
        "bounded_software_comparison_no_research_qualification"
    )


def test_two_arm_cartesian_matrix_is_order_independent():
    api = capability()
    plan, records = prepared(api, accounts=("A", "B"), scenarios=("base", "fee2"))
    first = api.build_settled_nonrl_comparison(plan, records)
    second = api.build_settled_nonrl_comparison(plan, tuple(reversed(records)))
    assert len(records) == 8 and len(first.payload["groups"]) == 4
    assert first.payload == second.payload and first.digest == second.digest


def test_partial_close_keeps_both_rows_without_a_paired_delta():
    api = capability()
    plan, records = prepared(api, partial=True)
    row = api.build_settled_nonrl_comparison(plan, records).payload["groups"][0]
    assert not row["comparison_complete"] and row["nonrl_minus_cash"] is None
    assert not row["roles"]["nonrl"]["settlement_complete"]
    assert row["roles"]["cash"]["final_equity"] == 1000


def test_settled_drawdown_breach_has_no_paired_delta():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    groups, runs = [], []
    for kind in ("nonrl", "cash"):
        folds, env = carrier(kind=kind)
        for name in ("open", "high", "low", "close", "mark_price"):
            values = env.dataset.resolved_array(name).copy()
            values[8:] = 50
            object.__setattr__(env.dataset, name, values)
        groups.append(plans(runner, env, folds, kind=kind))
        runs.append((folds, env))
    group = api.TerminalAllocationNonRLControls(*groups)
    account, scenario = group.key
    plan = api.SettledNonRLComparisonPlan((group,), (account,), (scenario,))
    records = tuple(
        runner.run_terminal_allocation_execution(folds, env, arm).payload
        for (folds, env), arm in zip(runs, group.plans, strict=True)
    )
    row = api.build_settled_nonrl_comparison(plan, records).payload["groups"][0]
    assert row["roles"]["nonrl"]["settlement_complete"]
    assert row["roles"]["nonrl"]["final_equity"] == pytest.approx(742)
    assert row["roles"]["nonrl"]["maximum_drawdown"] == pytest.approx(0.258)
    assert not row["comparison_complete"] and row["nonrl_minus_cash"] is None


def repin_native(plan, raw):
    native = GlobalAllocationExecutionPlan.from_payload(raw)
    recipe = _terminal_recipe(
        native,
        plan.settlement_stop_index,
        plan.closing_clocks_ns,
        plan.closing_source_digest,
        plan.bound.clock,
    )
    return replace(
        plan,
        native_plan=native,
        bound=replace(
            plan.bound,
            objective=replace(
                plan.bound.objective, deployment_recipe_digest=content_digest(recipe)
            ),
        ),
    )


@pytest.mark.parametrize("field", ["forecast_context_digest", "cost_beliefs_digest"])
def test_same_financial_market_cannot_hide_different_decision_inputs(field):
    api = capability()
    _, _, group = pair(api)
    raw = group.cash_plan.native_plan.payload
    assert field in raw["common"]
    raw["common"][field] = "a" * 64
    foreign = repin_native(group.cash_plan, raw)
    with pytest.raises(ValueError):
        api.TerminalAllocationNonRLControls(group.nonrl_plan, foreign)


def test_same_common_context_cannot_hide_different_native_source_envelope():
    api = capability()
    _, _, group = pair(api)
    raw = group.cash_plan.native_plan.payload
    raw["cell"]["runtime"]["source_context_digest"] = "a" * 64
    foreign = repin_native(group.cash_plan, raw)
    with pytest.raises(ValueError):
        api.TerminalAllocationNonRLControls(group.nonrl_plan, foreign)


def test_forecast_factor_report_is_not_relabelled_as_a_two_arm_baseline():
    api = capability()
    _, _, _, declared = group_fixture()
    with pytest.raises(ValueError):
        api.TerminalAllocationNonRLControls(declared[1], declared[2])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "extra"])
def test_two_arm_observation_roster_cannot_be_filtered_or_substituted(mutation):
    api = capability()
    plan, records = prepared(api)
    values = list(records)
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[1] = values[0]
    elif mutation == "foreign":
        values[1]["plan_digest"] = "a" * 64
    else:
        values.append(values[0])
    with pytest.raises(ValueError):
        api.build_settled_nonrl_comparison(plan, tuple(values))


def test_summary_tampering_is_rejected_by_the_closed_native_receipt_reader():
    api = capability()
    plan, records = prepared(api)
    records[0]["settled_profit_rate"] = 1.0
    with pytest.raises(ValueError):
        api.build_settled_nonrl_comparison(plan, records)


def test_declared_two_arm_matrix_cannot_omit_or_duplicate_a_group():
    api = capability()
    _, _, group = pair(api)
    with pytest.raises(ValueError):
        api.SettledNonRLComparisonPlan((group,), ("A", "B"), ("base",))
    with pytest.raises(ValueError):
        api.SettledNonRLComparisonPlan((group, group), ("A",), ("base",))
