"""A predeclared complete matrix; no fit, Study or economic data access."""

from dataclasses import replace
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_global_execution import declare, forecast_controls
from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.forecasts.simple_return import SimpleReturnTrainingSet
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def capability():
    name = "trade_rl.evaluation.allocation_settled_forecast_comparison"
    assert util.find_spec(name) is not None, (
        "settled controlled forecast report missing"
    )
    return import_module(name)


def group_fixture(account="A", scenario="base", *, partial=False):
    global_api = import_module("trade_rl.evaluation.allocation_global_execution")
    terminal = import_module("trade_rl.evaluation.allocation_terminal_execution")
    risk = PreTradeRiskConfig(max_abs_weight=0.4, max_turnover=None)
    folds, (original,) = fixture(ranges=((6, 7),), risk=risk)
    vintage = original.stream.vintages[0]
    old = vintage.training.log_training
    times = original.dataset.timestamps
    start, end = np.ones(2), np.array([1.0, 1 + np.sqrt(19) / 4])
    trace = replace(
        old.trace,
        row_symbols=("S0", "S0"),
        start_times=times[:2],
        end_times=times[1:3],
        start_available_at=times[:2],
        end_available_at=times[1:3],
        start_close=start,
        end_close=end,
    )
    training = SimpleReturnTrainingSet(
        replace(
            old,
            features=np.zeros((2, 1)),
            labels=np.log(end),
            label_end_times=times[1:3],
            sample_weights=np.ones(2),
            trace=trace,
        )
    )
    allocator = replace(original.allocator, risk_aversion=1)
    envs = []
    for role, forecast in zip(
        ("control", "treatment", "cash"), (1 / 8, 19 / 24, 1 / 8), strict=True
    ):
        v = replace(
            vintage,
            training=training,
            model=replace(
                vintage.model,
                coefficients=np.zeros(1),
                intercept=forecast,
                n_samples=2,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
            ),
        )
        stream = replace(
            original.stream,
            vintages=(v,),
            packets=tuple(
                replace(
                    p,
                    expected_simple_return=forecast,
                    fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                    vintage_digest=v.digest,
                    decision_close=100,
                )
                for p in original.stream.packets
            ),
        )
        prices = np.full((12, 1), 100.0)
        volume = original.dataset.volume.copy()
        if partial:
            volume[8] = 1
        dataset = replace(
            original.dataset,
            open=prices,
            high=prices,
            low=prices,
            close=prices,
            mark_price=prices,
            volume=volume,
        )
        action = replace(
            original.action_contract, mode="direct" if role == "cash" else "residual"
        )
        cost = replace(
            original.execution_cost, fee_rate=0.004 if scenario == "fee2" else 0.002
        )
        from trade_rl.simulation import MarketExecutor
        from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile

        economics = MarketExecutor(
            dataset, cost, insolvency_valuation="retain_debt"
        ).execution_policy_digest
        profile = AllocationRuntimeProfile(
            economics, content_digest(risk), 1000.0, "USD", 3600, 3600
        )
        recipe = allocation_recipe_payload_v2(
            action,
            ("signal",),
            allocator=allocator,
            expected_horizon_seconds=3600,
            observation_schema=original.observation_schema,
            runtime_profile=profile,
        )
        bound = replace(
            original.bound,
            objective=replace(
                original.bound.objective,
                economics_digest=economics,
                deployment_recipe_digest=content_digest(recipe),
            ),
        )
        envs.append(
            AllocationTradingEnv(
                dataset,
                stream=stream,
                estimates=tuple(
                    replace(e, buy_cost=0.025, sell_cost=0.025, exit_cost=0.0)
                    for e in original._estimates.values()
                ),
                bound=bound,
                action_contract=action,
                allocator=allocator,
                execution_cost=cost,
                risk_config=risk,
                feature_indices=(0,),
                symbol_index=0,
                start_index=6,
                stop_index=7,
                account_id=account,
                observation_schema=original.observation_schema,
            )
        )
    native = tuple(
        declare(global_api, e, folds, kind=k, scenario=scenario)
        for e, k in zip(envs, ("nonrl", "nonrl", "cash"), strict=True)
    )
    controls = forecast_controls(global_api, native)
    plans = tuple(
        terminal.declare_terminal_allocation_execution(
            folds, e, p, settlement_stop_index=8
        )
        for e, p in zip(envs, native, strict=True)
    )
    return folds, tuple(envs), controls, plans


def prepared(api, *, partial=False, accounts=("A",), scenarios=("base",)):
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    groups, records = [], []
    for account in accounts:
        for scenario in scenarios:
            folds, envs, controls, plans = group_fixture(
                account, scenario, partial=partial
            )
            groups.append(api.TerminalAllocationForecastControls(controls, *plans))
            records.extend(
                runner.run_terminal_allocation_execution(
                    folds, env, plan, forecast_controls=controls
                ).payload
                for env, plan in zip(envs, plans, strict=True)
            )
    return api.SettledForecastComparisonPlan(tuple(groups), accounts, scenarios), tuple(
        records
    )


def test_literal_forecast_trade_cost_harms_settled_profit():
    api = capability()
    plan, records = prepared(api)
    report = api.build_settled_forecast_comparison(plan, records).payload
    assert report["comparison_complete"]
    group = report["groups"][0]
    assert group["roles"]["control"]["final_equity"] == pytest.approx(94936 / 95)
    assert group["roles"]["treatment"]["final_equity"] == pytest.approx(998.4)
    assert group["roles"]["cash"]["final_equity"] == 1000
    assert group["treatment_minus_control"] == pytest.approx(-88 / 95000)
    assert group["treatment_minus_cash"] == pytest.approx(-0.0016)
    assert (
        report["claim_scope"] == "bounded_software_comparison_no_research_qualification"
    )


def test_full_account_scenario_product_is_order_independent():
    api = capability()
    plan, records = prepared(api, accounts=("A", "B"), scenarios=("base", "fee2"))
    assert len(records) == 12
    a = api.build_settled_forecast_comparison(plan, records)
    b = api.build_settled_forecast_comparison(plan, tuple(reversed(records)))
    assert a.payload == b.payload and a.digest == b.digest
    assert len(a.payload["groups"]) == 4


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "foreign", "swapped", "extra"]
)
def test_matrix_cannot_omit_duplicate_swap_or_add_roles(mutation):
    api = capability()
    plan, records = prepared(api)
    values = list(records)
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[1] = values[0]
    elif mutation == "foreign":
        values[1]["plan"]["native_plan"]["common"]["account_id"] = "foreign"
    elif mutation == "swapped":
        values[1]["plan"] = values[0]["plan"]
        values[1]["plan_digest"] = values[0]["plan_digest"]
    else:
        values.append(values[0])
    with pytest.raises(ValueError):
        api.build_settled_forecast_comparison(plan, tuple(values))


def test_partial_close_produces_no_qualified_delta():
    api = capability()
    plan, records = prepared(api, partial=True)
    report = api.build_settled_forecast_comparison(plan, records).payload
    assert not report["comparison_complete"]
    assert report["groups"][0]["treatment_minus_control"] is None
    assert report["groups"][0]["treatment_minus_cash"] is None


def test_declared_matrix_requires_every_account_scenario_group():
    api = capability()
    folds, envs, controls, plans = group_fixture()
    group = api.TerminalAllocationForecastControls(controls, *plans)
    with pytest.raises(ValueError):
        api.SettledForecastComparisonPlan((group,), ("A", "B"), ("base",))
    with pytest.raises(ValueError):
        api.SettledForecastComparisonPlan((group, group), ("A",), ("base",))
