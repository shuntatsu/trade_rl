"""No-fit native/comparison controls; validity content is bounded software only."""

import math
from dataclasses import replace
from fractions import Fraction
from importlib import import_module, util

import pytest

from tests.evaluation.test_allocation_global_execution import declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationComparisonContract,
    AllocationComparisonScenario,
    AllocationValidity,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_comparison_scenario_digest,
    continuous_account_profit_and_drawdown,
)
from trade_rl.evaluation.allocation_global_execution import (
    ObservedGlobalAllocationExecution,
)
from trade_rl.evaluation.robustness.walk_forward.stitching import StitchMode
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

SCOPE = "bounded_software_g2"


def capability():
    name = "trade_rl.evaluation.allocation_global_comparison_evidence"
    assert util.find_spec(name) is not None, (
        "completed native evidence adapter is missing"
    )
    return import_module(name)


def contract_for_plan(plan, *, direct_pin=None, seeds=(0, 7)):
    common, cell = plan.payload["common"], plan.payload["cell"]
    scenario = AllocationComparisonScenario(
        common["scenario"],
        allocation_comparison_scenario_digest(
            name=common["scenario"],
            dataset_id=common["dataset_id"],
            forecast_context_digest=common["forecast_context_digest"],
            economics_digest=common["economics_digest"],
            risk_digest=common["risk_digest"],
        ),
        True,
    )
    return AllocationComparisonContract(
        dataset_id=common["dataset_id"],
        objective_digest=common["objective_digest"],
        clock_digest=common["clock_digest"],
        forecast_context_digest=common["forecast_context_digest"],
        economics_digest=common["economics_digest"],
        risk_digest=common["risk_digest"],
        fold_plan_digest=common["fold_plan_digest"],
        nonrl_recipe_digest=cell["original_candidate_digest"],
        residual_recipe_digest=cell["original_candidate_digest"],
        direct_recipe_digest=direct_pin or cell["original_candidate_digest"],
        account_mode="independent_symbol",
        initial_capital=common["initial_capital"],
        scenarios=(scenario,),
        rl_seeds=seeds,
        maximum_drawdown=0.2,
    )


def validity_payload(contract, plan, observed, *, validity="valid", scope=SCOPE):
    """Test-side declaration, never an authenticated review or production approval."""
    common, cell = plan.payload["common"], plan.payload["cell"]
    return {
        "schema": "allocation_global_validity_record_v1",
        "validity": validity,
        "reasons": [] if validity == "valid" else ["declared software counterexample"],
        "assurance_scope": scope,
        "contract_digest": contract.digest,
        "cell": {
            "kind": cell["kind"],
            "seed": cell["seed"],
            "scenario": common["scenario"],
        },
        "common_digest": plan.common_digest,
        "cell_plan_digest": plan.cell_plan_digest,
        "receipt_digest": observed.receipt.digest,
        "implementation_digest": common["implementation_digest"],
        "runtime_environment_digest": common["runtime_environment_digest"],
        "review_evidence": {
            "record_id": "declared-g2-software-design",
            "digest": "a" * 64,
        },
        "native_accounting_evidence": {
            "record_id": "literal-fraction-g2-oracle",
            "digest": "b" * 64,
        },
    }


def build(
    api,
    contract,
    plan,
    observed,
    *,
    validity="valid",
    scope=SCOPE,
    expected_scope=SCOPE,
):
    record = api.GlobalAllocationValidityRecord.from_payload(
        validity_payload(contract, plan, observed, validity=validity, scope=scope)
    )
    builder = (
        api.build_global_allocation_cash_control_evidence
        if plan.payload["cell"]["kind"] == "cash"
        else api.build_global_allocation_comparison_evidence
    )
    return builder(
        contract,
        observed,
        expected_plan=plan,
        validity_record=record,
        expected_assurance_scope=expected_scope,
    )


def completed(*, kind="nonrl", boundary=7, cash_rate=None):
    from trade_rl.evaluation import allocation_global_execution as execution

    env = global_control_fixture(
        mode="direct" if kind == "cash" else "residual", cash_rate=cash_rate
    )
    folds = global_control_folds(boundary)
    plan = declare(execution, env, folds, kind=kind)
    observed = execution.run_declared_global_allocation_execution(folds, env, plan)
    return env, folds, plan, observed


@pytest.mark.parametrize("boundary", [7, 9])
def test_literal_native_equity_and_cumulative_drawdown_are_primary(boundary):
    env, _, plan, observed = completed(boundary=boundary)
    # Connected RED occurs only after original four native fills actually happened.
    assert env.book.fill_count == 4
    api = capability()
    row = build(api, contract_for_plan(plan), plan, observed)
    assert row.terminal_profit_rate == pytest.approx(
        float(Fraction(252971, 5500000)), abs=1e-12
    )
    assert row.max_drawdown == pytest.approx(float(Fraction(16529, 5769500)), abs=1e-12)
    facts = observed.receipt.payload["rows"]
    assert facts[0]["native"]["book"]["cash"] == 499
    assert facts[0]["native"]["book"]["equity"] == 1049
    assert facts[-1]["native"]["book"]["equity"] == pytest.approx(
        float(Fraction(5752971, 5500))
    )
    assert facts[-1]["native"]["book"]["total_cost"] == pytest.approx(
        float(Fraction(22029, 5500))
    )
    assert row.seed is row.policy_digest is None
    assert row.validity is AllocationValidity.VALID and row.coverage_complete
    assert row.opening_state_digest == observed.receipt.payload["opening_state_digest"]
    assert row.closing_state_digest == observed.receipt.payload["closing_state_digest"]


@pytest.mark.parametrize(
    "cash_rate,endpoint",
    [(0, 1000), (0.876, float(Fraction(10004000600040001, 10000000000000)))],
)
def test_cash_uses_actual_interest_and_common_identity(cash_rate, endpoint):
    _, _, nonrl_plan, nonrl_observed = completed(cash_rate=cash_rate)
    env, _, plan, observed = completed(kind="cash", cash_rate=cash_rate)
    assert nonrl_plan.common_digest == plan.common_digest
    api = capability()
    contract = contract_for_plan(
        nonrl_plan, direct_pin=plan.payload["cell"]["original_candidate_digest"]
    )
    cash, nonrl = (
        build(api, contract, plan, observed),
        build(api, contract, nonrl_plan, nonrl_observed),
    )
    assert cash.oos_source_digest == nonrl.oos_source_digest
    assert cash.terminal_profit_rate == pytest.approx(endpoint / 1000 - 1, abs=1e-12)
    assert cash.max_drawdown == 0 and env.book.fill_count == 0


def test_intra_event_dd_is_not_replaced_by_bar_return_dd():
    from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
    from trade_rl.evaluation import allocation_global_execution as execution

    folds, (env,) = fixture(ranges=((6, 7),), signals=(1, 0, 0, 0))
    plan = declare(execution, env, folds)
    observed = execution.run_declared_global_allocation_execution(folds, env, plan)
    api = capability()
    row = build(api, contract_for_plan(plan), plan, observed)
    _, bar_dd = continuous_account_profit_and_drawdown(
        observed.native_result.walk_forward.stitched.returns
    )
    assert bar_dd == 0 and row.max_drawdown == pytest.approx(0.001)
    assert row.terminal_profit_rate == pytest.approx(0.049)


@pytest.mark.parametrize(
    "change",
    [
        "values",
        "kind",
        "annualization",
        "indices",
        "boundaries",
        "gaps",
        "mode",
        "diagnostics",
    ],
)
def test_stitched_only_tamper_rejects_even_if_closed_folds_and_endpoint_match(change):
    _, _, plan, observed = completed()
    result = observed.native_result.walk_forward
    stitched = result.stitched
    variants = {
        "values": {
            "returns": ReturnSeries(
                (0.0, 0.0, 0.0, math.prod(1 + r for r in stitched.returns.values) - 1),
                ReturnKind.DECISION_STEP,
                stitched.returns.periods_per_year,
            )
        },
        "kind": {"returns": replace(stitched.returns, kind=ReturnKind.BASE_BAR)},
        "annualization": {"returns": replace(stitched.returns, periods_per_year=123)},
        "indices": {"fold_indices": (99, 100)},
        "boundaries": {"boundaries": ((6, 8), (8, 10))},
        "gaps": {"gaps": ((7, 8),)},
        "mode": {"mode": StitchMode.INDEPENDENT_FOLDS},
        "diagnostics": {
            "diagnostics": replace(
                stitched.diagnostics, total_cost=stitched.diagnostics.total_cost + 1
            )
        },
    }
    api = capability()
    with pytest.raises(ValueError, match="stitch"):
        changed = replace(
            observed.native_result,
            walk_forward=replace(
                result, stitched=replace(stitched, **variants[change])
            ),
        )
        altered = ObservedGlobalAllocationExecution(changed, observed.receipt)
        build(api, contract_for_plan(plan), plan, altered)


def test_explicit_invalid_and_required_scope_are_preserved():
    _, _, plan, observed = completed()
    api = capability()
    contract = contract_for_plan(plan)
    row = build(api, contract, plan, observed, validity="invalid")
    assert row.validity is AllocationValidity.INVALID
    with pytest.raises(ValueError, match="scope"):
        build(api, contract, plan, observed, scope="independent_g3")
    with pytest.raises(ValueError, match="scope"):
        build(api, contract, plan, observed, expected_scope="anything")


@pytest.mark.parametrize(
    "field",
    [
        "contract_digest",
        "common_digest",
        "cell_plan_digest",
        "receipt_digest",
        "implementation_digest",
        "runtime_environment_digest",
    ],
)
def test_stale_validity_binding_rejects(field):
    _, _, plan, observed = completed()
    api, contract = capability(), contract_for_plan(plan)
    payload = validity_payload(contract, plan, observed)
    payload[field] = "f" * 64
    record = api.GlobalAllocationValidityRecord.from_payload(payload)
    with pytest.raises(ValueError, match="validity"):
        api.build_global_allocation_comparison_evidence(
            contract,
            observed,
            expected_plan=plan,
            validity_record=record,
            expected_assurance_scope=SCOPE,
        )


def test_validity_is_closed_canonical_and_not_a_digest_or_default():
    _, _, plan, observed = completed()
    api, contract = capability(), contract_for_plan(plan)
    payload = validity_payload(contract, plan, observed)
    record = api.GlobalAllocationValidityRecord.from_payload(payload)
    detached = record.payload
    detached["reasons"].append("mutation")
    assert record.payload == payload
    for changed in (
        payload | {"approved": True},
        payload | {"validity": None},
        payload | {"assurance_scope": "unscoped"},
        payload | {"validity": "invalid", "reasons": []},
        payload | {"cell": payload["cell"] | {"seed": True}},
    ):
        with pytest.raises(ValueError):
            api.GlobalAllocationValidityRecord.from_payload(changed)
    with pytest.raises(ValueError):
        api.GlobalAllocationValidityRecord(b" " + canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        api.build_global_allocation_comparison_evidence(
            contract,
            observed,
            expected_plan=plan,
            validity_record="a" * 64,
            expected_assurance_scope=SCOPE,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_mode", "shared_portfolio"),
        ("dataset_id", "f" * 64),
        ("objective_digest", "f" * 64),
        ("clock_digest", "f" * 64),
        ("forecast_context_digest", "f" * 64),
        ("initial_capital", 2000),
        ("nonrl_recipe_digest", "f" * 64),
        ("economics_digest", "f" * 64),
    ],
)
def test_comparison_contract_binds_native_global_context(field, value):
    _, _, plan, observed = completed()
    api = capability()
    contract = replace(contract_for_plan(plan), **{field: value})
    with pytest.raises(ValueError, match="contract|context|candidate|scenario"):
        build(api, contract, plan, observed)


def repinned_observed(observed, mutate):
    """Content-consistency attack fixture, never an alleged authentic execution."""
    from trade_rl.evaluation.allocation_global_execution import (
        GlobalAllocationExecutionReceipt,
    )
    from trade_rl.evaluation.robustness.walk_forward.stitching import stitch_oos

    raw = observed.receipt.payload
    mutate(raw)
    original = observed.native_result.walk_forward
    folds = []
    for segment, source in zip(raw["segments"], original.folds, strict=True):
        segment["returns"] = [
            r["native"]["execution"]["interval_net_return"]
            for r in raw["rows"]
            if r["fold_index"] == segment["fold_index"]
        ]
        diagnostics = replace(
            source.diagnostics,
            **{
                k: v
                for k, v in segment["diagnostics"].items()
                if k not in {"closed_trades", "termination_reasons"}
            },
        )
        folds.append(
            replace(
                source,
                returns=replace(source.returns, values=tuple(segment["returns"])),
                diagnostics=diagnostics,
            )
        )
    folds = tuple(folds)
    result = replace(
        original,
        folds=folds,
        stitched=stitch_oos(folds, mode=StitchMode.CONTINUOUS_ACCOUNT),
    )
    native = replace(observed.native_result, walk_forward=result)
    return ObservedGlobalAllocationExecution(
        native, GlobalAllocationExecutionReceipt.from_payload(raw)
    )


@pytest.mark.parametrize(
    "field",
    [
        "interval_net_return",
        "interval_log_return",
        "interval_cost",
        "interval_funding",
        "interval_borrow_cost",
        "filled_turnover",
        "fill_count",
        "rebalance_events",
    ],
)
def test_interval_accounting_facts_are_bound_to_native_cumulative_book(field):
    _, _, plan, observed = completed()

    def mutate(raw):
        raw["rows"][1]["native"]["execution"][field] += 1

    changed = repinned_observed(observed, mutate)
    api = capability()
    with pytest.raises(ValueError, match="native|interval"):
        build(api, contract_for_plan(plan), plan, changed)


@pytest.mark.parametrize(
    "counter",
    [
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "borrow_cost",
        "n_trades",
        "rebalance_events",
    ],
)
def test_well_typed_fold_counter_edit_cannot_launder_native_totals(counter):
    _, _, plan, observed = completed()

    def mutate(raw):
        raw["segments"][0]["diagnostics"][counter] += 1

    changed = repinned_observed(observed, mutate)
    api = capability()
    with pytest.raises(ValueError, match="segment"):
        build(api, contract_for_plan(plan), plan, changed)


def test_compensating_interval_changes_with_same_terminal_product_reject():
    _, _, plan, observed = completed()
    before = observed.native_result.walk_forward.stitched.returns.values

    def mutate(raw):
        first, second = raw["rows"][1:3]
        factor = (1 + before[1]) * (1 + before[2])
        first["native"]["execution"].update(
            interval_net_return=0.0, interval_log_return=0.0
        )
        second["native"]["execution"].update(
            interval_net_return=factor - 1, interval_log_return=math.log(factor)
        )

    changed = repinned_observed(observed, mutate)
    assert math.prod(
        1 + r for r in changed.native_result.walk_forward.stitched.returns.values
    ) == pytest.approx(math.prod(1 + r for r in before))
    api = capability()
    with pytest.raises(ValueError, match="interval net return"):
        build(api, contract_for_plan(plan), plan, changed)


@pytest.mark.parametrize(
    "field", ["peak_value", "max_drawdown", "equity", "exact_quantities"]
)
def test_native_book_tamper_rejects_without_scoring(field):
    _, _, plan, observed = completed()

    def mutate(raw):
        book = raw["rows"][1]["native"]["book"]
        book[field] = ["1"] if field == "exact_quantities" else 0

    api = capability()
    with pytest.raises(ValueError):
        changed = repinned_observed(observed, mutate)
        build(api, contract_for_plan(plan), plan, changed)


def test_whole_expected_plan_and_exact_observed_types_are_required():
    from types import SimpleNamespace

    from trade_rl.evaluation.allocation_global_execution import (
        GlobalAllocationExecutionPlan,
    )

    _, _, plan, observed = completed()
    api, contract = capability(), contract_for_plan(plan)
    changed = plan.payload
    changed["common"]["account_id"] = "other-account"
    wrong = GlobalAllocationExecutionPlan.from_payload(changed)
    record = api.GlobalAllocationValidityRecord.from_payload(
        validity_payload(contract, plan, observed)
    )
    for candidate, expected in (
        (observed, wrong),
        (
            SimpleNamespace(
                native_result=observed.native_result, receipt=observed.receipt
            ),
            plan,
        ),
        (observed.native_result.walk_forward, plan),
        (observed.receipt, plan),
    ):
        with pytest.raises(ValueError, match="expected plan|exact completed"):
            api.build_global_allocation_comparison_evidence(
                contract,
                candidate,
                expected_plan=expected,
                validity_record=record,
                expected_assurance_scope=SCOPE,
            )


def test_signed_economic_prefix_is_retained_but_never_becomes_completed_evidence():
    from trade_rl.evaluation import allocation_global_execution as execution

    env, folds = (
        global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True),
        global_control_folds(7),
    )
    plan = declare(execution, env, folds)
    with pytest.raises(ValueError, match="economic termination") as failure:
        execution.run_declared_global_allocation_execution(folds, env, plan)
    prefix = failure.value.global_execution_receipt
    assert len(prefix.payload["rows"]) == 1
    assert prefix.payload["rows"][0]["native"]["book"]["equity"] == -501
    assert prefix.payload["rows"][0]["native"]["book"]["exact_quantities"] == ["0"]
    api = capability()
    with pytest.raises(ValueError, match="exact completed"):
        api.build_global_allocation_comparison_evidence(
            contract_for_plan(plan),
            prefix,
            expected_plan=plan,
            validity_record=object(),
            expected_assurance_scope=SCOPE,
        )


def test_declared_common_drift_changes_global_oos_while_legacy_projection_is_fixed():
    from trade_rl.evaluation import allocation_global_execution as execution
    from trade_rl.evaluation.allocation_comparison_evidence import (
        allocation_comparison_oos_source_digest,
    )

    _, _, plan, observed = completed()
    env = global_control_fixture()
    env.account_id = "changed-account-with-same-declared-contract"
    folds = global_control_folds(7)
    other = declare(execution, env, folds)
    changed = execution.run_declared_global_allocation_execution(folds, env, other)
    contract = contract_for_plan(plan)
    assert allocation_comparison_oos_source_digest(
        contract, scenario="base"
    ) == allocation_comparison_oos_source_digest(
        contract_for_plan(other), scenario="base"
    )
    api = capability()
    left, right = (
        build(api, contract, plan, observed),
        build(api, contract, other, changed),
    )
    assert left.oos_source_digest != right.oos_source_digest


def test_candidate_mode_changes_cell_but_not_raw_common_context():
    from trade_rl.evaluation import allocation_global_execution as execution

    residual, direct = global_control_fixture(), global_control_fixture(mode="direct")
    folds = global_control_folds(7)
    left, right = (
        declare(execution, residual, folds),
        declare(execution, direct, folds, kind="cash"),
    )
    assert left.common_digest == right.common_digest
    assert left.cell_plan_digest != right.cell_plan_digest
    observed = execution.run_declared_global_allocation_execution(folds, direct, right)
    api = capability()
    with pytest.raises(ValueError, match="candidate"):
        build(api, contract_for_plan(left), right, observed)


def fake_global_model(env, seed, path, monkeypatch):
    """No fitting or SB3. Existing fake manifest/model/native guards still run."""
    from tests.evaluation.allocation_fee_stress_fixture import (
        FakeModel,
        declared_manifest,
    )
    from trade_rl.strategies.rl import allocation_artifact as archive
    from trade_rl.strategies.rl import allocation_model as owner
    from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy

    monkeypatch.setattr(owner, "validate_allocation_protocol_model", lambda *_: None)
    raw = declared_manifest(env)
    raw["training"]["seed"] = seed
    initial = FakeModel(raw)
    initial.seed = seed
    digest = archive.save_allocation_policy(path, AllocationPPOPolicy(initial, raw))

    fake_global_model_loader(env, raw, monkeypatch)
    return path, digest


def clone_carrier(env, *, dataset=None, cost=None, allocator=None):
    from copy import deepcopy
    from dataclasses import asdict

    from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
    from trade_rl.simulation import MarketExecutor

    dataset = env.dataset if dataset is None else dataset
    cost = env.execution_cost if cost is None else cost
    allocator = env.allocator if allocator is None else allocator
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    recipe = deepcopy(env.recipe)
    recipe["runtime_profile"]["economics_digest"] = economics
    recipe["runtime_profile"]["calendar_kind"] = dataset.calendar_kind.value
    recipe["runtime_profile"]["execution_bar_hours"] = dataset.bar_hours
    recipe["allocator"] = asdict(allocator)
    bound = replace(
        env.bound,
        objective=replace(
            env.bound.objective,
            economics_digest=economics,
            deployment_recipe_digest=content_digest(recipe),
        ),
    )
    return AllocationTradingEnv(
        dataset,
        stream=env.stream,
        estimates=tuple(env._estimates.values()),
        bound=bound,
        action_contract=env.action_contract,
        allocator=allocator,
        execution_cost=cost,
        risk_config=env.risk_config,
        feature_indices=env.feature_indices,
        symbol_index=env.symbol_index,
        start_index=env.start_index,
        stop_index=env.stop_index,
        account_id=env.account_id,
        observation_schema=env.observation_schema,
        feature_preprocessing=env.feature_preprocessing,
    )


def full_matrix(tmp_path, monkeypatch):
    from trade_rl.evaluation import allocation_global_execution as execution
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
    )

    folds = global_control_folds(7)
    residual, direct = global_control_fixture(), global_control_fixture(mode="direct")
    residual_plan = declare(execution, residual, folds)
    direct_plan = declare(execution, direct, folds, kind="cash")
    contract = contract_for_plan(
        residual_plan,
        direct_pin=direct_plan.payload["cell"]["original_candidate_digest"],
    )
    stress_carrier = global_control_fixture(fee=0.004)
    stress_plan = declare(execution, stress_carrier, folds, scenario="fee_x2")
    common = stress_plan.payload["common"]
    contract = replace(
        contract,
        scenarios=contract.scenarios
        + (
            AllocationComparisonScenario(
                "fee_x2",
                allocation_comparison_scenario_digest(
                    name="fee_x2",
                    dataset_id=common["dataset_id"],
                    forecast_context_digest=common["forecast_context_digest"],
                    economics_digest=common["economics_digest"],
                    risk_digest=common["risk_digest"],
                ),
                True,
            ),
        ),
    )
    api = capability()
    rows, cash = [], []
    observations = []
    for scenario, fee in (("base", 0.002), ("fee_x2", 0.004)):
        for kind in ("nonrl", "cash", "residual_ppo", "direct_ppo"):
            for seed in (None,) if kind in {"nonrl", "cash"} else (0, 7):
                env = global_control_fixture(
                    mode="residual" if kind in {"nonrl", "residual_ppo"} else "direct",
                    fee=fee,
                )
                extra = {}
                if seed is not None:
                    base = global_control_fixture(mode=env.action_contract.mode)
                    if scenario != "base":
                        env = clone_carrier(
                            base, cost=replace(base.execution_cost, fee_rate=fee)
                        )
                    path = tmp_path / f"{kind}-{seed}"
                    if scenario == "base":
                        root, digest = fake_global_model(base, seed, path, monkeypatch)
                    else:
                        import json

                        root = path
                        digest = content_digest(
                            json.loads((path / "manifest.json").read_text("utf-8"))
                        )
                        # Rebind the same no-fit loader; no new model publication.
                        fake_global_model_loader(
                            base,
                            json.loads((path / "manifest.json").read_text("utf-8")),
                            monkeypatch,
                        )
                    artifacts = tuple(
                        AllocationFoldPolicyArtifact(
                            f.fold_index, root, digest, base.recipe_digest
                        )
                        for f in folds
                    )
                    extra = {"seed": seed, "reset_seed": 17, "artifacts": artifacts}
                    if scenario != "base":
                        extra.update(
                            base_env=base,
                            contract=contract,
                            fee_factor=2.0,
                            reset_seed=seed,
                        )
                plan = declare(
                    execution, env, folds, kind=kind, scenario=scenario, **extra
                )
                observed = execution.run_declared_global_allocation_execution(
                    folds,
                    env,
                    plan,
                    **({"artifacts": extra["artifacts"]} if seed is not None else {}),
                    **(
                        {"base_env": extra["base_env"], "contract": contract}
                        if "base_env" in extra
                        else {}
                    ),
                )
                row = build(api, contract, plan, observed)
                (cash if kind == "cash" else rows).append(row)
                observations.append((plan, observed))
    return contract, tuple(rows), tuple(cash), tuple(observations)


def fake_global_model_loader(env, manifest, monkeypatch):
    from tests.evaluation.allocation_fee_stress_fixture import FakeModel
    from trade_rl.strategies.rl import allocation_artifact as archive

    def load(_verified_path):
        model = FakeModel(
            manifest,
            actions=(2, 2, 2, 2)
            if env.action_contract.mode == "residual"
            else (3, 0, 0, 0),
        )
        model.seed = manifest["training"]["seed"]
        return model

    monkeypatch.setattr(archive, "_load_policy", load)


def test_complete_no_fit_matrix_preserves_seed_scenario_cash_and_invalid_rules(
    tmp_path, monkeypatch
):
    from trade_rl.evaluation.allocation_cash_control import allocation_cash_reference
    from trade_rl.evaluation.allocation_comparison import (
        validate_allocation_comparison_evidence,
    )
    from trade_rl.evaluation.allocation_selection import (
        AllocationSelectionOutcome,
        AllocationSelectionRule,
        select_allocation_candidate,
    )

    contract, rows, cash, observations = full_matrix(tmp_path, monkeypatch)
    assert len(rows) == 10 and len(cash) == 2
    assert len(validate_allocation_comparison_evidence(contract, rows)) == 10
    references = tuple(allocation_cash_reference(row) for row in cash)
    rule = AllocationSelectionRule(contract.digest, 1000, 1000)
    assert (
        select_allocation_candidate(contract, rows, references, rule).outcome
        is AllocationSelectionOutcome.NO_WINNER
    )
    first = next(
        (plan, observed)
        for plan, observed in observations
        if plan.payload["cell"]["kind"] == "nonrl"
        and plan.payload["common"]["scenario"] == "base"
    )
    invalid = build(capability(), contract, *first, validity="invalid")
    replaced = tuple(
        invalid if row.candidate.value == "nonrl" and row.scenario == "base" else row
        for row in rows
    )
    assert (
        select_allocation_candidate(contract, replaced, references, rule).outcome
        is AllocationSelectionOutcome.INVALID
    )
    for altered in (
        rows[:-1],
        rows + (rows[0],),
        (replace(rows[0], scenario="unregistered"),) + rows[1:],
        (replace(rows[2], seed=99),) + rows[:2] + rows[3:],
        tuple(
            replace(row, recipe_digest="f" * 64)
            if row.candidate.value == "direct_ppo" and row.seed == 7
            else row
            for row in rows
        ),
        (replace(rows[0], oos_source_digest="f" * 64),) + rows[1:],
        (replace(rows[0], opening_state_digest="f" * 64),) + rows[1:],
    ):
        with pytest.raises(ValueError):
            validate_allocation_comparison_evidence(contract, altered)
    changed = tuple(
        replace(row, policy_digest="f" * 64)
        if row.candidate.value == "direct_ppo"
        and row.seed == 7
        and row.scenario == "fee_x2"
        else row
        for row in rows
    )
    with pytest.raises(ValueError, match="one policy"):
        validate_allocation_comparison_evidence(contract, changed)
    with pytest.raises(ValueError, match="cash"):
        select_allocation_candidate(contract, rows, references[:-1], rule)
    with pytest.raises(ValueError, match="cash"):
        select_allocation_candidate(contract, rows, references + (references[0],), rule)
    with pytest.raises(ValueError, match="source"):
        select_allocation_candidate(
            contract,
            rows,
            (replace(references[0], oos_source_digest="f" * 64), references[1]),
            rule,
        )
    ordinary = next(
        (plan, observed)
        for plan, observed in observations
        if plan.payload["cell"]["kind"] == "direct_ppo"
        and plan.payload["cell"]["seed"] == 7
        and plan.payload["common"]["scenario"] == "base"
    )
    assert ordinary[0].payload["cell"]["reset_seed"] == 17
    assert build(capability(), contract, *ordinary).seed == 7


def test_session_annualization_preserves_fold_semantics_without_inverse_clock_assumption():
    from trade_rl.data.contracts import MarketCalendarKind
    from trade_rl.evaluation import allocation_global_execution as execution

    original = global_control_fixture()
    dataset = replace(
        original.dataset, calendar_kind=MarketCalendarKind.SESSION, periods_per_year=252
    )
    env = clone_carrier(original, dataset=dataset)
    folds = global_control_folds(7)
    plan = declare(execution, env, folds)
    observed = execution.run_declared_global_allocation_execution(folds, env, plan)
    row = build(capability(), contract_for_plan(plan), plan, observed)
    assert observed.native_result.walk_forward.stitched.returns.periods_per_year == 252
    assert row.terminal_profit_rate == pytest.approx(float(Fraction(252971, 5500000)))


@pytest.mark.parametrize(
    "counter",
    [
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "borrow_cost",
        "fill_count",
        "rebalance_events",
    ],
)
def test_native_cumulative_counters_cannot_be_changed_independently_of_interval(
    counter,
):
    _, _, plan, observed = completed()

    def mutate(raw):
        raw["rows"][2]["native"]["book"][counter] += 1

    changed = repinned_observed(observed, mutate)
    with pytest.raises(ValueError, match="interval"):
        build(capability(), contract_for_plan(plan), plan, changed)


@pytest.mark.parametrize("rate", [0.001, -0.001])
def test_signed_funding_native_case_is_not_treated_as_unsigned_or_monotone(rate):
    import numpy as np

    from trade_rl.evaluation import allocation_global_execution as execution

    original = global_control_fixture(signals=(1, 1, 1, 1))
    # Settlement mark100 differs from valuation mark110 at the first boundary.
    dataset = replace(
        original.dataset,
        funding_rate=np.full((12, 1), rate),
        funding_price_rate=np.full((12, 1), 100 * rate),
    )
    env = clone_carrier(original, dataset=dataset)
    folds = global_control_folds(7)
    plan = declare(execution, env, folds)
    observed = execution.run_declared_global_allocation_execution(folds, env, plan)
    # Actual native funding, including its sign; no fabricated funding counter.
    funding = [
        row["native"]["book"]["funding_pnl"] for row in observed.receipt.payload["rows"]
    ]
    assert funding[-1] * rate < 0
    first_book = observed.receipt.payload["rows"][0]["native"]["book"]
    assert first_book["quantities"] == [5.0]
    assert funding[0] == pytest.approx(-500 * rate)
    assert first_book["cash"] == pytest.approx(499 - 500 * rate)
    assert first_book["equity"] == pytest.approx(1049 - 500 * rate)
    row = build(capability(), contract_for_plan(plan), plan, observed)
    assert row.terminal_profit_rate == pytest.approx(
        env.book.portfolio_value / 1000 - 1
    )


@pytest.mark.parametrize("reference", ["review_evidence", "native_accounting_evidence"])
def test_validity_requires_declared_separate_record_references(reference):
    _, _, plan, observed = completed()
    api, contract = capability(), contract_for_plan(plan)
    raw = validity_payload(contract, plan, observed)
    for value in (
        {"digest": "a" * 64},
        {"record_id": "", "digest": "a" * 64},
        {"record_id": "record", "digest": "bad"},
        "a" * 64,
    ):
        with pytest.raises(ValueError):
            api.GlobalAllocationValidityRecord.from_payload(raw | {reference: value})


def test_validity_record_binds_exact_cell_key_and_retains_declared_content():
    _, _, plan, observed = completed()
    api, contract = capability(), contract_for_plan(plan)
    raw = validity_payload(contract, plan, observed)
    for cell in (raw["cell"] | {"kind": "cash"}, raw["cell"] | {"scenario": "other"}):
        record = api.GlobalAllocationValidityRecord.from_payload(raw | {"cell": cell})
        with pytest.raises(ValueError, match="validity"):
            api.build_global_allocation_comparison_evidence(
                contract,
                observed,
                expected_plan=plan,
                validity_record=record,
                expected_assurance_scope=SCOPE,
            )
    declared = api.GlobalAllocationValidityRecord.from_payload(
        raw | {"assurance_scope": "independent_g3"}
    )
    assert (
        build(
            api,
            contract,
            plan,
            observed,
            scope="independent_g3",
            expected_scope="independent_g3",
        ).validity_evidence_digest
        == declared.digest
    )
    # Acceptance authenticates neither these test reference IDs nor a reviewer.
    assert (
        declared.payload["review_evidence"]["record_id"]
        == "declared-g2-software-design"
    )


def test_unsigned_counter_decrease_rejects_even_when_all_declared_deltas_agree():
    _, _, plan, observed = completed()

    def mutate(raw):
        # All content remains well typed and self-consistent, but native total
        # trading costs cannot decrease. Signed funding has a different rule.
        raw["rows"][1]["native"]["book"]["total_cost"] = 0.5
        raw["rows"][1]["native"]["execution"]["interval_cost"] = -0.5
        raw["rows"][2]["native"]["execution"]["interval_cost"] = (
            raw["rows"][2]["native"]["book"]["total_cost"] - 0.5
        )
        raw["segments"][1]["diagnostics"]["total_cost"] = (
            raw["rows"][-1]["native"]["book"]["total_cost"]
            - raw["rows"][0]["native"]["book"]["total_cost"]
        )

    changed = repinned_observed(observed, mutate)
    # The native fold's scalar totals and interval deltas are all consistent.
    with pytest.raises(ValueError, match="unsigned|counter"):
        build(capability(), contract_for_plan(plan), plan, changed)


def test_declared_drawdown_guardrail_must_match_native_objective_risk_binding():
    _, _, plan, observed = completed()
    contract = replace(contract_for_plan(plan), maximum_drawdown=0.1)
    with pytest.raises(ValueError, match="drawdown"):
        build(capability(), contract, plan, observed)


def repin_context_declaration(
    native, *, equity_delta=0.0, weight_delta=0.0, dd_delta=0.0
):
    """Rebuild well-typed claim content to reach the new cross-interval guard."""
    import numpy as np

    from trade_rl.evaluation.rl_allocation.transition_facts import _json
    from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
    from trade_rl.strategies.allocation import (
        AfterCostTargetAllocator,
        AllocationContext,
        AllocationInputs,
    )
    from trade_rl.strategies.allocation_action import (
        AllocationActionContract,
        AllocationDecision,
    )

    declared = native["proposal"]["decision"]
    baseline = declared["baseline"]
    inputs = AllocationInputs(
        **{
            key: np.datetime64(value, "ns")
            if key in {"decision_time", "available_at", "horizon_end"}
            else value
            for key, value in baseline["inputs"].items()
        }
    )
    context = dict(baseline["context"])
    context["equity"] += equity_delta
    context["current_weight"] += weight_delta
    context["decision_time"] = np.datetime64(context["decision_time"], "ns")
    context = AllocationContext(**context)
    allocator = AfterCostTargetAllocator(**baseline["allocator"])
    decision = AllocationDecision(
        baseline=allocator.propose(inputs, context),
        action_contract=AllocationActionContract(**declared["action_contract"]),
        feature_names=tuple(declared["feature_names"]),
        feature_values=tuple(declared["feature_values"]),
        max_drawdown=declared["max_drawdown"] + dd_delta,
        initial_capital=declared["initial_capital"],
        remaining_steps=declared["remaining_steps"],
        pending_gross=declared["pending_gross"],
        pending_count=declared["pending_count"],
    )
    proposal = decision.propose(native["proposal"]["raw_action"])
    native["proposal"], native["proposal_digest"], native["decision_digest"] = (
        _json(proposal),
        proposal.digest,
        decision.decision_digest,
    )
    native["current_weights"][native["symbol_index"]] = context.current_weight
    target = np.zeros(len(native["current_weights"]))
    target[native["symbol_index"]] = proposal.target_weight
    native["risk"] = _json(
        PreTradeRisk(PreTradeRiskConfig(**native["risk_config"])).constrain(
            target,
            current=np.asarray(native["current_weights"]),
            drawdown=decision.max_drawdown,
        )
    )


@pytest.mark.parametrize(
    "offset,change", [(0, "equity"), (1, "equity"), (1, "weight"), (1, "dd")]
)
def test_cross_interval_context_claim_cannot_bypass_prior_native_account(
    offset, change
):
    _, _, plan, observed = completed()

    def mutate(raw):
        repin_context_declaration(
            raw["rows"][offset]["native"],
            **{f"{change}_delta": 0.01 if change != "equity" else 1.0},
        )

    changed = repinned_observed(observed, mutate)
    # Existing exact receipt/Observed constructors have accepted the repinned
    # claim; the new C cross-interval account guard must still reject it.
    with pytest.raises(
        ValueError, match="opening interval equity|current weight|decision drawdown"
    ):
        build(capability(), contract_for_plan(plan), plan, changed)


def test_partial_native_fill_is_distinct_from_completed_order_and_requested_turnover():
    from trade_rl.evaluation import allocation_global_execution as execution

    env, folds = (
        global_control_fixture(volume=1, zero_later_volume=True, signals=(1, 1, 1, 1)),
        global_control_folds(7),
    )
    plan = declare(execution, env, folds)
    observed = execution.run_declared_global_allocation_execution(folds, env, plan)
    first = observed.receipt.payload["rows"][0]["native"]["execution"]
    assert first["fill_count"] == 1 and first["completed_fill_count"] == 0
    assert first["filled_turnover"] < first["requested_turnover"]
    row = build(capability(), contract_for_plan(plan), plan, observed)
    assert env.book.fill_count == 1
    assert row.terminal_profit_rate == pytest.approx(-0.0002)


def test_registered_original_candidate_cannot_hide_actual_runtime_recipe_drift():
    from trade_rl.evaluation import allocation_global_execution as execution
    from trade_rl.evaluation.allocation_global_execution import (
        GlobalAllocationExecutionPlan,
        GlobalAllocationExecutionReceipt,
    )

    base = global_control_fixture()
    folds = global_control_folds(7)
    original_plan = declare(execution, base, folds)
    allocator = replace(base.allocator, upper_weight=0.25)
    env = clone_carrier(base, allocator=allocator)
    variant_plan = declare(execution, env, folds)
    actual = execution.run_declared_global_allocation_execution(
        folds, env, variant_plan
    )
    assert original_plan.common_digest == variant_plan.common_digest
    raw = actual.receipt.payload
    original = original_plan.payload["cell"]
    raw["plan"]["cell"].update(
        {
            key: original[key]
            for key in (
                "original_recipe",
                "original_recipe_digest",
                "original_candidate_digest",
                "control_policy_digest",
            )
        }
    )
    expected = GlobalAllocationExecutionPlan.from_payload(raw["plan"])
    raw["cell_plan_digest"] = expected.cell_plan_digest
    for row in raw["rows"]:
        row["policy_digest"] = original["control_policy_digest"]
    result = replace(
        actual.native_result.walk_forward,
        policy_digests=(original["control_policy_digest"],) * len(folds),
    )
    native = replace(
        actual.native_result,
        candidate_recipe_digest=original["original_candidate_digest"],
        walk_forward=result,
    )
    changed = ObservedGlobalAllocationExecution(
        native, GlobalAllocationExecutionReceipt.from_payload(raw)
    )
    with pytest.raises(ValueError, match="candidate|runtime"):
        build(capability(), contract_for_plan(original_plan), expected, changed)
