"""Held small actual-SB3 software protocol, never a market-profit experiment."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_allocation_fee_stress import declaration
from tests.evaluation.test_allocation_preprocessing_runtime import (
    bind_frozen,
    frozen_args,
)
from tests.integrations.test_allocation_schedule_ppo_runtime import dated_args
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    run_artifact_bound_global_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.scheduled_training import (
    fit_allocation_ppo_schedule,
)
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
    allocation_training_window,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")


def persist_software_evidence(path, receipt, *, software_kind, native_facts=None):
    payload = {
        "schema": "allocation_fee_runtime_software_evidence_v1",
        "software_kind": software_kind,
        "learned_alpha_established": False,
        "receipt": receipt,
    }
    if native_facts is not None:
        payload["asserted_native_endpoint_facts"] = native_facts
    path.write_bytes(canonical_json_bytes(payload))


def persist_global_software_evidence(path, evidence, *, software_kind):
    path.parent.mkdir()
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "allocation_global_runtime_comparison_software_evidence_v1",
                "software_kind": software_kind,
                "learned_alpha_established": False,
                **evidence,
            }
        )
    )


def runtime_case(*, frozen):
    first = frozen_args(stop=10) if frozen else dated_args(1)
    first["bound"] = replace(
        first["bound"], clock=replace(first["bound"].clock, rollout_steps=4)
    )
    second = (
        bind_frozen(dated_args(2), first["feature_preprocessing"])
        if frozen
        else dated_args(2)
    )
    children = tuple(AllocationTradingEnv(**args) for args in (first, second))
    windows = tuple(allocation_training_window(child) for child in children)
    schedule = AllocationTrainingSchedule(windows, tuple(w.window_id for w in windows))
    fit = fit_allocation_ppo_schedule(
        AllocationTrainingScheduleEnv(schedule, children),
        total_timesteps=8,
        seed=7,
        training_protocol=protocol(n_steps=4, batch_size=4, n_epochs=1),
    )
    oos = (
        bind_frozen(dated_args(4), first["feature_preprocessing"])
        if frozen
        else dated_args(4)
    )
    base = AllocationTradingEnv(**oos)
    cost = replace(base.execution_cost, fee_rate=0.004)
    economics = MarketExecutor(
        base.dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    recipe = deepcopy(base.recipe)
    recipe["runtime_profile"]["economics_digest"] = economics
    stressed = oos | {
        "execution_cost": cost,
        "bound": replace(
            base.bound,
            objective=replace(
                base.bound.objective,
                economics_digest=economics,
                deployment_recipe_digest=content_digest(recipe),
            ),
        ),
    }
    return fit.inference_policy(), base, AllocationTradingEnv(**stressed)


def execute(base, stress, root, digest, *, evidence, second=None):
    folds, contract, artifacts = declaration(base, stress, root, digest)
    if second is not None:
        artifacts = (
            artifacts[0],
            replace(artifacts[1], bundle_root=second[0], expected_digest=second[1]),
        )
    from tests.evaluation.test_allocation_global_comparison_evidence import (
        SCOPE,
        validity_payload,
    )
    from trade_rl.evaluation import allocation_global_execution as execution
    from trade_rl.evaluation.allocation_global_comparison_evidence import (
        GlobalAllocationValidityRecord,
        build_global_allocation_comparison_evidence,
    )
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    provenance = build_candidate_run_provenance()
    expected_plan = execution.declare_global_allocation_execution(
        folds,
        stress,
        kind="direct_ppo",
        scenario="configured_fee_x2",
        seed=7,
        reset_seed=7,
        artifacts=artifacts,
        base_env=base,
        contract=contract,
        fee_factor=2.0,
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_digest=provenance["runtime_environment_digest"],
    )
    observed = execution.run_declared_global_allocation_execution(
        folds,
        stress,
        expected_plan,
        artifacts=artifacts,
        base_env=base,
        contract=contract,
    )
    result = observed.native_result
    validity_record = GlobalAllocationValidityRecord.from_payload(
        validity_payload(contract, expected_plan, observed)
    )
    row = build_global_allocation_comparison_evidence(
        contract,
        observed,
        expected_plan=expected_plan,
        validity_record=validity_record,
        expected_assurance_scope=SCOPE,
    )
    assert observed.receipt.payload["reset_count"] == 1
    assert len(observed.receipt.payload["rows"]) == 4
    assert row.seed == 7 and row.max_drawdown == stress.book.max_drawdown
    assert row.terminal_profit_rate == pytest.approx(
        stress.book.portfolio_value / stress.initial_capital - 1
    )
    evidence.update(
        expected_plan=expected_plan.payload,
        receipt=observed.receipt.payload,
        validity_record=validity_record.payload,
        comparison_evidence=row.payload(),
    )
    return result, folds, artifacts


@pytest.mark.parametrize("frozen", [False, True])
def test_actual_unmodified_scheduled_policy_fee_view_save_reload(tmp_path, frozen):
    policy, base, stress = runtime_case(frozen=frozen)
    root = tmp_path / "genuine-original"
    digest = save_allocation_policy(root, policy)
    archive_before = (root / "policy.zip").read_bytes()
    evidence = {}
    result, _, _ = execute(base, stress, root, digest, evidence=evidence)
    receipt = result.receipt
    assert (root / "policy.zip").read_bytes() == archive_before
    assert receipt["model_states_before"] == receipt["model_states_after"]
    assert receipt["original_policy_digests"] == [digest, digest]
    assert (
        receipt["runtime_recipe_digest"] == stress.recipe_digest != base.recipe_digest
    )
    assert [row["decision_index"] for row in receipt["actor_rows"]] == [6, 7, 8, 9]
    field = stress.observation_schema.fields.index("remaining_horizon_fraction")
    observations = [
        np.frombuffer(bytes.fromhex(row["input_hex"]), dtype=np.float32)
        for row in receipt["actor_rows"]
    ]
    assert [float(obs[field]) for obs in observations] == [1, 0.75, 0.5, 0.25]
    assert [
        policy.model.predict(obs, deterministic=True)[0].item() for obs in observations
    ] == [row["action"] for row in receipt["actor_rows"]]
    if frozen:
        assert base.feature_preprocessing.digest == stress.feature_preprocessing.digest
    with pytest.raises(ValueError, match="runtime recipe"):
        policy.action(observations[0], runtime_recipe_digest=stress.recipe_digest)
    persist_software_evidence(
        tmp_path / "software-evidence.json",
        receipt,
        software_kind=(
            "genuine_unchanged_scheduled_v3"
            if frozen
            else "genuine_unchanged_scheduled_v2"
        ),
    )

    persist_global_software_evidence(
        tmp_path / "global-comparison" / "software-evidence.json",
        evidence,
        software_kind="genuine_unchanged_scheduled_v3"
        if frozen
        else "genuine_unchanged_scheduled_v2",
    )


def test_actual_constant_actor_native_fee_oracle_is_separate_from_learning(tmp_path):
    policy, base, stress = runtime_case(frozen=True)
    originals = []
    # Deliberate fixed-action software fixture; all modifications precede both
    # published frozen archives and every OOS execution.
    for action in (3, 0):
        with torch.no_grad():
            policy.model.policy.action_net.weight.zero_()
            policy.model.policy.action_net.bias.fill_(-20)
            policy.model.policy.action_net.bias[action] = 20
        root = tmp_path / f"constant-{action}"
        originals.append((root, save_allocation_policy(root, policy)))
    evidence = {}
    result, folds, artifacts = execute(
        base, stress, *originals[0], evidence=evidence, second=originals[1]
    )
    run_artifact_bound_global_allocation_walk_forward(
        folds, base, artifacts, reset_seed=7
    )
    assert [row["action"] for row in result.receipt["actor_rows"]] == [3, 0, 0, 0]
    assert result.receipt["actor_rows"][1]["cash"] == 498.0
    assert result.receipt["actor_rows"][1]["equity"] == 1048.0
    assert base.book.cash == 499.0 and stress.book.cash == 498.0
    assert base.book.total_cost == 1.0 and stress.book.total_cost == 2.0
    assert base.book.fill_count == stress.book.fill_count == 1
    assert base.book.exact_quantities == stress.book.exact_quantities == (5,)
    assert result.receipt["model_states_before"] == result.receipt["model_states_after"]
    persist_software_evidence(
        tmp_path / "software-evidence.json",
        result.receipt,
        software_kind="deliberately_constant_v3_software_fixture",
        native_facts={
            "actions": [row["action"] for row in result.receipt["actor_rows"]],
            "first_stress_fill": {
                "cash": result.receipt["actor_rows"][1]["cash"],
                "equity": result.receipt["actor_rows"][1]["equity"],
            },
            "base": {
                "cash": base.book.cash,
                "total_cost": base.book.total_cost,
                "fill_count": base.book.fill_count,
                "exact_quantities": [str(q) for q in base.book.exact_quantities],
            },
            "stress": {
                "cash": stress.book.cash,
                "total_cost": stress.book.total_cost,
                "fill_count": stress.book.fill_count,
                "exact_quantities": [str(q) for q in stress.book.exact_quantities],
            },
        },
    )

    persist_global_software_evidence(
        tmp_path / "global-comparison" / "software-evidence.json",
        evidence,
        software_kind="deliberately_constant_v3_software_fixture",
    )
