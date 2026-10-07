"""No-fit native received-input and execution-fact software controls."""

from importlib import import_module, util

import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


def capability():
    name = "trade_rl.evaluation.allocation_global_execution"
    assert util.find_spec(name) is not None, "closed global execution facade is missing"
    return import_module(name)


def declare(api, env, folds, *, kind="nonrl", scenario="base", **kwargs):
    provenance = build_candidate_run_provenance()
    return api.declare_global_allocation_execution(
        folds,
        env,
        kind=kind,
        scenario=scenario,
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_digest=provenance["runtime_environment_digest"],
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["nonrl", "cash"])
@pytest.mark.parametrize("boundary", [7, 9])
def test_completed_receipt_has_four_original_inputs_and_native_events(kind, boundary):
    api = capability()
    env = global_control_fixture(
        mode="direct" if kind == "cash" else "residual", cash_rate=0.876
    )
    folds = global_control_folds(boundary)
    plan = declare(api, env, folds, kind=kind)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    receipt = result.receipt.payload
    assert receipt["status"] == "completed"
    assert receipt["reset_count"] == 1
    assert len(receipt["rows"]) == 4
    assert [row["decision_index"] for row in receipt["rows"]] == [6, 7, 8, 9]
    assert [row["call_outcome"]["action"] for row in receipt["rows"]] == [
        0 if kind == "cash" else 2
    ] * 4
    assert [row["native"]["processing_index"] for row in receipt["rows"]] == [
        7,
        8,
        9,
        10,
    ]
    assert receipt["native_terminal"] is True and receipt["termination_reason"] is None
    assert receipt["model_states_before"] == receipt["model_states_after"] == []
    assert receipt["plan"]["cell"]["seed"] is None
    assert "validity" not in receipt and "score" not in receipt
    assert env._transition_recorder is None
    restored = api.GlobalAllocationExecutionReceipt.from_payload(receipt)
    assert restored.digest == result.receipt.digest
    if kind == "cash":
        assert env.book.cash == pytest.approx(1000 * (1 + 0.876 / 8760) ** 4)
        assert env.book.fill_count == 0


def test_common_context_is_candidate_independent_and_cell_recipe_is_separate():
    api = capability()
    folds = global_control_folds(7)
    residual, direct = global_control_fixture(), global_control_fixture(mode="direct")
    nonrl, cash = (
        declare(api, residual, folds),
        declare(api, direct, folds, kind="cash"),
    )
    assert nonrl.payload["common"] == cash.payload["common"]
    assert nonrl.common_digest == cash.common_digest
    assert nonrl.payload["cell"] != cash.payload["cell"]


def test_economic_stop_keeps_actual_native_prefix_and_original_error():
    api = capability()
    env = global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True)
    folds = global_control_folds(7)
    plan = declare(api, env, folds)
    with pytest.raises(ValueError, match="economic termination") as raised:
        api.run_declared_global_allocation_execution(folds, env, plan)
    receipt = raised.value.global_execution_receipt.payload
    assert receipt["status"] == "economic_stop"
    assert (
        len(receipt["rows"]) == 1
        and receipt["rows"][0]["native"]["book"]["equity"] == -501
    )
    assert env.book.cash == env.book.portfolio_value == -501 and env.index == 7
    assert receipt["termination_reason"] == env.book.termination_reason
    assert env._transition_recorder is None


def test_occupied_foreign_observer_rejects_before_reset_and_is_preserved(monkeypatch):
    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    foreign = object()
    env._transition_recorder = foreign
    monkeypatch.setattr(env, "reset", lambda **kw: pytest.fail("reset"))
    with pytest.raises(ValueError, match="occupied"):
        api.run_declared_global_allocation_execution(folds, env, plan)
    assert env._transition_recorder is foreign and not hasattr(env, "book")


def test_post_native_source_mutation_retains_event_as_integrity_failure(monkeypatch):
    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    native = env.step

    def changed(action):
        result = native(action)
        object.__setattr__(env.dataset, "periods_per_year", 9000)
        return result

    monkeypatch.setattr(env, "step", changed)
    with pytest.raises(ValueError, match="source|runtime") as raised:
        api.run_declared_global_allocation_execution(folds, env, plan)
    receipt = raised.value.global_execution_receipt.payload
    assert receipt["status"] == "integrity_failure"
    assert len(receipt["rows"]) == 1
    assert receipt["rows"][0]["native"]["processing_index"] == env.index == 7
    assert env.book.fill_count == 1
    assert env._transition_recorder is None


def test_closed_plan_and_receipt_reject_unknown_fields_and_unobserved_claims():
    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionPlan.from_payload(
            plan.payload | {"validity": "VALID"}
        )
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    bad = result.receipt.payload
    bad["rows"][0]["input"]["seam"] = "model_predict_tensor"
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionReceipt.from_payload(bad)


def test_structurally_matching_foreign_collector_rejects_before_reset(monkeypatch):
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )
    from trade_rl.evaluation.rl_allocation.global_walk_forward import (
        run_global_allocation_walk_forward,
    )

    env, folds = global_control_fixture(), global_control_folds(7)

    class ForeignCollector:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def freeze_execution(self, result):
            return b""

        def loaded_policies(self, policies):
            pass

        def before_reset(self, *args):
            pass

        def after_reset(self):
            pass

        def before_action(self, *args):
            pass

        def after_action(self, action):
            pass

        def action_failed(self, error):
            pass

        def after_step(self, *args):
            pass

        def completed(self, result):
            pass

        def failed(self, error):
            pass

    policies = tuple(
        AllocationFoldPolicy("a" * 64, env.recipe_digest, lambda *_: 2) for _ in folds
    )
    monkeypatch.setattr(
        env, "reset", lambda **kw: pytest.fail("foreign collector reached reset")
    )
    with pytest.raises(ValueError, match="exact native observational collector"):
        run_global_allocation_walk_forward(
            folds, env, policies, collector=ForeignCollector()
        )
    assert env._transition_recorder is None and not hasattr(env, "book")


def test_completed_reader_rejects_inconsistent_segments_policy_and_provenance():
    from copy import deepcopy

    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    receipt = api.run_declared_global_allocation_execution(
        folds, env, plan
    ).receipt.payload
    mutations = [
        lambda p: p["segments"].clear(),
        lambda p: p["segments"][0].update(range=[6, 8]),
        lambda p: p["segments"][1].update(opening_state_digest="b" * 64),
        lambda p: p["rows"][0].update(fold_index=99),
        lambda p: p["rows"][0].update(policy_digest="b" * 64),
        lambda p: p["rows"][0].update(terminated=True),
        lambda p: p["rows"][-1].update(terminated=False),
        lambda p: p["provenance_after"].update(implementation_digest="b" * 64),
        lambda p: p.update(
            model_states_before=["b" * 64], model_states_after=["b" * 64]
        ),
    ]
    for mutate in mutations:
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(ValueError):
            api.GlobalAllocationExecutionReceipt.from_payload(bad)


def test_received_argument_is_copied_before_mutating_same_public_call():
    import numpy as np

    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )
    from trade_rl.evaluation.rl_allocation.global_execution_context import (
        GlobalAllocationExecutionCollector,
    )
    from trade_rl.evaluation.rl_allocation.global_walk_forward import (
        run_global_allocation_walk_forward,
    )

    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    capture = GlobalAllocationExecutionCollector(
        env, plan.payload["common"], plan.payload["cell"]
    )
    received = []

    def mutate(observation, recipe):
        received.append(observation.copy())
        observation[:] = 0
        return 2

    policies = tuple(
        AllocationFoldPolicy(
            plan.payload["cell"]["control_policy_digest"], env.recipe_digest, mutate
        )
        for _ in folds
    )
    run_global_allocation_walk_forward(folds, env, policies, collector=capture)
    assert len(capture.rows) == len(received) == 4
    for row, before in zip(capture.rows, received, strict=True):
        assert bytes.fromhex(row["input"]["hex"]) == before.tobytes()
        assert np.isfinite(before).all() and before.any()


def test_none_keeps_legacy_foreign_slot_and_omitted_bytes():
    from trade_rl.artifacts import canonical_json_bytes
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )
    from trade_rl.evaluation.rl_allocation.global_walk_forward import (
        run_global_allocation_walk_forward,
    )
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        freeze_allocation_execution,
    )

    snapshots = []
    for explicit in (False, True):
        env, folds = global_control_fixture(), global_control_folds(7)

        class ForeignObserver:
            def __init__(self):
                self.traces = []

            def freeze_execution(self, result):
                raw = freeze_allocation_execution(env, result)
                self.traces.append(raw)
                return raw

        observer = ForeignObserver()
        env._transition_recorder = observer
        policies = tuple(
            AllocationFoldPolicy("a" * 64, env.recipe_digest, lambda *_: 2)
            for _ in folds
        )
        result = run_global_allocation_walk_forward(
            folds, env, policies, **({"collector": None} if explicit else {})
        )
        assert env._transition_recorder is observer
        snapshots.append(
            (
                observer.traces,
                canonical_json_bytes(result),
                env.executor._rng.bit_generator.state,
            )
        )
    assert snapshots[0] == snapshots[1]


def test_fake_artifact_policy_seed_and_reset_seed_stay_distinct(tmp_path, monkeypatch):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import (
        FakeModel,
        native_pair,
        publish_fake,
    )
    from tests.evaluation.test_allocation_fee_stress import declaration
    from trade_rl.artifacts import canonical_json_bytes
    from trade_rl.evaluation.rl_allocation.global_walk_forward import (
        run_artifact_bound_global_allocation_walk_forward,
    )

    api = capability()
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    loads = []
    monkeypatch.setattr(
        archive, "_load_policy", lambda path: (loads.append(path), model)[1]
    )
    plan = declare(
        api,
        base,
        folds,
        kind="residual_ppo",
        seed=7,
        reset_seed=17,
        artifacts=artifacts,
    )
    resets = []
    native_reset = base.reset
    monkeypatch.setattr(
        base, "reset", lambda **kw: (resets.append(kw["seed"]), native_reset(**kw))[1]
    )
    observed = api.run_declared_global_allocation_execution(
        folds, base, plan, artifacts=artifacts
    )
    assert resets == [17] and len(loads) == 2 and len(model.calls) == 4
    assert observed.receipt.payload["plan"]["cell"]["seed"] == 7
    assert observed.receipt.payload["plan"]["cell"]["reset_seed"] == 17
    reference, _ = native_pair()
    other = FakeModel(
        archive.read_allocation_policy_manifest(
            root, expected_digest=digest, expected_recipe_digest=reference.recipe_digest
        )
    )
    monkeypatch.setattr(archive, "_load_policy", lambda _: other)
    original = run_artifact_bound_global_allocation_walk_forward(
        folds, reference, artifacts, reset_seed=17
    )
    assert canonical_json_bytes(original) == canonical_json_bytes(
        observed.native_result
    )
    assert [v.tobytes() for v in model.calls] == [v.tobytes() for v in other.calls]
    assert (
        base.executor._rng.bit_generator.state
        == reference.executor._rng.bit_generator.state
    )
    assert base.book == reference.book


@pytest.mark.parametrize("mutation", ["source", "cost", "model", "nonfinite_model"])
def test_fake_action_drift_keeps_received_call_but_no_native_event(
    tmp_path, monkeypatch, mutation
):
    import numpy as np

    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration

    api = capability()
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    plan = declare(api, base, folds, kind="residual_ppo", seed=7, artifacts=artifacts)

    def changed():
        if mutation == "source":
            object.__setattr__(base.dataset, "periods_per_year", 9000)
        elif mutation == "cost":
            object.__setattr__(base.execution_cost, "fee_rate", 0.003)
        else:
            model.weights["actor.weight"][0, 0] = (
                np.nan if mutation == "nonfinite_model" else 2
            )

    model.before_predict = changed
    with pytest.raises(ValueError) as raised:
        api.run_declared_global_allocation_execution(
            folds, base, plan, artifacts=artifacts
        )
    receipt = raised.value.global_execution_receipt.payload
    assert receipt["status"] == "integrity_failure" and len(receipt["rows"]) == 1
    assert (
        receipt["rows"][0]["native"] is None
        and receipt["rows"][0]["call_outcome"]["status"] == "returned"
    )
    assert len(model.calls) == 1 and base.index == 6 and base.book.fill_count == 0
    assert base._transition_recorder is None


def test_whole_zip_roster_preflight_before_load_or_reset(tmp_path, monkeypatch):
    from dataclasses import replace
    from shutil import copytree

    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration

    api = capability()
    base, stress = native_pair()
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    bad = tmp_path / "second"
    copytree(root, bad)
    artifacts = (artifacts[0], replace(artifacts[1], bundle_root=bad))
    plan = declare(api, base, folds, kind="residual_ppo", seed=7, artifacts=artifacts)
    (bad / "policy.zip").write_bytes(b"changed second archive")
    monkeypatch.setattr(archive, "_load_policy", lambda *_: pytest.fail("backend load"))
    monkeypatch.setattr(base, "reset", lambda **kw: pytest.fail("reset"))
    with pytest.raises(ValueError, match="archive digest"):
        api.run_declared_global_allocation_execution(
            folds, base, plan, artifacts=artifacts
        )
    assert not hasattr(base, "book")


def test_received_native_partial_no_fill_carry_preserves_single_account():
    api = capability()
    env = global_control_fixture(
        signals=(1, 1, 1, 1),
        zero_later_volume=True,
        cost_changes={
            "max_participation_rate": 0.00001,
            "random_seed": 73,
            "slippage_std": 0.01,
        },
    )
    folds = global_control_folds(7)
    plan = declare(api, env, folds)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    rows = result.receipt.payload["rows"]
    assert len(rows) == 4 and rows[0]["native"]["active_orders"]
    assert any(
        e["event_type"] == "partial_fill" for e in rows[0]["native"]["order_events"]
    )
    assert any(e["event_type"] == "no_fill" for e in rows[1]["native"]["order_events"])
    assert rows[0]["after_state_digest"] == rows[1]["before_state_digest"]
    assert env.book.fill_count == 1


def test_loaded_policy_seed_mismatch_fails_before_reset(tmp_path, monkeypatch):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration

    api = capability()
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    plan = declare(
        api,
        base,
        folds,
        kind="residual_ppo",
        seed=7,
        reset_seed=17,
        artifacts=artifacts,
    )
    model.seed = 17
    calls = []
    monkeypatch.setattr(
        archive, "_load_policy", lambda path: (calls.append(path), model)[1]
    )
    monkeypatch.setattr(base, "reset", lambda **kw: pytest.fail("reset"))
    with pytest.raises(
        ValueError, match="actual PPO runtime differs from its training receipt"
    ):
        api.run_declared_global_allocation_execution(
            folds, base, plan, artifacts=artifacts
        )
    assert len(calls) == 1 and not hasattr(base, "book")


def test_shared_raw_context_survives_frozen_candidate_preprocessing():
    from tests.evaluation.allocation_fee_stress_fixture import native_pair

    api = capability()
    base, _ = native_pair()
    frozen, _ = native_pair(frozen=True)
    folds = global_control_folds(7)
    # Both seedless native controls reuse the same raw source but different
    # existing candidate observation/preprocessing declarations.
    first, second = declare(api, base, folds), declare(api, frozen, folds)
    assert first.payload["common"] == second.payload["common"]
    assert (
        first.payload["cell"]["original_recipe"]
        != second.payload["cell"]["original_recipe"]
    )


def test_rejected_detached_facts_remain_explicit_integrity_record(monkeypatch):
    import json

    import trade_rl.evaluation.rl_allocation.global_execution_context as lower
    from trade_rl.artifacts import canonical_json_bytes

    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    original = lower.freeze_allocation_execution

    def rejected(env, result):
        raw = json.loads(original(env, result))
        raw["processing_index"] = 99
        return canonical_json_bytes(raw)

    monkeypatch.setattr(lower, "freeze_allocation_execution", rejected)
    with pytest.raises(ValueError) as raised:
        api.run_declared_global_allocation_execution(folds, env, plan)
    receipt = raised.value.global_execution_receipt.payload
    row = receipt["rows"][0]
    assert receipt["status"] == "integrity_failure"
    assert row["native"]["processing_index"] == 99
    assert row["native_validation_failure"] is not None
    assert row["after_state_digest"] is None and env.index == 6
    with pytest.raises(ValueError):
        api.ObservedGlobalAllocationExecution(
            object(), raised.value.global_execution_receipt
        )


def test_completion_requires_observed_opening_hash_and_exact_cash_reset_seed():
    api = capability()
    env, folds = global_control_fixture(mode="direct"), global_control_folds(7)
    plan = declare(api, env, folds, kind="cash")
    record = api.run_declared_global_allocation_execution(
        folds, env, plan
    ).receipt.payload
    record["opening_state_digest"] = None
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionReceipt.from_payload(record)
    bad = plan.payload
    bad["cell"]["reset_seed"] = False
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionPlan.from_payload(bad)


@pytest.mark.parametrize(
    "returned",
    [True, 1.5, 4, float("nan"), object()],
    ids=["bool", "float", "range", "nonfinite", "unsupported"],
)
def test_invalid_public_return_is_detached_before_rejecting_native_step(
    tmp_path, monkeypatch, returned
):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration
    from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy

    api = capability()
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    monkeypatch.setattr(
        AllocationPPOPolicy, "action", lambda self, *args, **kwargs: returned
    )
    plan = declare(api, base, folds, kind="residual_ppo", seed=7, artifacts=artifacts)
    with pytest.raises(ValueError, match="integer within 0..3") as raised:
        api.run_declared_global_allocation_execution(
            folds, base, plan, artifacts=artifacts
        )
    receipt = raised.value.global_execution_receipt.payload
    outcome = receipt["rows"][0]["call_outcome"]
    assert outcome is not None and outcome["status"] == "invalid_return"
    assert outcome["validation_failure"]["message"] == str(raised.value)
    assert outcome["encoding"] in {
        "native_scalar",
        "unavailable_nonfinite",
        "unavailable_unsupported",
    }
    if (
        isinstance(returned, (bool, int, float))
        and outcome["encoding"] == "native_scalar"
    ):
        assert outcome["value"] == returned and type(outcome["value"]) is type(returned)
    else:
        assert outcome["value"] is None
    assert (
        receipt["status"] == "integrity_failure"
        and receipt["rows"][0]["native"] is None
    )
    assert base.index == 6 and base.book.fill_count == 0 and len(model.calls) == 0
    assert base._transition_recorder is None
    assert (
        api.GlobalAllocationExecutionReceipt.from_payload(receipt).digest
        == raised.value.global_execution_receipt.digest
    )


@pytest.mark.parametrize("fault", ["source", "model"])
def test_post_insolvency_integrity_fault_preserves_signed_book_and_its_classification(
    tmp_path, monkeypatch, fault
):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import publish_fake
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
    )

    api = capability()
    env = global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True)
    folds = global_control_folds(7)
    artifacts = ()
    model = None
    if fault == "model":
        root, digest, model = publish_fake(env, tmp_path, monkeypatch)
        artifacts = tuple(
            AllocationFoldPolicyArtifact(f.fold_index, root, digest, env.recipe_digest)
            for f in folds
        )
        monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    plan = declare(
        api,
        env,
        folds,
        **(
            {"kind": "residual_ppo", "seed": 7, "artifacts": artifacts}
            if fault == "model"
            else {}
        ),
    )
    native_step = env.step

    def changed(action):
        result = native_step(action)
        if fault == "source":
            object.__setattr__(env.dataset, "periods_per_year", 9000)
        else:
            model.weights["actor.weight"][0, 0] = 2
        return result

    monkeypatch.setattr(env, "step", changed)
    with pytest.raises(ValueError) as raised:
        api.run_declared_global_allocation_execution(
            folds, env, plan, artifacts=artifacts
        )
    receipt = raised.value.global_execution_receipt.payload
    assert receipt["status"] == "integrity_failure"
    assert receipt["termination_reason"] == env.book.termination_reason
    assert receipt["rows"][0]["native"]["book"]["cash"] == env.book.cash == -501
    assert (
        receipt["rows"][0]["native"]["book"]["equity"]
        == env.book.portfolio_value
        == -501
    )
    assert env.index == 7 and env._transition_recorder is None


@pytest.mark.parametrize("kind", ["nonrl", "cash"])
@pytest.mark.parametrize(
    "field", ["forecast_context_digest", "runtime_recipe_digest", "candidate"]
)
def test_observed_control_wrapper_rechecks_all_plan_metadata(kind, field):
    from dataclasses import replace

    api = capability()
    env = global_control_fixture(mode="direct" if kind == "cash" else "residual")
    folds = global_control_folds(7)
    plan = declare(api, env, folds, kind=kind)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    changed = replace(result.native_result)
    actual_field = (
        ("carrier_recipe_digest" if kind == "cash" else "candidate_recipe_digest")
        if field == "candidate"
        else field
    )
    object.__setattr__(changed, actual_field, "b" * 64)
    with pytest.raises(ValueError):
        api.ObservedGlobalAllocationExecution(changed, result.receipt)


def test_observed_fee_wrapper_binds_legacy_metadata_and_full_original_bytes(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    from dataclasses import replace

    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration
    from trade_rl.artifacts import canonical_json_bytes

    api = capability()
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, contract, artifacts = declaration(base, stress, root, digest)
    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    plan = declare(
        api,
        stress,
        folds,
        kind="residual_ppo",
        scenario="configured_fee_x2",
        seed=7,
        artifacts=artifacts,
        base_env=base,
        contract=contract,
        fee_factor=2.0,
    )
    result = api.run_declared_global_allocation_execution(
        folds, stress, plan, artifacts=artifacts, base_env=base, contract=contract
    )
    for key, value in [
        ("scenario", "changed"),
        ("seed", 17),
        ("contract_digest", "b" * 64),
        ("original_policy_digests", ["b" * 64] * 2),
        ("model_states_before", ["b" * 64] * 2),
        ("binding_digests", ["b" * 64] * 2),
    ]:
        raw = deepcopy(result.native_result.receipt)
        raw[key] = value
        altered = replace(result.native_result, _receipt=canonical_json_bytes(raw))
        with pytest.raises(ValueError):
            api.ObservedGlobalAllocationExecution(altered, result.receipt)


def test_closed_receipt_recomputes_actual_cost_payload_economics():
    api = capability()
    from trade_rl.artifacts import content_digest

    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    bad = result.receipt.payload
    bad["plan"]["cell"]["runtime"]["cost_payload"]["fee_rate"] = 0.004
    bad["cell_plan_digest"] = content_digest(bad["plan"]["cell"])
    with pytest.raises(ValueError):
        api.GlobalAllocationExecutionReceipt.from_payload(bad)
