"""Fee-only native global controls; fake policy does not imply learned alpha."""

import json
from dataclasses import replace
from hashlib import sha256
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
from tests.evaluation.test_allocation_continuous_walk_forward import fold
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.evaluation.allocation_comparison import AllocationCandidateKind
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_comparison_scenario_digest,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
)
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    run_global_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.policy_admission import (
    AllocationFoldPolicyArtifact,
)
from trade_rl.strategies.rl.allocation_artifact import load_allocation_policy


def declaration(base, stress, root, digest, boundary=7):
    from tests.evaluation.test_allocation_comparison_evidence import contract_for
    from trade_rl.evaluation.allocation_comparison import AllocationComparisonScenario

    folds = (fold(0, 6, boundary), fold(1, boundary, 10))
    contract = contract_for(base, folds)
    scenario = AllocationComparisonScenario(
        "configured_fee_x2",
        allocation_comparison_scenario_digest(
            name="configured_fee_x2",
            dataset_id=base.dataset.dataset_id,
            forecast_context_digest=base.stream.digest,
            economics_digest=stress.executor.execution_policy_digest,
            risk_digest=content_digest(stress.risk_config),
        ),
        True,
    )
    from dataclasses import replace

    contract = replace(
        contract, scenarios=contract.scenarios + (scenario,), rl_seeds=(7,)
    )
    artifacts = tuple(
        AllocationFoldPolicyArtifact(f.fold_index, root, digest, base.recipe_digest)
        for f in folds
    )
    return folds, contract, artifacts


def run(base, stress, root, digest, *, boundary=7):
    folds, contract, artifacts = declaration(base, stress, root, digest, boundary)
    name = "trade_rl.evaluation.rl_allocation.fee_stress_admission"
    if util.find_spec(name) is None:
        # Connected RED: actual native reset and ordinary prediction guard, not
        # an import-error test. Without the fee view no native fill can occur.
        loaded = load_allocation_policy(
            root, expected_digest=digest, expected_recipe_digest=base.recipe_digest
        )
        policies = tuple(
            AllocationFoldPolicy(
                digest,
                stress.recipe_digest,
                lambda observation, actual: loaded.action(
                    observation, runtime_recipe_digest=actual
                ),
            )
            for _ in folds
        )
        try:
            run_global_allocation_walk_forward(folds, stress, policies, reset_seed=7)
        except ValueError as error:
            assert "runtime recipe differs" in str(error)
        assert stress.book.fill_count == 1, "fee-view missing: native fill count is 0"
        raise AssertionError("unexpected unadmitted ordinary route")
    return import_module(name).run_fee_stressed_global_allocation(
        folds,
        base_env=base,
        stress_env=stress,
        artifacts=artifacts,
        contract=contract,
        candidate=AllocationCandidateKind.RESIDUAL_PPO,
        scenario="configured_fee_x2",
        fee_factor=2.0,
        reset_seed=7,
    )


@pytest.mark.parametrize("boundary", [7, 9])
@pytest.mark.parametrize("volume", [100, 100000])
def test_same_frozen_policy_fee_difference_reaches_native_global_account(
    tmp_path, monkeypatch, boundary, volume
):
    base, stress = native_pair(volume=volume)
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    result = run(base, stress, root, digest, boundary=boundary)
    receipt = result.receipt
    assert receipt["actor_rows"][1]["cash"] == 498.0
    assert receipt["actor_rows"][1]["equity"] == 1048.0
    assert stress.book.exact_quantities == (5,)
    assert stress.book.cash == 498.0
    assert stress.book.total_cost == 2.0
    assert stress.book.fill_count == 1
    assert not hasattr(base, "book")
    assert result.walk_forward.policy_digests == (digest, digest)
    assert (
        receipt["runtime_recipe_digest"] == stress.recipe_digest != base.recipe_digest
    )
    assert allocation_candidate_recipe_digest(
        base.recipe
    ) == allocation_candidate_recipe_digest(stress.recipe)
    field = stress.observation_schema.fields.index("remaining_horizon_fraction")
    assert [float(values[field]) for values in model.calls] == [1, 0.75, 0.5, 0.25]
    assert [row["decision_index"] for row in receipt["actor_rows"]] == [6, 7, 8, 9]
    assert receipt["model_states_before"] == receipt["model_states_after"]


@pytest.mark.parametrize(
    "change", ["spread_rate", "multiplier", "random_seed", "slippage_std"]
)
def test_nonfee_cost_change_fails_before_native_reset(tmp_path, monkeypatch, change):
    base, stress = native_pair(**{change: 0.01 if change != "random_seed" else 17})
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="fee|cost"):
        run(base, stress, root, digest)
    assert not hasattr(stress, "book")


@pytest.mark.parametrize("mutation", ["cost", "weights", "cash", "clock"])
def test_prediction_mutation_stops_before_first_native_execution(
    tmp_path, monkeypatch, mutation
):
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    cached = stress.executor.execution_policy_digest

    def mutate():
        if mutation == "cost":
            object.__setattr__(stress.execution_cost, "fee_rate", 0.008)
            assert stress.executor.execution_policy_digest == cached
        elif mutation == "weights":
            model.weights["actor.weight"][0, 0] = 2
        elif mutation == "cash":
            stress.book.cash -= 1
        else:
            stress.index += 1

    model.before_predict = mutate
    expected = {
        "cost": "contents changed",
        "weights": "tensor contents changed",
        "cash": "drawdown evidence",
        "clock": "processing clock",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        run(base, stress, root, digest)
    assert stress.book.fill_count == 0


def test_contract_type_is_rejected_before_seed_or_optional_backend(
    tmp_path, monkeypatch
):
    base, stress = native_pair()
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    folds, _, artifacts = declaration(base, stress, root, digest)
    module = import_module("trade_rl.evaluation.rl_allocation.fee_stress_admission")
    monkeypatch.setattr(
        module, "load_allocation_policy", lambda *a, **k: pytest.fail("backend")
    )
    with pytest.raises(ValueError, match="comparison contract"):
        module.run_fee_stressed_global_allocation(
            folds,
            base_env=base,
            stress_env=stress,
            artifacts=artifacts,
            contract=None,
            candidate=AllocationCandidateKind.RESIDUAL_PPO,
            scenario="configured_fee_x2",
            fee_factor=2.0,
            reset_seed=7,
        )
    assert not hasattr(stress, "book")


def test_declared_seed_mismatch_is_metadata_failure_before_backend(
    tmp_path, monkeypatch
):
    base, stress = native_pair()
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    folds, contract, artifacts = declaration(base, stress, root, digest)
    contract = replace(contract, rl_seeds=(8,))
    module = import_module("trade_rl.evaluation.rl_allocation.fee_stress_admission")
    monkeypatch.setattr(
        module,
        "load_allocation_policy",
        lambda *a, **k: pytest.fail("backend before seed metadata"),
    )
    with pytest.raises(ValueError, match="seed"):
        module.run_fee_stressed_global_allocation(
            folds,
            base_env=base,
            stress_env=stress,
            artifacts=artifacts,
            contract=contract,
            candidate=AllocationCandidateKind.RESIDUAL_PPO,
            scenario="configured_fee_x2",
            fee_factor=2.0,
            reset_seed=8,
        )
    assert not hasattr(stress, "book")


@pytest.mark.parametrize("failure", ["metadata", "archive"])
def test_late_roster_failure_prevents_reset_and_respects_backend_boundary(
    tmp_path, monkeypatch, failure
):
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    folds, contract, artifacts = declaration(base, stress, root, digest)
    raw = json.loads((root / "manifest.json").read_bytes())
    bad = tmp_path / "bad-second"
    bad.mkdir()
    if failure == "metadata":
        raw["training"]["sources"][1]["source"]["decision_counts"][-1] = 9
        policy_bytes = (root / "policy.zip").read_bytes()
    else:
        policy_bytes = b"tampered late archive"
    (bad / "manifest.json").write_bytes(canonical_json_bytes(raw))
    (bad / "policy.zip").write_bytes(policy_bytes)
    artifacts = (
        artifacts[0],
        replace(artifacts[1], bundle_root=bad, expected_digest=content_digest(raw)),
    )
    module = import_module("trade_rl.evaluation.rl_allocation.fee_stress_admission")
    import trade_rl.strategies.rl.allocation_artifact as archive

    loads = []
    monkeypatch.setattr(
        archive,
        "_load_policy",
        lambda path: loads.append(sha256(path.read_bytes()).hexdigest()) or model,
    )
    with pytest.raises(ValueError):
        module.run_fee_stressed_global_allocation(
            folds,
            base_env=base,
            stress_env=stress,
            artifacts=artifacts,
            contract=contract,
            candidate=AllocationCandidateKind.RESIDUAL_PPO,
            scenario="configured_fee_x2",
            fee_factor=2.0,
            reset_seed=7,
        )
    assert loads == [], "late canonical ZIP corruption reached the first backend"
    assert not hasattr(stress, "book")
    assert model.calls == []


@pytest.mark.parametrize("phase", ["before_predict", "during_predict"])
@pytest.mark.parametrize(
    "metadata", ["volume_units", "periods_per_year", "feature_config_digest"]
)
def test_same_object_native_metadata_mutation_stops_before_first_execution(
    tmp_path, monkeypatch, phase, metadata
):
    base, stress = native_pair(volume=100)
    dataset = stress.dataset
    assert dataset is base.dataset
    assert dataset.open[7, 0] == 100 and dataset.volume[7, 0] == 100
    assert dataset.market_notional(7).tolist() == [10000.0]
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.evaluation.allocation as native
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    executions = []
    execute = native.execute_target_statefully

    def observed_execute(*args, **kwargs):
        executions.append(kwargs["start_index"])
        return execute(*args, **kwargs)

    monkeypatch.setattr(native, "execute_target_statefully", observed_execute)

    def mutate():
        value = {
            "volume_units": (VolumeUnit.QUOTE_NOTIONAL,),
            "periods_per_year": 365.0,
            "feature_config_digest": "f" * 64,
        }[metadata]
        object.__setattr__(dataset, metadata, value)
        assert stress.dataset is base.dataset is dataset
        if metadata == "volume_units":
            assert dataset.market_notional(7).tolist() == [100.0]

    if phase == "during_predict":
        model.before_predict = mutate
    else:
        reset = stress.reset

        def admitted_reset(**kwargs):
            result = reset(**kwargs)
            mutate()
            return result

        monkeypatch.setattr(stress, "reset", admitted_reset)
    rejected = None
    try:
        run(base, stress, root, digest)
    except ValueError as error:
        rejected = error
    assert executions == [], (
        f"metadata {metadata} changed native first capacity: execution indices {executions}"
    )
    assert isinstance(rejected, ValueError) and "runtime" in str(rejected)
    assert stress.book.fill_count == 0
    assert len(model.calls) == (1 if phase == "during_predict" else 0)


def test_native_base_control_has_one_reset_global_terminal_and_literal_cash(
    tmp_path, monkeypatch
):
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    stressed = run(base, stress, root, digest)
    model.actions, model.calls = iter((2, 0, 0, 0)), []
    ordinary = load_allocation_policy(
        root, expected_digest=digest, expected_recipe_digest=base.recipe_digest
    )
    folds, _, _ = declaration(base, stress, root, digest)
    resets, terminals = [], []
    native_reset, native_step = base.reset, base.step

    def reset(**kwargs):
        resets.append(kwargs)
        return native_reset(**kwargs)

    def step(action):
        result = native_step(action)
        if result[2]:
            terminals.append(base.index)
        return result

    monkeypatch.setattr(base, "reset", reset)
    monkeypatch.setattr(base, "step", step)
    policies = tuple(
        AllocationFoldPolicy(
            digest,
            base.recipe_digest,
            lambda observation, actual: ordinary.action(
                observation, runtime_recipe_digest=actual
            ),
        )
        for _ in folds
    )
    run_global_allocation_walk_forward(folds, base, policies, reset_seed=7)
    assert resets == [{"seed": 7}] and terminals == [10]
    assert base.book.exact_quantities == stress.book.exact_quantities == (5,)
    assert base.book.cash == 499.0 and base.book.total_cost == 1.0
    assert base.book.fill_count == stress.book.fill_count == 1
    field = base.observation_schema.fields.index("cash_over_initial_capital")

    # Independent toward-zero float32 projection of the literal native cash.
    def toward_zero(value):
        rounded = np.float32(value)
        return (
            np.nextafter(rounded, np.float32(0)) if float(rounded) > value else rounded
        )

    assert model.calls[1][field] == toward_zero(499 / 1000)
    stress_input = np.frombuffer(
        bytes.fromhex(stressed.receipt["actor_rows"][1]["input_hex"]), dtype=np.float32
    )
    assert stress_input[field] == toward_zero(498 / 1000)
    assert model.calls[1][field] != stress_input[field]
    equity = base.observation_schema.fields.index("equity_over_initial_capital")
    assert model.calls[1][equity] == toward_zero(1049 / 1000)
    assert stress_input[equity] == toward_zero(1048 / 1000)
    weight = base.observation_schema.fields.index("current_weight")
    assert model.calls[1][weight] == toward_zero(550 / 1049)
    assert stress_input[weight] == toward_zero(550 / 1048)


def test_in_place_actor_source_change_is_detected_before_execution(
    tmp_path, monkeypatch
):
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)

    def mutate():
        values = stress.dataset.features.copy()
        values[8, 0, 0] = 17
        object.__setattr__(stress.dataset, "features", values)

    model.before_predict = mutate
    with pytest.raises(ValueError, match="runtime or account changed"):
        run(base, stress, root, digest)
    assert stress.book.fill_count == 0


def test_v3_frozen_prefix_is_identical_and_never_refitted_by_fee_run(
    tmp_path, monkeypatch
):
    base, stress = native_pair(frozen=True)
    assert base.recipe["schema"] == "allocation_ppo_recipe_v3"
    prefix = base.feature_preprocessing.digest
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.evaluation.rl_allocation.preprocessing as preprocessing
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    monkeypatch.setattr(
        preprocessing,
        "fit_allocation_feature_preprocessing",
        lambda *a, **k: pytest.fail("refit"),
    )
    result = run(base, stress, root, digest)
    assert (
        base.feature_preprocessing.digest
        == stress.feature_preprocessing.digest
        == prefix
    )
    assert (
        result.receipt["runtime"]["recipe"]["observation"][
            "feature_preprocessing_digest"
        ]
        == prefix
    )
    assert [call[0] for call in model.calls] == [1, 1, 1, 1]


def test_fee_none_and_observed_preserve_original_receipt_and_call_counts(
    tmp_path, monkeypatch
):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import FakeModel
    from tests.evaluation.test_allocation_global_execution import capability, declare
    from trade_rl.evaluation.rl_allocation.fee_stress_admission import (
        run_fee_stressed_global_allocation,
    )

    api = capability()
    base, _ = native_pair()
    root, digest, _ = publish_fake(base, tmp_path, monkeypatch)
    raw = archive.read_allocation_policy_manifest(
        root, expected_digest=digest, expected_recipe_digest=base.recipe_digest
    )
    snapshots = []
    for route in ("omitted", "none", "observed"):
        base, stress = native_pair()
        model = FakeModel(raw)
        loads = []
        monkeypatch.setattr(
            archive, "_load_policy", lambda path: (loads.append(path), model)[1]
        )
        folds, contract, artifacts = declaration(base, stress, root, digest)
        if route == "observed":
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
            wrapped = api.run_declared_global_allocation_execution(
                folds,
                stress,
                plan,
                artifacts=artifacts,
                base_env=base,
                contract=contract,
            )
            result = wrapped.native_result
            receipt = wrapped.receipt.payload
            assert receipt["rows"][0]["native"]["book"]["cash"] == 498
            assert receipt["rows"][0]["native"]["book"]["equity"] == 1048
            assert (
                receipt["plan"]["cell"]["original_recipe_digest"] == base.recipe_digest
            )
        else:
            result = run_fee_stressed_global_allocation(
                folds,
                base_env=base,
                stress_env=stress,
                artifacts=artifacts,
                contract=contract,
                candidate=AllocationCandidateKind.RESIDUAL_PPO,
                scenario="configured_fee_x2",
                fee_factor=2.0,
                reset_seed=7,
                **({"collector": None} if route == "none" else {}),
            )
        assert len(loads) == 2 and len(model.calls) == 4
        assert stress._transition_recorder is None
        snapshots.append(
            (
                canonical_json_bytes(result.receipt),
                canonical_json_bytes(result.walk_forward),
                [v.tobytes() for v in model.calls],
                stress.book,
                stress.order_book,
                stress.executor._rng.bit_generator.state,
            )
        )
    assert snapshots[0] == snapshots[1] == snapshots[2]
    # Immutable-parent bytes, independently captured before any B source edits.
    assert len(snapshots[0][0]) == 20359
    assert (
        sha256(snapshots[0][0]).hexdigest()
        == "8934a49781271b4e12647b00deccc06149366a1c2811c1718fdff3654bdd3a21"
    )


def test_fee_action_guard_raising_does_not_invent_public_return(tmp_path, monkeypatch):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.test_allocation_global_execution import capability, declare

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
    model.before_predict = lambda: model.weights["actor.weight"].__setitem__((0, 0), 2)
    with pytest.raises(
        ValueError, match="fee view model tensor contents changed"
    ) as raised:
        api.run_declared_global_allocation_execution(
            folds, stress, plan, artifacts=artifacts, base_env=base, contract=contract
        )
    receipt = raised.value.global_execution_receipt.payload
    assert receipt["rows"][0]["call_outcome"]["status"] == "raised"
    assert "action" not in receipt["rows"][0]["call_outcome"]
    assert receipt["rows"][0]["native"] is None and len(model.calls) == 1
    assert stress.index == 6 and stress.book.fill_count == 0
