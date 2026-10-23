"""Fixed genuine learning evidence; conformance and learning have separate gates."""

import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests.integrations import settled_learnability_fixture as fixture
from trade_rl.artifacts import content_digest
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)

PROTOCOL = dict(
    n_steps=32,
    batch_size=32,
    n_epochs=10,
    gamma=1.0,
    gae_lambda=0.95,
    learning_rate=0.002,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    pi_layers=(32, 32),
    vf_layers=(32, 32),
    adam_betas=(0.9, 0.999),
    adam_eps=1e-5,
    adam_weight_decay=0.0,
    adam_amsgrad=False,
)
CASES = {
    "signal": {"price_response": 0.02, "fee": 0.0005},
    "fee_only": {"price_response": 0.0, "fee": 0.005},
}
SLOTS = [(seed, case) for seed in (0, 7, 17) for case in ("signal", "fee_only")]


def parameters(policy):
    """Independent literal serialization, including any registered buffers."""
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "order": "C",
            "hex": array.tobytes(order="C").hex(),
        }
        for name, tensor in sorted(policy.state_dict().items())
        for array in (tensor.detach().cpu().numpy().copy(),)
    }


def evaluate(case, actor, *, cue_transform=1, storage=None, key="evaluation"):
    rows = [{"signs": list(signs), "status": "NOT_RUN"} for signs in fixture.PATTERNS]
    result = {
        "patterns": rows,
        "mean_gain": None,
        "mean_alignment": None,
    }
    if storage is not None:
        storage[key] = result
    for row in rows:
        try:
            env = fixture.environment(
                row["signs"], cue_transform=cue_transform, **CASES[case]
            )
            fixture.run(
                env, lambda observation: actor(observation, env), diagnostic=row
            )
        except BaseException as error:
            row["status"] = "INVALID"
            row["error"] = fixture.error_summary(error)
            raise
    result["mean_gain"] = float(np.mean([row["gain"] for row in rows]))
    result["mean_alignment"] = float(np.mean([row["alignment"] for row in rows]))
    return result


def probabilities(model, observations):
    """Read categorical masses on identical inputs with no state/RNG change."""
    import torch

    before = parameters(model.policy)
    rng_before = torch.get_rng_state().clone()
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    try:
        assert not model.policy.training and model.device.type == "cpu"
        with torch.no_grad():
            assert not torch.is_grad_enabled()
            distribution = model.policy.get_distribution(
                torch.as_tensor(observations, dtype=torch.float32)
            )
            result = distribution.distribution.probs.detach().cpu().numpy().copy()
    finally:
        model.policy.set_training_mode(was_training)
    assert parameters(model.policy) == before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert result.shape == (64, 4) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
    return result


def controls(case, *, storage):
    for key, actor in (
        ("cash", lambda _observation, _env: 0),
        ("cue_rule", lambda observation, _env: fixture.cue_rule(observation)),
        ("long_hold", lambda _observation, env: 3 if env.index == 6 else 0),
        ("short_hold", lambda _observation, env: 1 if env.index == 6 else 0),
    ):
        evaluate(case, actor, storage=storage, key=key)


def learning_criteria(case, report):
    initial, final = report["initial"], report["trained"]
    if case == "fee_only":
        return {
            "zero_effective_trading": all(
                row["gain"] == 0.0 and row["fill_count"] == 0
                for row in final["patterns"]
            ),
            "learned_entry_probability_reduction": report["entry_probability_reduction"]
            >= 0.20,
        }
    return {
        "positive_after_cost_gain": final["mean_gain"] > 0.02,
        "improvement_over_initialization": final["mean_gain"] - initial["mean_gain"]
        > 0.015,
        "effective_delayed_direction": final["mean_alignment"] >= 0.75,
        "removed_cue_degradation": final["mean_gain"]
        - report["cue_removed"]["mean_gain"]
        >= 0.015,
        "inverted_cue_nonpositive": report["cue_inverted"]["mean_gain"] <= 0.0,
    }


def verify_group(group, *, case="signal", cue_transform=1):
    """Recompute gains/direction from actual native rows, not reported averages."""
    assert [row["signs"] for row in group["patterns"]] == [
        list(signs) for signs in fixture.PATTERNS
    ]
    gains, alignments = [], []
    for row in group["patterns"]:
        assert row["status"] == "VALID"
        expected = fixture.environment(
            row["signs"], cue_transform=cue_transform, **CASES[case]
        )
        transitions = row["transitions"]
        assert len(transitions) == 16 and len(row["actor_rewards"]) == 16
        physical = [
            native for transition in transitions for native in transition["rows"]
        ]
        assert [native["processing_index"] for native in physical] == list(range(7, 25))
        assert [native["kind"] for native in physical] == ["actor"] * 16 + [
            "reserve"
        ] * 2
        equity, peak, drawdown, fills, directions = 1000.0, 1000.0, 0.0, 0, []
        for index, transition in enumerate(transitions):
            assert transition["schema"] == "allocation_settled_transition_v1"
            assert (
                transition["episode_id"] == 1
                and transition["account_id"] == expected.account_id
            )
            assert transition["bound_digest"] == expected.bound.digest
            assert transition["recipe_digest"] == expected.recipe_digest
            assert transition["source_envelope_digest"] == expected._source_digest
            assert len(transition["rows"]) == (3 if index == 15 else 1)
            assert transition["rows"][0]["kind"] == "actor"
            assert transition["rows"][0]["start_index"] == 6 + index
            assert transition["actor_step"] == index
            assert transition["actor_index"] == 6 + index
            assert transition["initial_capital"] == 1000.0
            assert not transition["truncated"]
            assert transition["terminated"] is (index == 15)
            for native in transition["rows"]:
                assert native["clock_valid"] is True
                assert native["returned_clock_types"] == {
                    "next_index": "int",
                    "bars_advanced": "int",
                }
                execution = native["native"]["execution"]
                assert (
                    type(execution["next_index"]) is int
                    and type(execution["bars_advanced"]) is int
                )
                assert (
                    execution["next_index"]
                    == native["processing_index"]
                    == native["start_index"] + 1
                )
                assert execution["bars_advanced"] == 1
                book = native["native"]["book"]
                assert len(book["quantities"]) == len(book["exact_quantities"]) == 1
                assert book["as_of_index"] == native["processing_index"]
                assert book["as_of_dataset_id"] == expected.dataset.dataset_id
                assert (
                    book["mark_prices"]
                    == expected.dataset.mark_price[native["processing_index"]].tolist()
                )
                assert book["termination_reason"] is None and not book["insolvent"]
                assert (
                    np.isfinite(book["max_drawdown"])
                    and 0 <= book["max_drawdown"] <= 0.2
                )
                assert native["opening_equity"] == equity
                equity = book["equity"]
                assert np.isfinite(equity)
                assert np.isfinite(book["peak_value"]) and book["peak_value"] >= peak
                assert book["peak_value"] >= equity and book["peak_value"] > 0
                peak = book["peak_value"]
                assert book["max_drawdown"] >= drawdown
                assert book["max_drawdown"] + 1e-12 >= max(0, 1 - max(equity, 0) / peak)
                drawdown = book["max_drawdown"]
                assert (
                    type(execution["fill_count"]) is int
                    and execution["fill_count"] >= 0
                )
                fills += execution["fill_count"]
                assert book["fill_count"] == fills
                if native["processing_index"] in (9, 13, 17, 21):
                    sign = row["signs"][(native["processing_index"] - 9) // 4]
                    directions.append(
                        int(sign * Fraction(book["exact_quantities"][0]) > 0)
                    )
            previous = (
                1000.0
                if index == 0
                else transitions[index - 1]["rows"][-1]["native"]["book"]["equity"]
            )
            assert abs(transition["reward"] - (equity - previous) / 1000.0) < 1e-12
            assert transition["reward"] == row["actor_rewards"][index]
        witness = row["witness"]
        assert witness == transitions[-1]["settlement_witness"]
        assert (
            witness["settled"]
            is witness["eligible"]
            is witness["full_coverage"]
            is True
        )
        assert witness["physical_steps_processed"] == 18
        assert (
            witness["actor_steps_processed"] == 16
            and witness["reserve_steps_processed"] == 2
        )
        assert witness["frontier_index"] == 24 and witness["maximum_drawdown"] <= 0.2
        assert witness["maximum_drawdown"] == drawdown
        terminal = witness["native_frontier"]
        assert terminal["book"] == physical[-1]["native"]["book"]
        assert terminal["active_orders"] == physical[-1]["native"]["active_orders"]
        assert terminal["book"]["termination_reason"] is None
        assert all(Fraction(q) == 0 for q in terminal["book"]["exact_quantities"])
        assert terminal["book"]["quantities"] == [0.0]
        assert terminal["book"]["cash"] == equity
        assert not terminal["active_orders"]
        gain = (equity - 1000.0) / 1000.0
        assert row["gain"] == gain
        assert abs(sum(row["actor_rewards"]) - gain) < 1e-12
        assert row["fill_count"] == terminal["book"]["fill_count"]
        # A fee-only path has no payoff; the captured effective direction is zero.
        alignment = sum(directions) / 4 if CASES[case]["price_response"] != 0 else 0.0
        assert row["alignment"] == alignment
        gains.append(gain)
        alignments.append(alignment)
    assert group["mean_gain"] == float(np.mean(gains))
    assert group["mean_alignment"] == float(np.mean(alignments))


def verify_slot(path, report):
    """Independently close persisted cells, source, trace and artifact evidence."""
    assert report["software_conformance"] == "VALID"
    assert report["case_parameters"] == CASES[report["case"]]
    protocol = AllocationPPOTrainingProtocol(**PROTOCOL)
    assert (
        report["protocol"] == protocol.payload()
        and report["protocol_digest"] == protocol.digest
    )
    assert report["fixture_source"] == Path(fixture.__file__).read_bytes().hex()
    assert report["test_source"] == Path(__file__).read_bytes().hex()
    for name in ("initial", "trained", "cue_removed", "cue_inverted", "reload"):
        verify_group(
            report[name],
            case=report["case"],
            cue_transform={"cue_removed": 0, "cue_inverted": -1}.get(name, 1),
        )
    for group in report["controls"].values():
        verify_group(group, case=report["case"])
    assert set(report["controls"]) == {"cash", "cue_rule", "long_hold", "short_hold"}
    assert report["reload"] == report["trained"]
    if report["case"] == "fee_only":
        before, after = (
            np.asarray(report["initial_entry_probabilities"]),
            np.asarray(report["final_entry_probabilities"]),
        )
        assert before.shape == after.shape == (64,)
        assert np.isfinite(before).all() and np.isfinite(after).all()
        assert ((before >= 0) & (before <= 1)).all() and (
            (after >= 0) & (after <= 1)
        ).all()
        assert report["entry_probability_reduction"] == float(
            before.mean() - after.mean()
        )
    criteria = learning_criteria(report["case"], report)
    assert report["criteria"] == criteria
    assert report["learning_status"] == ("PASS" if all(criteria.values()) else "FAILED")
    training = report["training_manifest"]["training"]
    assert (
        training["actual_timesteps"]
        == training["requested_timesteps"]
        == report["requested_timesteps"]
        == 4096
    )
    assert training["optimization"]["successful_optimizer_step_calls"] == 1280
    from trade_rl.evaluation.rl_allocation.settled_transition_validation import (
        validate_settled_transition_capture,
    )
    from trade_rl.strategies.rl.allocation_artifact import (
        read_allocation_policy_manifest,
    )

    env = fixture.environment(**CASES[report["case"]])
    with gzip.open(
        path.parent / "actual-capture.json.gz", "rt", encoding="utf-8"
    ) as file:
        capture = json.load(file)
    projection = validate_settled_transition_capture(
        env,
        capture,
        requested_timesteps=4096,
        optimization=training["optimization"],
        training_protocol=protocol,
    )
    assert all(training[key] == value for key, value in projection.items())
    assert training["seed"] == report["seed"]
    assert (
        training["protocol"] == protocol.payload()
        and training["protocol_digest"] == protocol.digest
    )
    manifest = read_allocation_policy_manifest(
        path.parent / "bundle",
        expected_digest=report["bundle_digest"],
        expected_recipe_digest=env.recipe_digest,
    )
    assert manifest["training"] == training
    assert (
        hashlib.sha256((path.parent / "bundle" / "policy.zip").read_bytes()).hexdigest()
        == manifest["policy_sha256"]
    )
    kinds = {}
    for event in capture["events"]:
        kind = event["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "actor_forward" and kinds[kind] == 1:
            assert event["parameters_digest"] == report["initial_parameters_digest"]
        if kind == "rollout" and kinds[kind] == 1:
            assert (
                content_digest(event["parameters_before"])
                == report["initial_parameters_digest"]
            )
        if kind == "training_end":
            assert event["actual_timesteps"] == 4096
    assert (
        kinds.get("failure", 0) == 0
        and kinds["training_start"] == kinds["training_end"] == 1
    )
    assert kinds["append"] == 4096 and kinds["rollout"] == kinds["update"] == 128
    return {
        str(file.relative_to(path.parent)): hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        for file in sorted(path.parent.rglob("*"))
        if file.is_file()
    }


def aggregate_slots(paths):
    """Exact six-slot registry; missing/failed/duplicate evidence never qualifies."""
    result = {
        "schema": "settled_synthetic_learnability_aggregate_v1",
        "slots": [
            {"seed": seed, "case": case, "status": "NOT_RUN"} for seed, case in SLOTS
        ],
        "errors": [],
        "software_conformance": "INCOMPLETE",
        "learning_status": "NOT_ESTABLISHED",
        "market_profit": "NOT_ESTABLISHED",
        "formal_approval": "NOT_ESTABLISHED",
    }
    seen = set()
    last_slot = -1
    for path in paths:
        try:
            report = json.loads(path.read_bytes())
            key = (report["seed"], report["case"])
            assert type(report["seed"]) is int
            assert key in SLOTS and key not in seen, "unexpected or duplicate slot"
            assert SLOTS.index(key) > last_slot, "frozen invocation order differs"
            last_slot = SLOTS.index(key)
            seen.add(key)
            row = result["slots"][SLOTS.index(key)]
            row["status"] = "INVALID"
            row["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["closure"] = verify_slot(path, report)
            row.update(
                status="VALID",
                learning_status=report["learning_status"],
                criteria=report["criteria"],
            )
        except BaseException as error:
            result["errors"].append({"path": str(path), **fixture.error_summary(error)})
    if result["errors"]:
        result["software_conformance"] = "INVALID"
    elif seen == set(SLOTS):
        result["software_conformance"] = "VALID"
        result["learning_status"] = (
            "PASS"
            if all(row["learning_status"] == "PASS" for row in result["slots"])
            else "FAILED"
        )
    return result


@pytest.fixture(scope="module")
def slot_registry(tmp_path_factory):
    paths = []
    root = tmp_path_factory.mktemp("fixed-settled-learnability")
    destination = root / "aggregate.json"
    destination.write_text(
        json.dumps(aggregate_slots([]), indent=2) + "\n", encoding="utf-8"
    )
    yield paths
    aggregate = aggregate_slots(paths)
    destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    assert aggregate["software_conformance"] != "INVALID", aggregate["errors"]
    if len(paths) == len(SLOTS):
        assert aggregate["software_conformance"] == "VALID", aggregate["errors"]


def persist(root, report, trace, *, primary_error=None):
    """Preserve every complete/failed slot, never an optimistic partial status."""
    failures = []

    def record(error):
        failures.append(error)
        report["software_conformance"] = "INVALID"
        report["learning_status"] = "NOT_ESTABLISHED"
        report.setdefault("persistence_errors", []).append(fixture.error_summary(error))

    try:
        if trace is not None:
            with gzip.open(
                root / "actual-capture.json.gz", "wt", encoding="utf-8"
            ) as file:
                json.dump(trace.payload(), file)
    except BaseException as error:
        record(error)
    try:
        (root / "learnability-evidence.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException as error:
        record(error)
    if failures:
        if primary_error is not None:
            for error in failures:
                fixture.note_failure(
                    primary_error, "Evidence persistence failed", error
                )
        else:
            raise failures[0]


@pytest.mark.parametrize("seed,case", SLOTS)
def test_fixed_settled_learner_records_every_frozen_slot(
    seed, case, tmp_path, slot_registry
):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from trade_rl.evaluation.rl_allocation.settled_inference import (
        load_settled_allocation_policy,
    )
    from trade_rl.evaluation.rl_allocation.settled_training import (
        build_settled_allocation_ppo,
        fit_settled_allocation_ppo,
    )
    from trade_rl.evaluation.rl_allocation.settled_transition_trace import (
        SettledTransitionRecorder,
    )
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    root = tmp_path / "fixed-settled-learnability"
    root.mkdir()
    slot_registry.append(root / "learnability-evidence.json")
    protocol = AllocationPPOTrainingProtocol(**PROTOCOL)
    report = {
        "schema": "settled_synthetic_learnability_evidence_v1",
        "seed": seed,
        "case": case,
        "case_parameters": CASES[case],
        "protocol": protocol.payload(),
        "protocol_digest": protocol.digest,
        "requested_timesteps": 4096,
        "software_conformance": "INCOMPLETE",
        "learning_status": "NOT_ESTABLISHED",
        "market_profit": "NOT_ESTABLISHED",
        "formal_approval": "NOT_ESTABLISHED",
        "factorial_is_independent_market_sample": False,
        "fixture_source": Path(fixture.__file__).read_bytes().hex(),
        "test_source": Path(__file__).read_bytes().hex(),
    }
    trace = None
    primary_error = None
    try:
        initial = build_settled_allocation_ppo(
            fixture.environment(**CASES[case]), seed=seed, training_protocol=protocol
        )
        initial_parameters = parameters(initial.policy)
        report["initial_parameters_digest"] = content_digest(initial_parameters)
        evaluate(
            case,
            lambda observation, _env: int(
                initial.predict(observation, deterministic=True)[0]
            ),
            storage=report,
            key="initial",
        )
        report["controls"] = {}
        controls(case, storage=report["controls"])
        assert all(
            row["gain"] == 0.0 and row["fill_count"] == 0
            for row in report["controls"]["cash"]["patterns"]
        )
        cash_observations = np.asarray(
            [
                obs
                for row in report["controls"]["cash"]["patterns"]
                for obs in row["entry_observations"]
            ],
            dtype=np.float32,
        )
        if case == "fee_only":
            report["initial_entry_probabilities"] = (
                probabilities(initial, cash_observations)[:, [1, 3]]
                .sum(axis=1)
                .tolist()
            )
        env = fixture.environment(**CASES[case])
        trace = SettledTransitionRecorder(env)
        trained = fit_settled_allocation_ppo(
            env,
            total_timesteps=4096,
            seed=seed,
            training_protocol=protocol,
            transition_trace=trace,
        )
        assert trace.complete
        events = [json.loads(row) for row in trace.events]
        first_forward = next(
            event for event in events if event["kind"] == "actor_forward"
        )
        first_rollout = next(event for event in events if event["kind"] == "rollout")
        assert first_forward["parameters_digest"] == content_digest(initial_parameters)
        assert first_rollout["parameters_before"] == initial_parameters
        receipt = trained.manifest["training"]
        assert receipt["actual_timesteps"] == receipt["requested_timesteps"] == 4096
        assert receipt["optimization"]["successful_optimizer_step_calls"] == 1280
        assert parameters(trained.model.policy) != initial_parameters
        report["training_manifest"] = trained.manifest
        evaluate(
            case,
            lambda observation, runtime: trained.action(
                observation, runtime_recipe_digest=runtime.recipe_digest
            ),
            storage=report,
            key="trained",
        )
        evaluate(
            case,
            lambda observation, runtime: trained.action(
                observation, runtime_recipe_digest=runtime.recipe_digest
            ),
            cue_transform=0,
            storage=report,
            key="cue_removed",
        )
        evaluate(
            case,
            lambda observation, runtime: trained.action(
                observation, runtime_recipe_digest=runtime.recipe_digest
            ),
            cue_transform=-1,
            storage=report,
            key="cue_inverted",
        )
        if case == "fee_only":
            report["final_entry_probabilities"] = (
                probabilities(trained.model, cash_observations)[:, [1, 3]]
                .sum(axis=1)
                .tolist()
            )
            report["entry_probability_reduction"] = float(
                np.mean(report["initial_entry_probabilities"])
                - np.mean(report["final_entry_probabilities"])
            )
        digest = save_allocation_policy(root / "bundle", trained)
        loaded = load_settled_allocation_policy(
            root / "bundle",
            env=fixture.environment(**CASES[case]),
            expected_digest=digest,
        )
        report["bundle_digest"] = digest
        evaluate(
            case,
            lambda observation, runtime: loaded.action(
                observation, runtime_recipe_digest=runtime.recipe_digest
            ),
            storage=report,
            key="reload",
        )
        assert report["reload"] == report["trained"]
        report["criteria"] = learning_criteria(case, report)
        report["learning_status"] = (
            "PASS" if all(report["criteria"].values()) else "FAILED"
        )
        report["software_conformance"] = "VALID"
    except BaseException as error:
        primary_error = error
        report["software_conformance"] = "INVALID"
        report["learning_status"] = "NOT_ESTABLISHED"
        report["error"] = fixture.error_summary(error)
        raise
    finally:
        persist(root, report, trace, primary_error=primary_error)
    # Learning failure is a retained falsifier, not a software pass assertion.
    assert report["software_conformance"] == "VALID"
    assert report["learning_status"] in {"PASS", "FAILED"}
