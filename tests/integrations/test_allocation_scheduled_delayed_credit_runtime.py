"""Fixed finite synthetic actor evidence; failed gates remain failed."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from tests.evaluation.allocation_scheduled_delayed_fixture import (
    BUDGET,
    HELDOUT_ROSTER,
    ROLLOUT_FACTORS,
    SEEDS,
    TRAIN_ROSTER,
    delayed_env,
    delayed_preprocessing,
    delayed_protocol,
    delayed_schedule,
)
from tests.integrations.test_allocation_scheduled_transition_runtime import (
    independent_parameter_pins,
)
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.scheduled_training import (
    fit_allocation_ppo_schedule,
)
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace import (
    ScheduledAllocationTransitionRecorder,
)
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace_io import (
    publish_scheduled_allocation_transition_trace,
    read_scheduled_allocation_transition_trace,
)
from trade_rl.evaluation.rl_allocation.training_protocol import construct_protocol_ppo
from trade_rl.strategies.rl.allocation_artifact import (
    load_allocation_policy,
    save_allocation_policy,
)


def heldout_native(predict, *, n_steps, frozen):
    rows = []
    for day, cue in HELDOUT_ROSTER:
        env = delayed_env(day, cue, n_steps=n_steps, frozen=frozen)
        obs, _ = env.reset(seed=17)
        actions = []
        earliest = None
        for index in range(6, 22):
            code = predict(obs, env.recipe_digest)
            actions.append(code)
            obs, _, done, truncated, info = env.step(code)
            assert not truncated and done is (index == 21)
            if index == 6:
                earliest = {
                    "quantity": str(env.book.exact_quantities[0]),
                    "cost": info["execution"].interval_cost,
                    "filled_notional": info["execution"].filled_notional,
                }
        quantity = float(earliest["quantity"])
        assert env.book.as_of_index == 22
        assert str(env.book.exact_quantities[0]) == earliest["quantity"]
        assert quantity in (-4.0, 0.0, 4.0)
        if quantity:
            assert earliest["cost"] == 1.0 and earliest["filled_notional"] == 512.0
        else:
            assert earliest["cost"] == 0.0 and earliest["filled_notional"] == 0.0
        expected_profit = quantity * 128 * 0.08 * cue - earliest["cost"]
        assert env.book.portfolio_value - 1024 == pytest.approx(
            expected_profit, rel=0, abs=1e-10
        )
        rows.append(
            {
                "day": day,
                "cue": cue,
                "dataset_id": env.dataset.dataset_id,
                "actions": actions,
                "earliest": earliest,
                "alignment": int(quantity * cue > 0),
                "net_gain": (env.book.portfolio_value - 1024) / 1024,
                "cash": env.book.cash,
                "equity": env.book.portfolio_value,
                "quantity": str(env.book.exact_quantities[0]),
                "terminal_index": env.book.as_of_index,
                "recipe_digest": env.recipe_digest,
            }
        )
    return rows


def outcome(rows, initial):
    alignment = float(np.mean([row["alignment"] for row in rows]))
    gain = float(np.mean([row["net_gain"] for row in rows]))
    baseline = float(np.mean([row["net_gain"] for row in initial]))
    improvement = gain - baseline
    gates = {
        "alignment": alignment >= 0.75,
        "net_gain": gain > 0.025,
        "improvement": improvement > 0.02,
    }
    return {
        "alignment": alignment,
        "mean_net_gain": gain,
        "initial_mean_net_gain": baseline,
        "improvement": improvement,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAILED",
        "reason": None
        if all(gates.values())
        else "INITIAL_BASELINE_CEILING"
        if baseline >= 0.0190234375
        else "FIXED_LEARNABILITY_GATES",
    }


def run_fixed_actor_cells(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    frozen = delayed_preprocessing()
    results = []
    # Freeze the entire finite protocol before even constructing an initial actor.
    declarations = {str(n): delayed_protocol(n).payload() for n in ROLLOUT_FACTORS}
    schedules = {
        str(n): delayed_schedule(n, frozen=frozen).schedule.payload()
        for n in ROLLOUT_FACTORS
    }
    before_fit = {
        "schema": "scheduled_delayed_actor_invocation_v1",
        "all_cells": [[seed, n] for seed in SEEDS for n in ROLLOUT_FACTORS],
        "train_roster": list(TRAIN_ROSTER),
        "heldout_roster": list(HELDOUT_ROSTER),
        "budget": BUDGET,
        "protocols": declarations,
        "schedules": schedules,
        "frozen_preprocessing": frozen.payload(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_helper_sha256": sha256(
            Path(delayed_env.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "gates": {
            "alignment_min": 0.75,
            "mean_gain_strict_min": 0.025,
            "improvement_strict_min": 0.02,
        },
        "baseline_ceiling_threshold": 0.0190234375,
        "causal_pathway": "NOT_ESTABLISHED",
        "market_profit": "NOT_ESTABLISHED",
        "run_policy": "all_six_once_no_retry",
    }
    (root / "before-fit.json").write_bytes(canonical_json_bytes(before_fit))
    # All SIX are always attempted before any aggregate assertion. No early skip.
    for seed in SEEDS:
        for n_steps in ROLLOUT_FACTORS:
            cell = root / f"seed-{seed}-rollout-{n_steps}"
            cell.mkdir()
            declaration = delayed_protocol(n_steps)
            settings = {
                "seed": seed,
                "n_steps": n_steps,
                "budget": BUDGET,
                "protocol": declaration.payload(),
                "protocol_digest": declaration.digest,
                "frozen_preprocessing": frozen.payload(),
                "heldout_roster": list(HELDOUT_ROSTER),
                "schedule": schedules[str(n_steps)],
            }
            (cell / "settings.json").write_bytes(canonical_json_bytes(settings))
            runtime = trace = fit = None
            try:
                baseline_runtime = delayed_schedule(n_steps, frozen=frozen)
                initial_model = construct_protocol_ppo(
                    baseline_runtime, declaration, seed=seed
                )
                initial_arrays = {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in initial_model.policy.named_parameters()
                }
                initial_parameters = independent_parameter_pins(initial_arrays)
                initial_raw = {
                    name: {
                        "shape": list(value.shape),
                        "dtype": "<f4",
                        "bytes": value.tobytes().hex(),
                    }
                    for name, value in initial_arrays.items()
                }
                (cell / "initial-parameters.json").write_bytes(
                    canonical_json_bytes(initial_raw)
                )

                def initial_predict(observation, recipe):
                    action, _ = initial_model.predict(observation, deterministic=True)
                    return int(np.asarray(action).item())

                initial = heldout_native(
                    initial_predict, n_steps=n_steps, frozen=frozen
                )
                (cell / "initial.json").write_bytes(canonical_json_bytes(initial))
                initial_model = baseline_runtime = None
                runtime = delayed_schedule(n_steps, frozen=frozen)
                trace = ScheduledAllocationTransitionRecorder(
                    runtime, learner_diagnostics="native_ppo_update_v1"
                )
                fit = fit_allocation_ppo_schedule(
                    runtime,
                    total_timesteps=BUDGET,
                    seed=seed,
                    training_protocol=declaration,
                    transition_trace=trace,
                )
                policy = fit.inference_policy()
                bundle = cell / "policy"
                bundle_pin = save_allocation_policy(bundle, policy)
                sidecar = cell / "actor-trace"
                trace_pin = publish_scheduled_allocation_transition_trace(
                    trace,
                    sidecar,
                    bundle_root=bundle,
                    expected_bundle_digest=bundle_pin,
                )
                manifest = read_scheduled_allocation_transition_trace(
                    sidecar,
                    expected_digest=trace_pin,
                    bundle_root=bundle,
                    expected_bundle_digest=bundle_pin,
                    require_learner_diagnostics=True,
                )
                trained = heldout_native(
                    lambda obs, recipe: policy.action(
                        obs, runtime_recipe_digest=recipe
                    ),
                    n_steps=n_steps,
                    frozen=frozen,
                )
                loaded = load_allocation_policy(
                    bundle,
                    expected_digest=bundle_pin,
                    expected_recipe_digest=runtime.template_env.recipe_digest,
                )
                reloaded = heldout_native(
                    lambda obs, recipe: loaded.action(
                        obs, runtime_recipe_digest=recipe
                    ),
                    n_steps=n_steps,
                    frozen=frozen,
                )
                assert trained == reloaded
                events = [json.loads(row) for row in trace.events]
                updates = [row for row in events if row["kind"] == "update"]
                assert updates[0]["parameters_before"] == initial_parameters
                assert fit.receipt["actual_timesteps"] == BUDGET
                early = []
                boundaries = {
                    row["rollout"]: row for row in events if row["kind"] == "rollout"
                }
                for row in events:
                    if row["kind"] == "transition" and row["old"]["index"] == 6:
                        boundary = boundaries[row["rollout"]]
                        advantage = np.frombuffer(
                            bytes.fromhex(boundary["advantages"]), dtype="<f4"
                        ).reshape(n_steps, 1)[row["local"], 0]
                        target = np.frombuffer(
                            bytes.fromhex(boundary["returns"]), dtype="<f4"
                        ).reshape(n_steps, 1)[row["local"], 0]
                        early.append(
                            {
                                "sequence": row["sequence"],
                                "window_id": row["old"]["window_id"],
                                "episode": row["old"]["episode"],
                                "raw_action": row["actor"]["action_code"],
                                "approved": row["facts"]["risk"]["weights"],
                                "filled_quantity": row["facts"]["book"][
                                    "exact_quantities"
                                ],
                                "entry_cost": row["facts"]["execution"][
                                    "interval_cost"
                                ],
                                "raw_advantage": float(advantage),
                                "raw_return": float(target),
                                "actor_value": row["actor"]["value"],
                                "collector_reward": row["actor"]["reward"],
                                "log_prob": row["actor"]["log_prob"],
                                "boundary_done": boundary["done"],
                                "bootstrap_index": boundary["current"]["index"],
                                "bootstrap_value": boundary["bootstrap"]["value"],
                            }
                        )
                result = {
                    "seed": seed,
                    "n_steps": n_steps,
                    "status": "VALID",
                    "settings_digest": content_digest(settings),
                    "initial_parameters": initial_parameters,
                    "initial": initial,
                    "trained": trained,
                    "reloaded": reloaded,
                    "learning": outcome(trained, initial),
                    "fit_receipt": fit.receipt,
                    "bundle_digest": bundle_pin,
                    "trace_digest": trace_pin,
                    "trace_input_summary": manifest["input_summary"],
                    "early_rows": early,
                    "native_update_count": len(updates),
                    "successful_optimizer_steps": sum(
                        u["optimizer_steps"] for u in updates
                    ),
                    "actor_parameter_changes": sum(
                        u["parameters_before"]["actor"]
                        != u["parameters_after"]["actor"]
                        for u in updates
                    ),
                    "critic_parameter_changes": sum(
                        u["parameters_before"]["critic"]
                        != u["parameters_after"]["critic"]
                        for u in updates
                    ),
                }
                (cell / "result.json").write_bytes(canonical_json_bytes(result))
                results.append(result)
                fit = policy = loaded = runtime = trace = None
                events = updates = boundaries = None
            except Exception as error:
                if trace is not None:
                    (cell / "failed-partial-events.jsonl").write_bytes(
                        b"".join(row + b"\n" for row in trace.events)
                    )
                if fit is not None:
                    (cell / "failed-fit-receipt.json").write_bytes(
                        canonical_json_bytes(fit.receipt)
                    )
                failure = {
                    "seed": seed,
                    "n_steps": n_steps,
                    "status": "INVALID",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "settings_digest": content_digest(settings),
                }
                (cell / "result.json").write_bytes(canonical_json_bytes(failure))
                results.append(failure)
    return persist_comparison(root, results, before_fit)


def persist_comparison(root, results, before_fit):
    checks = []

    def require(name, condition, *, seed=None, n_steps=None):
        checks.append(
            {"check": name, "passed": bool(condition), "seed": seed, "n_steps": n_steps}
        )

    require(
        "exact_six_cell_roster",
        [(row["seed"], row["n_steps"]) for row in results]
        == [(seed, n) for seed in SEEDS for n in ROLLOUT_FACTORS],
    )
    for row in results:
        seed, n_steps = row["seed"], row["n_steps"]
        valid = row["status"] == "VALID"
        require("cell_valid", valid, seed=seed, n_steps=n_steps)
        if not valid:
            continue
        conditions = {
            "fixed_returned_optimizer_calls": row["successful_optimizer_steps"]
            == 10240,
            "fixed_realized_budget": row["fit_receipt"]["actual_timesteps"] == BUDGET,
            "actual_actor_changes": row["actor_parameter_changes"] > 0,
            "actual_critic_changes": row["critic_parameter_changes"] > 0,
            "paid_early_row": any(
                early["entry_cost"] == 1.0 for early in row["early_rows"]
            ),
            "four_heldout_rows": len(row["trained"]) == 4,
            "frozen_learning_outcome": row["learning"]
            == outcome(row["trained"], row["initial"]),
        }
        for name, condition in conditions.items():
            require(name, condition, seed=seed, n_steps=n_steps)
    for seed in SEEDS:
        paired = [row for row in results if row["seed"] == seed]
        available = len(paired) == 2 and all(row["status"] == "VALID" for row in paired)
        require("paired_initial_available", available, seed=seed)
        if not available:
            continue
        require(
            "paired_initial_parameters",
            paired[0]["initial_parameters"] == paired[1]["initial_parameters"],
            seed=seed,
        )
        same_native = len(paired[0]["initial"]) == len(
            paired[1]["initial"]
        ) == 4 and all(
            all(
                left[name] == right[name]
                for name in (
                    "dataset_id",
                    "actions",
                    "earliest",
                    "alignment",
                    "net_gain",
                    "cash",
                    "equity",
                    "quantity",
                )
            )
            for left, right in zip(
                paired[0]["initial"], paired[1]["initial"], strict=True
            )
        )
        require("paired_initial_native_account", same_native, seed=seed)
    failures = [check for check in checks if not check["passed"]]
    software = "INVALID" if failures else "VALID"
    short = [row for row in results if row["n_steps"] == 4]
    report = {
        "schema": "scheduled_delayed_actor_fixed_fixture_v1",
        "scope": "synthetic_beyond_rollout_trainability_only",
        "causal_through_bootstrap": "NOT_ESTABLISHED",
        "market_profit": "NOT_ESTABLISHED",
        "all_six_attempted": len(results) == 6,
        "software_evidence": software,
        "comparison_validity": software,
        "software_checks": checks,
        "software_failures": failures,
        "before_fit_digest": content_digest(before_fit),
        "short_learning": "PASS"
        if software == "VALID"
        and all(
            row["status"] == "VALID" and row["learning"]["status"] == "PASS"
            for row in short
        )
        else "FAILED",
        "results": results,
    }
    (root / "comparison.json").write_bytes(canonical_json_bytes(report))
    return report


def test_outcome_discriminator_preserves_baseline_ceiling_and_failed_learning():
    correct = [{"alignment": 1, "net_gain": 0.0390234375}] * 4
    high_initial = [{"alignment": 1, "net_gain": 0.0390234375}] * 2 + [
        {"alignment": 0, "net_gain": 0}
    ] * 2
    negative_initial = [{"alignment": 0, "net_gain": -0.0409765625}] * 4
    ceiling = outcome(correct, high_initial)
    assert (
        ceiling["status"] == "FAILED"
        and ceiling["reason"] == "INITIAL_BASELINE_CEILING"
    )
    assert ceiling["gates"] == {
        "alignment": True,
        "net_gain": True,
        "improvement": False,
    }
    assert outcome(correct, negative_initial)["status"] == "PASS"
    assert (
        outcome([{"alignment": 0, "net_gain": 0}] * 4, negative_initial)["status"]
        == "FAILED"
    )


def test_actual_fixed_six_cell_delayed_actor_evidence_retains_all_outcomes(tmp_path):
    report = run_fixed_actor_cells(tmp_path / "fixed-delayed-actor")
    assert [(row["seed"], row["n_steps"]) for row in report["results"]] == [
        (seed, n) for seed in SEEDS for n in ROLLOUT_FACTORS
    ]
    assert report["all_six_attempted"]
    assert report["software_evidence"] == report["comparison_validity"] == "VALID", (
        report["software_failures"]
    )
    assert all(row["status"] == "VALID" for row in report["results"]), report
    for row in report["results"]:
        assert row["learning"] == outcome(row["trained"], row["initial"])
        assert row["successful_optimizer_steps"] == 10240
        assert row["fit_receipt"]["actual_timesteps"] == 4096
        assert (
            row["actor_parameter_changes"] > 0 and row["critic_parameter_changes"] > 0
        )
        assert any(early["entry_cost"] == 1.0 for early in row["early_rows"])
        assert len(row["trained"]) == 4
    for seed in SEEDS:
        paired = [row for row in report["results"] if row["seed"] == seed]
        assert paired[0]["initial_parameters"] == paired[1]["initial_parameters"]
        for left, right in zip(paired[0]["initial"], paired[1]["initial"], strict=True):
            for name in (
                "dataset_id",
                "actions",
                "earliest",
                "alignment",
                "net_gain",
                "cash",
                "equity",
                "quantity",
            ):
                assert left[name] == right[name]
    assert report["short_learning"] == (
        "PASS"
        if all(
            row["learning"]["status"] == "PASS"
            for row in report["results"]
            if row["n_steps"] == 4
        )
        else "FAILED"
    )
    print(
        json.dumps(
            {
                "fixed_actor_evidence": str(tmp_path / "fixed-delayed-actor"),
                "short_learning": report["short_learning"],
                "cells": [
                    {
                        "seed": row["seed"],
                        "rollout": row["n_steps"],
                        "learning": row["learning"],
                    }
                    for row in report["results"]
                ],
            },
            sort_keys=True,
        )
    )


def test_fixed_actor_invocation_never_reuses_existing_output_or_starts_models(
    monkeypatch, tmp_path
):
    import sys

    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "kept"
    marker.write_bytes(b"original")

    def forbidden(*args, **kwargs):
        pytest.fail("existing output must reject before initial model construction")

    monkeypatch.setattr(sys.modules[__name__], "construct_protocol_ppo", forbidden)
    with pytest.raises(FileExistsError):
        run_fixed_actor_cells(root)
    assert marker.read_bytes() == b"original" and list(root.iterdir()) == [marker]


def test_all_six_mock_initial_failures_persist_without_skips_or_retry(
    monkeypatch, tmp_path
):
    import sys

    attempts = []

    def unavailable(env, declaration, *, seed):
        attempts.append((seed, declaration.n_steps))
        raise RuntimeError("fixed initial construction injected failure")

    monkeypatch.setattr(sys.modules[__name__], "construct_protocol_ppo", unavailable)
    root = tmp_path / "all-failures"
    report = run_fixed_actor_cells(root)
    assert attempts == [(seed, n) for seed in SEEDS for n in ROLLOUT_FACTORS]
    assert report["all_six_attempted"] and report["software_evidence"] == "INVALID"
    assert report["short_learning"] == "FAILED"
    assert all(
        row["status"] == "INVALID" and row["error_type"] == "RuntimeError"
        for row in report["results"]
    )
    assert all(
        (root / f"seed-{seed}-rollout-{n}" / "result.json").is_file()
        for seed, n in attempts
    )
    assert (root / "before-fit.json").is_file() and (root / "comparison.json").is_file()


def classifier_fixture_results():
    from copy import deepcopy

    initial = [
        {
            "alignment": 0,
            "net_gain": -0.0409765625,
            "dataset_id": str(i),
            "actions": [1],
            "earliest": {"quantity": "-4", "cost": 1.0},
            "cash": 1535.0,
            "equity": 982.04,
            "quantity": "-4",
        }
        for i in range(4)
    ]
    trained = [dict(row, alignment=1, net_gain=0.0390234375) for row in initial]
    return [
        {
            "seed": seed,
            "n_steps": n,
            "status": "VALID",
            "initial_parameters": {"actor": "a" * 64, "critic": "b" * 64},
            "initial": deepcopy(initial),
            "trained": deepcopy(trained),
            "learning": outcome(trained, initial),
            "successful_optimizer_steps": 10240,
            "fit_receipt": {"actual_timesteps": 4096},
            "actor_parameter_changes": 1,
            "critic_parameter_changes": 1,
            "early_rows": [{"entry_cost": 1.0}],
        }
        for seed in SEEDS
        for n in ROLLOUT_FACTORS
    ]


@pytest.mark.parametrize("mutation", ["wrong_initial_pair", "wrong_returned_count"])
def test_persisted_aggregate_classifies_invalid_conformance_before_valid_claim(
    tmp_path, mutation
):
    results = classifier_fixture_results()
    if mutation == "wrong_initial_pair":
        results[1]["initial_parameters"]["actor"] = "f" * 64
    else:
        results[0]["successful_optimizer_steps"] -= 1
    report = persist_comparison(tmp_path, results, {"no_models": True})
    persisted = json.loads((tmp_path / "comparison.json").read_bytes())
    assert persisted == report and report["results"] == results and len(results) == 6
    assert report["software_evidence"] == "INVALID"
    assert (
        report["comparison_validity"] == "INVALID"
        and report["short_learning"] != "PASS"
    )
    assert report["software_failures"]
    assert all(row["learning"]["status"] == "PASS" for row in report["results"])


def test_persisted_valid_software_keeps_failed_short_learning_separate(tmp_path):
    results = classifier_fixture_results()
    results[0]["trained"] = [
        dict(row, alignment=0, net_gain=0.0) for row in results[0]["trained"]
    ]
    results[0]["learning"] = outcome(results[0]["trained"], results[0]["initial"])
    report = persist_comparison(tmp_path, results, {"no_models": True})
    assert report["software_evidence"] == report["comparison_validity"] == "VALID"
    assert report["software_failures"] == [] and len(report["results"]) == 6
    assert (
        report["short_learning"] == "FAILED"
        and results[0]["learning"]["status"] == "FAILED"
    )
    assert json.loads((tmp_path / "comparison.json").read_bytes()) == report
