"""Actual SB3 consumption of the declared chronological allocation schedule."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
    allocation_training_window,
)
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)


def env_for(start: int, stop: int) -> AllocationTradingEnv:
    args = parameters(stop=10)
    dataset = args["dataset"]
    args["start_index"], args["stop_index"] = start, stop
    args["estimates"] = tuple(
        estimate
        for estimate in args["estimates"]
        if dataset.timestamps[start]
        <= estimate.decision_time
        < dataset.timestamps[stop]
    )
    clock = replace(
        args["bound"].clock,
        economic_horizon_seconds=(stop - start) * 3600,
        rollout_steps=4,
    )
    objective = args["bound"].objective
    profile = AllocationRuntimeProfile(
        objective.economics_digest,
        objective.risk_digest,
        objective.capital.initial_equities[0],
        objective.capital.currency,
        clock.decision_interval_seconds,
        clock.economic_horizon_seconds,
    )
    recipe = allocation_recipe_digest(
        args["action_contract"],
        ("signal",),
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=profile,
    )

    def utc(index: int) -> datetime:
        stamp = dataset.timestamps[index].astype("datetime64[us]").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    args["bound"] = BoundObjectiveClock(
        replace(
            objective,
            evaluation_start=utc(start),
            evaluation_stop_exclusive=utc(stop),
            deployment_recipe_digest=recipe,
        ),
        clock,
    )
    args["account_id"] = f"scheduled-fit-{start}-{stop}"
    return AllocationTradingEnv(**opt_in(args))


def scheduled() -> tuple[
    AllocationTrainingScheduleEnv, tuple[AllocationTradingEnv, ...]
]:
    first, second = env_for(6, 8), env_for(8, 10)
    windows = (allocation_training_window(first), allocation_training_window(second))
    schedule = AllocationTrainingSchedule(
        windows,
        train_window_ids=tuple(window.window_id for window in windows),
    )
    return AllocationTrainingScheduleEnv(schedule, (second, first)), (first, second)


def capability():
    from trade_rl.evaluation.rl_allocation.scheduled_training import (
        fit_allocation_ppo_schedule,
    )

    return fit_allocation_ppo_schedule


def test_actual_schedule_fit_cycles_declared_accounts_and_records_reuse():
    env, _ = scheduled()
    fit = capability()(
        env,
        total_timesteps=8,
        seed=7,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=2),
    )
    receipt = fit.receipt
    assert fit.model.num_timesteps == receipt["actual_timesteps"] == 8
    assert receipt["schema"] == "allocation_ppo_schedule_fit_receipt_v1"
    assert receipt["schedule_digest"] == env.schedule.digest
    assert receipt["protocol"]["ppo"]["n_steps"] == 4
    # DummyVecEnv immediately prepares the next episode after a true terminal.
    # The final prepared reset is retained as runtime evidence even though no
    # actor decision consumes it before this fit returns.
    assert receipt["usage"]["windows"] == [
        {
            "window_id": env.schedule.train_window_ids[0],
            "reset_count": 3,
            "decision_count": 4,
        },
        {
            "window_id": env.schedule.train_window_ids[1],
            "reset_count": 2,
            "decision_count": 4,
        },
    ]
    assert sum(row["decision_count"] for row in receipt["usage"]["windows"]) == 8
    assert receipt["consumption"]["windows"] == [
        {
            "window_id": env.schedule.train_window_ids[0],
            "decision_indices": [6, 7],
            "decision_counts": [2, 2],
            "observation_indices": [6, 7],
        },
        {
            "window_id": env.schedule.train_window_ids[1],
            "decision_indices": [8, 9],
            "decision_counts": [2, 2],
            "observation_indices": [8, 9],
        },
    ]
    assert "experience_count" not in canonical_json_bytes(receipt).decode()


def test_schedule_fit_requires_explicit_protocol_and_exact_rollout_budget(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as module

    env, _ = scheduled()
    monkeypatch.setattr(
        module,
        "construct_protocol_ppo",
        lambda *a, **k: pytest.fail("backend construction reached"),
    )
    with pytest.raises(ValueError, match="explicit"):
        module.fit_allocation_ppo_schedule(env, total_timesteps=8)
    with pytest.raises(ValueError, match="rollout"):
        module.fit_allocation_ppo_schedule(
            env,
            total_timesteps=6,
            training_protocol=protocol(n_steps=4, batch_size=2),
        )


def test_schedule_fit_revalidates_every_window_after_learning(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as module

    env, children = scheduled()
    original = module.construct_protocol_ppo

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        learn = model.learn

        def altered(**learn_kwargs):
            result = learn(**learn_kwargs)
            child = children[1]
            values = child.dataset.features.copy()
            values[8, 0, 0] += 9
            child.dataset = child.executor.dataset = replace(
                child.dataset, features=values
            )
            return result

        model.learn = altered
        return model

    monkeypatch.setattr(module, "construct_protocol_ppo", build)
    with pytest.raises(ValueError, match="source|schedule"):
        module.fit_allocation_ppo_schedule(
            env,
            total_timesteps=4,
            seed=7,
            training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
        )


def test_schedule_fit_receipt_is_detached_from_mutable_runtime_usage():
    env, _ = scheduled()
    fit = capability()(
        env,
        total_timesteps=4,
        seed=0,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
    )
    before = canonical_json_bytes(fit.receipt)
    env._usage[env.schedule.train_window_ids[0]]["decision_count"] += 100
    assert canonical_json_bytes(fit.receipt) == before


def test_schedule_fit_v4_reconstructs_frozen_prefix_once_and_records_it():
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args

    child = AllocationTradingEnv(**frozen_args(stop=8))
    window = allocation_training_window(child)
    schedule = AllocationTrainingSchedule(
        (window,), train_window_ids=(window.window_id,)
    )
    env = AllocationTrainingScheduleEnv(schedule, (child,))
    fit = capability()(
        env,
        total_timesteps=2,
        seed=0,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
    )
    receipt = fit.receipt
    assert receipt["preprocessing_fit"]["schema"] == (
        "allocation_ppo_preprocessing_fit_receipt_v1"
    )
    assert receipt["preprocessing_fit"]["declaration_digest"] == (
        child.feature_preprocessing.digest
    )
    assert receipt["consumption"]["windows"] == [
        {
            "window_id": window.window_id,
            "decision_indices": [6, 7],
            "decision_counts": [1, 1],
            "observation_indices": [6, 7],
        }
    ]


def dated_args(day, *, start=6, stop=10):
    from tests.evaluation.test_forecast_allocation import fitted
    from trade_rl.artifacts import content_digest

    args = parameters(stop=10)
    delta = np.timedelta64(day - 1, "D")
    args["dataset"] = replace(
        args["dataset"],
        dataset_id=content_digest({"synthetic_day": day}),
        timestamps=args["dataset"].timestamps + delta,
        available_at=args["dataset"].resolved_array("available_at") + delta,
    )
    args["stream"] = fitted(args["dataset"])
    args["start_index"], args["stop_index"] = start, stop
    args["estimates"] = tuple(
        replace(
            c,
            decision_time=c.decision_time + delta,
            available_at=c.available_at + delta,
            horizon_end=c.horizon_end + delta,
        )
        for c in args["estimates"][start - 6 : stop - 6]
    )
    clock = replace(args["bound"].clock, economic_horizon_seconds=(stop - start) * 3600)
    objective = replace(
        args["bound"].objective,
        evaluation_start=datetime(2026, 1, day, start, tzinfo=UTC),
        evaluation_stop_exclusive=datetime(2026, 1, day, stop, tzinfo=UTC),
    )
    args["bound"] = BoundObjectiveClock(objective, clock)
    return opt_in(args)


def test_actual_scheduled_fit_publishes_reloadable_same_horizon_continuous_policy(
    tmp_path,
):
    from tests.evaluation.test_allocation_continuation import book_economics
    from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
        run_artifact_bound_continuous_allocation_walk_forward,
    )
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    first, second = (AllocationTradingEnv(**dated_args(day)) for day in (1, 2))
    excluded = AllocationTradingEnv(**dated_args(3))
    windows = (
        allocation_training_window(first),
        allocation_training_window(second),
        allocation_training_window(excluded, role="held_out"),
    )
    schedule = AllocationTrainingSchedule(
        windows, tuple(w.window_id for w in windows[:2])
    )
    runtime = AllocationTrainingScheduleEnv(schedule, (first, second))
    fit = capability()(
        runtime,
        total_timesteps=8,
        seed=7,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
    )
    publish = getattr(fit, "inference_policy", None)
    assert callable(publish), "actual scheduled fit has no immutable inference handoff"
    policy = publish()
    assert policy.manifest["schema"] == "allocation_ppo_inference_bundle_v5"
    assert not hasattr(excluded, "book")
    root = tmp_path / "scheduled-policy"
    digest = save_allocation_policy(root, policy)
    loaded = load_allocation_policy(
        root, expected_digest=digest, expected_recipe_digest=first.recipe_digest
    )
    with pytest.raises(FileExistsError):
        save_allocation_policy(root, policy)

    oos_args = dated_args(4)
    memory = AllocationTradingEnv(**oos_args)
    saved = AllocationTradingEnv(**oos_args)
    assert memory.recipe_digest == first.recipe_digest
    observation, _ = memory.reset(seed=17)
    memory_actions = []
    for _ in range(4):
        action = policy.action(observation, runtime_recipe_digest=memory.recipe_digest)
        assert action == loaded.action(
            observation, runtime_recipe_digest=memory.recipe_digest
        )
        memory_actions.append(action)
        observation, _, _, _, _ = memory.step(action)
    artifact = AllocationFoldPolicyArtifact(0, root, digest, saved.recipe_digest)
    result = run_artifact_bound_continuous_allocation_walk_forward(
        folds_for(saved),
        (saved,),
        (artifact,),
        reset_seed=17,
    )
    assert result.policy_digests == (digest,)
    assert result.stitched.boundaries == ((6, 10),)
    assert book_economics(saved.book) == pytest.approx(book_economics(memory.book))
    assert saved.book.as_of_index == memory.book.as_of_index == 10
    assert len(memory_actions) == 4

    # Aggregate H4 does not make two locally bound H2 recipes compatible.
    wrong = tuple(
        AllocationTradingEnv(**dated_args(4, start=a, stop=b))
        for a, b in ((6, 8), (8, 10))
    )
    wrong_artifacts = tuple(
        AllocationFoldPolicyArtifact(i, root, digest, first.recipe_digest)
        for i in range(2)
    )
    with pytest.raises(ValueError, match="recipe"):
        run_artifact_bound_continuous_allocation_walk_forward(
            folds_for(*wrong),
            wrong,
            wrong_artifacts,
        )
    assert not any(hasattr(env, "book") for env in wrong)


@pytest.mark.parametrize(
    "field,value",
    [("seed", 17), ("num_timesteps", 12), ("_n_updates", 0), ("n_epochs", 2)],
)
def test_scheduled_actual_model_changes_are_rejected_before_save(
    tmp_path, monkeypatch, field, value
):
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    env, _ = scheduled()
    fit = capability()(
        env,
        total_timesteps=4,
        seed=7,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
    )
    policy = fit.inference_policy()
    monkeypatch.setattr(fit.model, field, value)
    monkeypatch.setattr(
        fit.model, "save", lambda _: pytest.fail("model serialization reached")
    )
    with pytest.raises(ValueError, match="actual"):
        save_allocation_policy(tmp_path / "invalid-policy", policy)
    assert not (tmp_path / "invalid-policy").exists()


def test_scheduled_preprocessing_bundle_and_policy_bytes_fail_closed(
    tmp_path, monkeypatch
):
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args
    from trade_rl.artifacts import content_digest
    from trade_rl.strategies.rl import allocation_artifact as artifact

    child = AllocationTradingEnv(**frozen_args(stop=8))
    window = allocation_training_window(child)
    schedule = AllocationTrainingSchedule((window,), (window.window_id,))
    runtime = AllocationTrainingScheduleEnv(schedule, (child,))
    fit = capability()(
        runtime,
        total_timesteps=2,
        seed=0,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
    )
    policy = fit.inference_policy()
    root = tmp_path / "frozen-policy"
    digest = artifact.save_allocation_policy(root, policy)
    loaded = artifact.load_allocation_policy(
        root, expected_digest=digest, expected_recipe_digest=child.recipe_digest
    )
    inference = AllocationTradingEnv(**frozen_args(stop=8))
    observation, _ = inference.reset()
    assert observation[0] == 3.0  # Raw 4, declared causal prefix mean/scale 1/1.
    assert policy.action(
        observation, runtime_recipe_digest=child.recipe_digest
    ) == loaded.action(observation, runtime_recipe_digest=child.recipe_digest)
    original = (root / "manifest.json").read_bytes()
    manifest = loaded.manifest
    manifest["training"]["preprocessing_fit"]["declaration_digest"] = "f" * 64
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    monkeypatch.setattr(
        artifact, "_load_policy", lambda _: pytest.fail("optional loader reached")
    )
    with pytest.raises(ValueError, match="preprocessing"):
        artifact.load_allocation_policy(
            root,
            expected_digest=content_digest(manifest),
            expected_recipe_digest=child.recipe_digest,
        )
    (root / "manifest.json").write_bytes(original)
    (root / "policy.zip").write_bytes(b"changed actual policy bytes")
    with pytest.raises(ValueError):
        artifact.load_allocation_policy(
            root, expected_digest=digest, expected_recipe_digest=child.recipe_digest
        )


def test_schedule_fit_rejects_non_native_seed_before_backend_construction(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as module

    class Seed(int):
        pass

    env, _ = scheduled()
    calls = []
    original = module.construct_protocol_ppo

    def build(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "construct_protocol_ppo", build)
    try:
        module.fit_allocation_ppo_schedule(
            env,
            total_timesteps=4,
            seed=Seed(7),
            training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
        )
    except ValueError:
        pass
    assert not calls, "non-native seed reached actual backend construction"


def test_actual_v3_scheduled_bundle_global_clock_matches_uninterrupted_policy(
    tmp_path, monkeypatch
):
    """Software identity only; four synthetic PPO decisions do not prove learning."""
    from tests.evaluation.test_allocation_continuation import book_economics
    from tests.evaluation.test_allocation_continuous_walk_forward import fold
    from tests.evaluation.test_allocation_preprocessing_runtime import (
        bind_frozen,
        frozen_args,
    )
    from trade_rl.evaluation.rl_allocation import global_walk_forward as consumer
    from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
    )
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    args = frozen_args(stop=10)
    child = AllocationTradingEnv(**args)
    window = allocation_training_window(child)
    schedule = AllocationTrainingSchedule((window,), (window.window_id,))
    fit = capability()(
        AllocationTrainingScheduleEnv(schedule, (child,)),
        total_timesteps=4,
        seed=7,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
    )
    policy = fit.inference_policy()
    root = tmp_path / "global-policy"
    digest = save_allocation_policy(root, policy)
    frozen = args["feature_preprocessing"]
    oos_args = bind_frozen(dated_args(4), frozen)
    direct, saved = AllocationTradingEnv(**oos_args), AllocationTradingEnv(**oos_args)
    assert direct.recipe_digest == child.recipe_digest
    assert direct.dataset.dataset_id != child.dataset.dataset_id
    obs, _ = direct.reset(seed=17)
    actions = []
    for _ in range(4):
        actions.append(policy.action(obs, runtime_recipe_digest=direct.recipe_digest))
        obs, _, done, truncated, _ = direct.step(actions[-1])
        assert not truncated and done is (direct.index == 10)
    folds = (fold(0, 6, 7), fold(1, 7, 10))
    artifacts = tuple(
        AllocationFoldPolicyArtifact(i, root, digest, child.recipe_digest)
        for i in range(2)
    )
    loaded_actions, resets = [], []
    original_load, original_reset = consumer.load_allocation_policy, saved.reset

    def load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        native_action = loaded.action

        def action(observation, **pins):
            value = native_action(observation, **pins)
            loaded_actions.append(value)
            return value

        monkeypatch.setattr(loaded, "action", action)
        return loaded

    def reset(**kwargs):
        resets.append(kwargs)
        return original_reset(**kwargs)

    monkeypatch.setattr(consumer, "load_allocation_policy", load)
    monkeypatch.setattr(saved, "reset", reset)
    from tests.evaluation.test_allocation_global_comparison_evidence import (
        SCOPE,
        contract_for_plan,
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
        saved,
        kind="direct_ppo",
        scenario="base",
        seed=7,
        reset_seed=17,
        artifacts=artifacts,
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_digest=provenance["runtime_environment_digest"],
    )
    comparison_contract = contract_for_plan(expected_plan, seeds=(7,))
    observed = execution.run_declared_global_allocation_execution(
        folds, saved, expected_plan, artifacts=artifacts
    )
    result = observed.native_result
    validity_record = GlobalAllocationValidityRecord.from_payload(
        validity_payload(comparison_contract, expected_plan, observed)
    )
    comparison_row = build_global_allocation_comparison_evidence(
        comparison_contract,
        observed,
        expected_plan=expected_plan,
        validity_record=validity_record,
        expected_assurance_scope=SCOPE,
    )
    assert observed.receipt.payload["reset_count"] == 1
    assert len(observed.receipt.payload["rows"]) == 4
    assert comparison_row.seed == 7
    assert comparison_row.terminal_profit_rate == pytest.approx(
        saved.book.portfolio_value / saved.initial_capital - 1
    )
    assert comparison_row.max_drawdown == saved.book.max_drawdown
    assert loaded_actions == actions
    assert resets == [{"seed": 17}]
    assert result.policy_digests == (digest, digest)
    assert result.stitched.boundaries == ((6, 7), (7, 10))
    assert book_economics(saved.book) == book_economics(direct.book)
    assert allocation_state_digest(saved) == allocation_state_digest(direct)

    # The complete roster is metadata-checked before even the first backend.
    from trade_rl.artifacts import content_digest

    bad_root = tmp_path / "bad-second"
    bad_root.mkdir()
    raw = policy.manifest
    raw["policy_sha256"] = "f" * 64
    (bad_root / "policy.zip").write_bytes(b"inert")
    (bad_root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    bad_artifact = AllocationFoldPolicyArtifact(
        1, bad_root, content_digest(raw), child.recipe_digest
    )
    monkeypatch.setattr(
        consumer,
        "load_allocation_policy",
        lambda *_args, **_kwargs: pytest.fail("backend before all metadata preflight"),
    )
    # Internally consistent repinned claims are still not authenticated fit
    # history. Move this complete H4 window so its terminal equals OOS start.
    from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
    from trade_rl.strategies.rl.allocation_training_schedule import (
        AllocationTrainingWindow,
    )

    training = raw["training"]
    source = training["sources"][0]["source"]
    source["decision_start"], source["terminal_time"] = (
        "2026-01-04T02:00:00.000000000",
        "2026-01-04T06:00:00.000000000",
    )
    objective = training["sources"][0]["objective"]
    objective["evaluation_start"], objective["evaluation_stop_exclusive"] = (
        "2026-01-04T02:00:00Z",
        "2026-01-04T06:00:00Z",
    )
    claimed = training["schedule"]["windows"][0]
    claimed["decision_start_ns"] += 68 * 3600 * 10**9
    claimed["terminal_time_ns"] += 68 * 3600 * 10**9
    envelope = source | {"decision_counts": [1] * 4}
    claimed["source_digest"] = content_digest(
        {k: v for k, v in envelope.items() if k != "dataset_id"}
    )
    identity = AllocationTrainingWindow.from_payload(claimed).window_id
    training["schedule"]["train_window_ids"][0] = identity
    training["schedule_digest"] = content_digest(training["schedule"])
    training["usage"]["schedule_digest"] = training["schedule_digest"]
    for field in ("usage", "consumption"):
        training[field]["windows"][0]["window_id"] = identity
    training["sources"][0]["window_id"] = identity
    assert validate_allocation_manifest(raw, require_policy=True) == raw
    (bad_root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    bad_artifact = replace(bad_artifact, expected_digest=content_digest(raw))
    fresh = AllocationTradingEnv(**oos_args)
    with pytest.raises(ValueError, match="strictly precede"):
        consumer.run_artifact_bound_global_allocation_walk_forward(
            folds, fresh, (artifacts[0], bad_artifact)
        )
    assert not hasattr(fresh, "book")

    evidence_root = tmp_path / "global-comparison"
    evidence_root.mkdir()
    (evidence_root / "software-evidence.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "allocation_global_runtime_comparison_software_evidence_v1",
                "software_kind": "genuine_unchanged_scheduled_v3_policy7_reset17",
                "learned_alpha_established": False,
                "expected_plan": expected_plan.payload,
                "receipt": observed.receipt.payload,
                "validity_record": validity_record.payload,
                "comparison_evidence": comparison_row.payload(),
            }
        )
    )
