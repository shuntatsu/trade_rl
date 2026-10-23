"""Independent native-child, actor and chronological admission witnesses."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.test_allocation_preprocessing_runtime import (
    bind_frozen,
    finite_forecast,
)
from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.evaluation.test_allocation_transition_trace import (
    CallbackBase,
    DetachedTensor,
)
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.preprocessing import (
    fit_allocation_feature_preprocessing,
)
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
    allocation_training_window,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)


def dated_args(day):
    args = parameters(stop=10)
    shift = np.timedelta64(day - 1, "D")
    data = args["dataset"]
    args["dataset"] = replace(
        data,
        dataset_id=content_digest({"scheduled_trace_day": day}),
        timestamps=data.timestamps + shift,
        available_at=data.resolved_array("available_at") + shift,
    )
    args["stream"] = finite_forecast(args["dataset"])
    args["estimates"] = tuple(
        replace(
            c,
            decision_time=c.decision_time + shift,
            available_at=c.available_at + shift,
            horizon_end=c.horizon_end + shift,
        )
        for c in args["estimates"]
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            args["bound"].objective,
            evaluation_start=datetime(2026, 1, day, 6, tzinfo=UTC),
            evaluation_stop_exclusive=datetime(2026, 1, day, 10, tzinfo=UTC),
        ),
    )
    return opt_in(args)


def scheduled_trace_env():
    arguments = []
    for day in (1, 2, 3):
        args = dated_args(day)
        data = args["dataset"]
        features = data.features.copy()
        features[:3, 0, 0] = [-1, 0, 1]
        features[6:, 0, 0] = day * 10 + np.arange(data.n_bars - 6)
        args["dataset"] = replace(data, features=features)
        args["stream"] = finite_forecast(args["dataset"])
        args["bound"] = replace(
            args["bound"], clock=replace(args["bound"].clock, rollout_steps=6)
        )
        arguments.append(args)
    data = arguments[0]["dataset"]
    frozen = fit_allocation_feature_preprocessing(
        data,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_start=0,
        fit_stop=3,
        fit_as_of=data.timestamps[3],
        first_decision_index=6,
    )
    children = [AllocationTradingEnv(**bind_frozen(args, frozen)) for args in arguments]
    windows = tuple(allocation_training_window(child) for child in children)
    return AllocationTrainingScheduleEnv(
        AllocationTrainingSchedule(windows, tuple(w.window_id for w in windows)),
        tuple(children),
    )


def capability():
    try:
        return import_module(
            "trade_rl.evaluation.rl_allocation.scheduled_transition_trace"
        ).ScheduledAllocationTransitionRecorder
    except ModuleNotFoundError:
        # The existing single-source recorder is the connected negative control.
        # It cannot attach to all schedule children and preserve this chain.
        from trade_rl.evaluation.rl_allocation.transition_trace import (
            AllocationTransitionRecorder,
        )

        return AllocationTransitionRecorder


def gae_witness(rewards, values, starts, last_value, done, *, gamma=1.0, lam=0.95):
    """Literal independently copied pinned SB3 operations including dtype promotion."""
    advantages = np.zeros_like(values, dtype=np.float32)
    running = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            nonterminal = 1.0 - np.array([done], dtype=bool)
            successor = np.array([last_value], dtype=np.float32)
        else:
            nonterminal = 1.0 - starts[step + 1]
            successor = values[step + 1]
        delta = rewards[step] + gamma * successor * nonterminal - values[step]
        running = delta + gamma * lam * nonterminal * running
        advantages[step] = running
    return advantages, advantages + values


def fake_capture(*, actions=(3, 0, 0, 0) * 3, zero_values=False):
    env = scheduled_trace_env()
    observations, rewards, starts, original_rows = [], [], [], []
    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        _last_episode_starts=np.array([True]),
        policy=SimpleNamespace(
            predict_values=lambda obs: DetachedTensor(
                np.array([[0 if zero_values else 7]], "<f4")
            )
        ),
    )
    try:
        recorder = capability()(env)
        recorder.attach(model)
    except (AttributeError, TypeError):
        # Behavioral RED: native schedule still runs but legacy observer cannot
        # capture child transitions. Check completed native chain, not API name.
        recorder = SimpleNamespace(events=(), detach=lambda: None)
    obs, info = env.reset(seed=7)
    model._last_obs = obs[None]
    model.env = SimpleNamespace(reset_infos=[info])
    callback = (
        recorder.callback(CallbackBase) if hasattr(recorder, "callback") else None
    )
    if callback is not None:
        callback.locals = {"self": model}
        callback._on_training_start()
    for sequence, action in enumerate(actions):
        before = obs.copy()
        old_id = env.active_window_id
        obs, reward, done, truncated, info = env.step(action)
        assert not truncated
        if done:
            info = dict(
                info,
                **{"terminal_observation": obs.copy(), "TimeLimit.truncated": False},
            )
            obs, reset_info = env.reset()
            model.env.reset_infos[0] = reset_info
        value = np.array([[0 if zero_values else sequence / 16]], "<f4")
        logpi = np.array([-0.5], "<f4")
        reward_array = np.array([reward], "<f4")
        original_rows.append(
            (
                old_id,
                before.copy(),
                action,
                value.copy(),
                logpi.copy(),
                reward_array.copy(),
            )
        )
        locals_ = {
            "self": model,
            "infos": [info],
            "dones": np.array([done]),
            "obs_tensor": DetachedTensor(before[None]),
            "actions": np.array([action], "<i8"),
            "values": DetachedTensor(value),
            "log_probs": DetachedTensor(logpi),
            "rewards": reward_array,
            "new_obs": obs[None],
        }
        observations.append(before[None])
        rewards.append(reward_array)
        starts.append(model._last_episode_starts.astype("<f4"))
        if callback is not None:
            callback.locals = locals_
            assert callback._on_step()
        model.num_timesteps += 1
        model._last_episode_starts = np.array([done])
        model._last_obs = obs[None]
        if (sequence + 1) % 6 == 0:
            batch = original_rows[-6:]
            buffer_values = np.array([row[3].reshape(1) for row in batch], "<f4")
            adv, ret = gae_witness(
                np.array(rewards[-6:], "<f4"),
                buffer_values,
                np.array(starts[-6:], "<f4"),
                0 if zero_values else 7,
                done,
            )
            model.rollout_buffer = SimpleNamespace(
                full=True,
                pos=6,
                generator_ready=False,
                observations=np.array(observations[-6:], "<f4"),
                actions=np.array([[[row[2]]] for row in batch], "<f4"),
                values=buffer_values,
                log_probs=np.array([row[4] for row in batch], "<f4"),
                rewards=np.array(rewards[-6:], "<f4"),
                episode_starts=np.array(starts[-6:], "<f4"),
                advantages=adv,
                returns=ret,
            )
            model.policy.predict_values(DetachedTensor(obs[None]))
            if callback is not None:
                callback._on_rollout_end()
    recorder.detach()
    return (
        env,
        recorder,
        [json.loads(event) for event in recorder.events],
        original_rows,
    )


def test_scheduled_collector_preserves_old_child_and_prepared_next_account():
    env, recorder, events, witness = fake_capture()
    rows = [event for event in events if event["kind"] == "transition"]
    assert len(rows) == 12, "scheduled native transitions were not captured"
    assert [row["old"]["window_id"] for row in rows] == [row[0] for row in witness]
    for row, actual in zip(rows, witness, strict=True):
        assert bytes.fromhex(row["actor"]["observation"]) == actual[1][None].tobytes()
        assert bytes.fromhex(row["actor"]["value"]) == actual[3].tobytes()
        assert bytes.fromhex(row["actor"]["log_prob"]) == actual[4].tobytes()
    assert rows[3]["old"]["window_ordinal"] == 0
    assert rows[3]["next"]["window_ordinal"] == 1
    assert rows[3]["old"]["index"] == 9 and rows[3]["next"]["index"] == 6
    assert rows[-1]["old"]["window_ordinal"] == 2
    assert rows[-1]["next"]["window_ordinal"] == 0
    assert rows[-1]["next"]["episode"] == 3
    assert rows[-1]["terminal"] == np.zeros(recorder.width, "<f4").tobytes().hex()
    boundaries = [event for event in events if event["kind"] == "rollout"]
    assert boundaries[0]["current"]["index"] == 8
    assert boundaries[0]["done"] is False
    assert boundaries[-1]["current"]["index"] == 6
    assert boundaries[-1]["done"] is True
    assert all(
        child._transition_recorder is None for child in env.training_environments
    )


def test_literal_gae_witness_live_bootstrap_and_terminal_mask():
    values = np.zeros((2, 1), "<f4")
    starts = np.array([[1], [0]], "<f4")
    rewards = np.array([[0], [10]], "<f4")
    _, terminal = gae_witness(rewards, values, starts, 7, True)
    _, live = gae_witness(rewards, values, starts, 7, False)
    assert terminal[:, 0].tolist() == [9.5, 10]
    assert live[:, 0].tolist() == pytest.approx([16.15, 17])


def test_occupied_later_child_slot_prevents_partial_attachment():
    env = scheduled_trace_env()
    occupied = object()
    env.training_environments[1]._transition_recorder = occupied
    cls = capability()
    try:
        recorder = cls(env)
        with pytest.raises(ValueError, match="occupied|unoccupied"):
            recorder.validate_fit(env, protocol(n_steps=6, batch_size=2))
    except AttributeError:
        pytest.fail("scheduled occupied-slot preflight is absent")
    assert env.training_environments[0]._transition_recorder is None
    assert env.training_environments[1]._transition_recorder is occupied


def test_diagnostics_mode_cannot_be_downgraded_and_replaced_predictor_is_restored():
    env = scheduled_trace_env()
    trace = capability()(env, learner_diagnostics="native_ppo_update_v1")
    with pytest.raises(AttributeError):
        trace.learner_diagnostics = "none"

    def original(obs):
        return obs

    model = SimpleNamespace(policy=SimpleNamespace(predict_values=original))
    trace.attach(model)
    model.policy.predict_values = lambda obs: None
    with pytest.raises(ValueError, match="replaced"):
        trace.detach()
    assert model.policy.predict_values is original
    assert all(
        child._transition_recorder is None for child in env.training_environments
    )
    assert not trace.complete
    trace._learner_diagnostics = "none"
    with pytest.raises(ValueError, match="mode changed"):
        _ = trace.learner_diagnostics


def fake_bundle(env):
    from trade_rl.evaluation.rl_allocation.scheduled_training import _preprocessing_fit
    from trade_rl.evaluation.rl_allocation.training_source import (
        allocation_training_source,
    )

    declaration = protocol(n_steps=6, batch_size=2, n_epochs=1)
    usage = env.usage_payload()
    total = sum(row["decision_count"] for row in usage["windows"])
    cycles = total // 12
    consumption = [
        {
            "window_id": w.window_id,
            "decision_indices": list(range(6, 10)),
            "decision_counts": [cycles] * 4,
            "observation_indices": list(range(6, 10)),
        }
        for w in env.schedule.training_windows
    ]
    template = env.template_env
    return {
        "schema": "allocation_ppo_inference_bundle_v5",
        "recipe": template.recipe,
        "recipe_digest": template.recipe_digest,
        "training": {
            "schema": "allocation_ppo_schedule_training_receipt_v1",
            "schedule": env.schedule.payload(),
            "schedule_digest": env.schedule.digest,
            "recipe_digest": template.recipe_digest,
            "seed": 7,
            "requested_timesteps": total,
            "actual_timesteps": total,
            "clock": template.bound.clock.payload(),
            "protocol": declaration.payload(),
            "protocol_digest": declaration.digest,
            "optimization": {
                "schema": "allocation_ppo_optimization_receipt_v1",
                "rollout_update_count": total // 6,
                "successful_optimizer_step_calls": total // 2,
                "epoch_iterations": total // 6,
                "optimizer_step_events_digest": "b" * 64,
                "final_capture": "after_final_train_v1",
            },
            "usage": usage,
            "consumption": {
                "schema": "allocation_training_schedule_consumption_v1",
                "windows": consumption,
            },
            "preprocessing_fit": _preprocessing_fit(env, declaration),
            "sources": [
                {
                    "window_id": w.window_id,
                    "objective": child.bound.objective.payload(),
                    "source": allocation_training_source(
                        child,
                        decision_counts=dict.fromkeys(range(6, 10), cycles),
                        observation_indices=tuple(range(6, 10)),
                    ),
                }
                for w, child in zip(
                    env.schedule.training_windows,
                    env.training_environments,
                    strict=True,
                )
            ],
            "ppo": {
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "n_steps": 6,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 0.002,
                "net_arch": {"pi": [32, 32], "vf": [32, 32]},
                "device": "cpu",
            },
        },
    }


def test_closed_scheduled_reader_joins_source_reuse_and_final_reset():
    env, _, events, _ = fake_capture()
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    summary = validate_scheduled_transition_events(events, fake_bundle(env))
    assert summary["transition_count"] == 12
    assert [row["actor_count"] for row in summary["input_summary"]["windows"]] == [
        4,
        4,
        4,
    ]
    assert [row["reset_count"] for row in summary["input_summary"]["windows"]] == [
        2,
        1,
        1,
    ]
    assert [
        row["prepared_reset_without_actor_count"]
        for row in summary["input_summary"]["windows"]
    ] == [1, 0, 0]


@pytest.mark.parametrize(
    "mutation",
    [
        "old_source",
        "next_episode",
        "same_index_other_bits",
        "action",
        "logpi",
        "value",
        "reward",
        "terminal",
        "buffer",
        "bootstrap",
        "gae",
        "extra",
        "episode_start",
        "done",
    ],
)
def test_relational_tamper_fails_even_with_all_content_pins_recomputed(mutation):
    env, _, events, _ = fake_capture()
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    changed = deepcopy(events)
    first = changed[1]
    if mutation == "old_source":
        first["old"]["dataset_id"] = "f" * 64
    elif mutation == "next_episode":
        changed[4]["next"]["episode"] += 1
    elif mutation == "same_index_other_bits":
        first["actor"]["observation"] = changed[5]["actor"]["observation"]
    elif mutation == "action":
        first["actor"]["action_code"] = 1
    elif mutation == "logpi":
        first["actor"]["log_prob"] = np.array([1], "<f4").tobytes().hex()
    elif mutation == "value":
        first["actor"]["value"] = np.array([[1]], "<f4").tobytes().hex()
    elif mutation == "reward":
        first["actor"]["reward"] = np.array([1], "<f4").tobytes().hex()
    elif mutation == "terminal":
        changed[4]["terminal"] = np.ones(47, "<f4").tobytes().hex()
    elif mutation == "buffer":
        changed[7]["buffer_digest"] = "f" * 64
    elif mutation == "bootstrap":
        changed[7]["bootstrap"]["observation"] = changed[1]["actor"]["observation"]
    elif mutation == "gae":
        changed[7]["advantages"] = np.zeros((6, 1), "<f4").tobytes().hex()
    elif mutation == "extra":
        first["extra"] = 0
    elif mutation == "episode_start":
        first["episode_start"] = False
    elif mutation == "done":
        first["done"] = True
    with pytest.raises(ValueError):
        validate_scheduled_transition_events(changed, fake_bundle(env))


def coherent_feature_reuse_tamper(events, env):
    """Reconstruct native proposals and all affected tensor links; no stale pins."""
    from trade_rl.evaluation.rl_allocation.transition_arrays import buffer_payload
    from trade_rl.evaluation.rl_allocation.transition_facts import _construct, _json
    from trade_rl.strategies.allocation import (
        AfterCostTargetAllocator,
        AllocationContext,
        AllocationInputs,
    )
    from trade_rl.strategies.allocation_action import (
        AllocationActionContract,
        AllocationDecision,
    )

    changed = deepcopy(events)
    target = next(
        row for row in changed if row["kind"] == "transition" and row["sequence"] == 12
    )
    raw = target["facts"]["proposal"]["decision"]
    baseline = raw["baseline"]
    inputs = _construct(
        AllocationInputs,
        baseline["inputs"],
        clocks=("decision_time", "available_at", "horizon_end"),
    )
    context = _construct(
        AllocationContext, baseline["context"], clocks=("decision_time",)
    )
    allocator = _construct(AfterCostTargetAllocator, baseline["allocator"])
    feature = raw["feature_values"][0] + 1.0
    decision = AllocationDecision(
        baseline=allocator.propose(inputs, context),
        action_contract=_construct(AllocationActionContract, raw["action_contract"]),
        **{
            key: (feature,)
            if key == "feature_values"
            else tuple(raw[key])
            if key == "feature_names"
            else raw[key]
            for key in (
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
    proposal = decision.propose(target["actor"]["action_code"])
    target["facts"]["proposal"] = _json(proposal)
    target["facts"]["proposal_digest"] = proposal.digest
    target["facts"]["decision_digest"] = decision.decision_digest
    obs = np.frombuffer(bytes.fromhex(target["actor"]["observation"]), "<f4").copy()
    frozen = env.template_env.feature_preprocessing
    obs[:1] = frozen.transform(
        (feature,),
        feature_names=frozen.normalizer.feature_names,
        feature_config_digest=frozen.feature_config_digest,
        source_normalization_digest=frozen.source_normalization_digest,
        decision_time_ns=target["old"]["time_ns"],
    )
    target["actor"]["observation"] = obs.tobytes().hex()
    prior = next(
        row for row in changed if row["kind"] == "transition" and row["sequence"] == 11
    )
    prior["next_observation"] = target["actor"]["observation"]
    for boundary in (row for row in changed if row["kind"] == "rollout"):
        batch = [
            row
            for row in changed
            if row["kind"] == "transition" and row["rollout"] == boundary["rollout"]
        ]
        boundary["buffer_digest"] = content_digest(buffer_payload(batch, len(obs)))
        if boundary["last"] == 11:
            boundary["bootstrap"]["observation"] = target["actor"]["observation"]
    return changed


def test_repeated_source_features_are_immutable_while_account_suffixes_vary():
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    env, _, events, _ = fake_capture(actions=(3, 0, 0, 0) * 6)
    bundle = fake_bundle(env)
    assert (
        validate_scheduled_transition_events(events, bundle)["transition_count"] == 24
    )
    changed = coherent_feature_reuse_tamper(events, env)
    with pytest.raises(ValueError, match="feature.*reuse|feature.*reset"):
        validate_scheduled_transition_events(changed, bundle)


def native_update_fixture(*, distribution=(3, 3), target_kl=None, zero_targets=False):
    """Non-model closed fixture with explicit deterministic optimizer framing."""
    from hashlib import sha256

    from trade_rl.artifacts import canonical_json_bytes

    env, _, events, _ = (
        fake_capture(actions=(0,) * 12, zero_values=True)
        if zero_targets
        else fake_capture()
    )
    bundle = fake_bundle(env)
    declaration = protocol(n_steps=6, batch_size=2, n_epochs=1, target_kl=target_kl)
    training = bundle["training"]
    training["protocol"] = declaration.payload()
    training["protocol_digest"] = declaration.digest
    training["optimization"]["successful_optimizer_step_calls"] = sum(distribution)
    digest = sha256()
    step = 0
    result = []
    for row in events:
        result.append(row)
        if row["kind"] != "rollout":
            continue
        count = distribution[row["rollout"]]
        for _ in range(count):
            step += 1
            raw = canonical_json_bytes(
                {
                    "step": step,
                    "rollout": row["rollout"] + 1,
                    "timesteps": (row["rollout"] + 1) * 6,
                }
            )
            digest.update(len(raw).to_bytes(8, "big") + raw)
        batch = [
            r
            for r in events
            if r["kind"] == "transition" and r["rollout"] == row["rollout"]
        ]
        targets = np.frombuffer(bytes.fromhex(row["returns"]), "<f4")
        values = np.array(
            [
                np.frombuffer(bytes.fromhex(r["actor"]["value"]), "<f4")[0]
                for r in batch
            ],
            "<f4",
        )
        variance = np.var(targets)
        ev = None if variance == 0 else float(1 - np.var(targets - values) / variance)
        result.append(
            {
                "kind": "update",
                "rollout": row["rollout"],
                "first": row["first"],
                "last": row["last"],
                "buffer_digest": row["buffer_digest"],
                "optimizer_steps": count,
                "epochs": 1,
                "n_updates": row["rollout"] + 1,
                "diagnostics": {
                    "value_loss": 0.0,
                    "policy_gradient_loss": 0.0,
                    "entropy_loss": 0.0,
                    "approx_kl": 0.0,
                    "clip_fraction": 0.0,
                    "explained_variance": ev,
                },
                "explained_variance_reason": "zero_target_variance"
                if ev is None
                else None,
                "parameters_before": {"actor": "a" * 64, "critic": "b" * 64},
                "parameters_after": {"actor": "a" * 64, "critic": "b" * 64},
            }
        )
    training["optimization"]["optimizer_step_events_digest"] = digest.hexdigest()
    return result, bundle


@pytest.mark.parametrize("mutation", ["wrong_digest", "redistributed_kl_calls"])
def test_native_update_event_digest_joins_chronological_counts(mutation):
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    events, bundle = native_update_fixture(distribution=(1, 2), target_kl=0.1)
    assert (
        validate_scheduled_transition_events(
            events, bundle, learner_diagnostics="native_ppo_update_v1"
        )["rollout_count"]
        == 2
    )
    if mutation == "wrong_digest":
        bundle["training"]["optimization"]["optimizer_step_events_digest"] = "f" * 64
    else:
        updates = [r for r in events if r["kind"] == "update"]
        updates[0]["optimizer_steps"], updates[1]["optimizer_steps"] = 2, 1
    with pytest.raises(ValueError, match="optimizer.*digest"):
        validate_scheduled_transition_events(
            events, bundle, learner_diagnostics="native_ppo_update_v1"
        )


def test_zero_target_variance_is_null_with_declared_reason_and_no_invented_zero():
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    events, bundle = native_update_fixture(zero_targets=True)
    updates = [row for row in events if row["kind"] == "update"]
    assert all(
        row["diagnostics"]["explained_variance"] is None
        and row["explained_variance_reason"] == "zero_target_variance"
        for row in updates
    )
    assert (
        validate_scheduled_transition_events(
            events, bundle, learner_diagnostics="native_ppo_update_v1"
        )["rollout_count"]
        == 2
    )
    updates[0]["diagnostics"]["explained_variance"] = 0.0
    updates[0]["explained_variance_reason"] = None
    with pytest.raises(ValueError, match="explained variance"):
        validate_scheduled_transition_events(
            events, bundle, learner_diagnostics="native_ppo_update_v1"
        )


def test_callbacks_setup_failure_leaves_no_registered_fit_hook(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as producer

    env = scheduled_trace_env()
    trace = capability()(env)
    hooks = {"foreign": object()}

    def register(callback):
        hooks["fit"] = callback
        return SimpleNamespace(remove=lambda: hooks.pop("fit", None))

    def predict(obs):
        return obs

    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        policy=SimpleNamespace(
            predict_values=predict,
            optimizer=SimpleNamespace(register_step_post_hook=register),
        ),
    )
    monkeypatch.setattr(
        producer, "construct_protocol_ppo", lambda *args, **kwargs: model
    )

    def broken_callbacks(name):
        assert name == "stable_baselines3.common.callbacks"
        raise RuntimeError("callbacks setup failed")

    monkeypatch.setattr(producer.importlib, "import_module", broken_callbacks)
    with pytest.raises(RuntimeError, match="callbacks setup failed"):
        producer.fit_allocation_ppo_schedule(
            env,
            total_timesteps=12,
            seed=7,
            training_protocol=protocol(n_steps=6, batch_size=2, n_epochs=1),
            transition_trace=trace,
        )
    assert set(hooks) == {"foreign"}
    assert not trace.complete and model.policy.predict_values is predict
    assert all(
        child._transition_recorder is None for child in env.training_environments
    )


def test_native_logger_update_ordinal_rejects_boolean_alias_before_event_capture():
    from trade_rl.artifacts import canonical_json_bytes

    env, _, events, _ = fake_capture()
    trace = capability()(env, learner_diagnostics="native_ppo_update_v1")
    boundary = next(row for row in events if row["kind"] == "rollout")
    trace._events = [canonical_json_bytes(boundary)]
    trace._rollouts = 1
    trace._pre_parameters = {"actor": "a" * 64, "critic": "b" * 64}
    tensor = DetachedTensor(np.array([1], "<f4"))
    trace._model = SimpleNamespace(
        _n_updates=1,
        policy=SimpleNamespace(
            named_parameters=lambda: [
                ("action_net.weight", tensor),
                ("value_net.weight", tensor),
            ]
        ),
        logger=SimpleNamespace(
            name_to_value={
                "train/n_updates": True,
                **{
                    "train/" + name: 0.0
                    for name in (
                        "value_loss",
                        "policy_gradient_loss",
                        "entropy_loss",
                        "approx_kl",
                        "clip_fraction",
                        "explained_variance",
                    )
                },
            }
        ),
    )
    with pytest.raises(ValueError, match="logger.*native|logger.*stale"):
        trace.observe_update(1, 3, 1)
    assert len(trace.events) == 1 and not trace.complete


def test_reused_source_prefix_accepts_actual_changed_account_suffix():
    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        validate_scheduled_transition_events,
    )

    env, _, events, _ = fake_capture(actions=(3, 0, 0, 0) * 3 + (1, 0, 0, 0) * 3)
    rows = {row["sequence"]: row for row in events if row["kind"] == "transition"}
    first = np.frombuffer(bytes.fromhex(rows[1]["actor"]["observation"]), "<f4")
    second = np.frombuffer(bytes.fromhex(rows[13]["actor"]["observation"]), "<f4")
    assert rows[1]["old"]["index"] == rows[13]["old"]["index"] == 7
    assert rows[1]["old"]["dataset_id"] == rows[13]["old"]["dataset_id"]
    assert first[:1].tobytes() == second[:1].tobytes()
    assert first[1:].tobytes() != second[1:].tobytes()
    assert (
        validate_scheduled_transition_events(events, fake_bundle(env))[
            "transition_count"
        ]
        == 24
    )
