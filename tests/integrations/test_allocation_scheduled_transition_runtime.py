"""Actual collector witnesses and pinned native GAE precision discriminators."""

import json
import random
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")


def independent_parameter_pins(parameters):
    """Name/shape/raw-byte oracle, without the production parameter projection."""
    from trade_rl.artifacts import canonical_json_bytes

    groups = {"actor": {}, "critic": {}}
    for name, value in parameters.items():
        assert value.dtype == np.dtype("<f4") and np.isfinite(value).all()
        group = (
            "critic"
            if name.startswith(("mlp_extractor.value_net.", "value_net."))
            else "actor"
        )
        groups[group][name] = {
            "shape": list(value.shape),
            "dtype": "<f4",
            "bytes": value.tobytes().hex(),
        }
    assert all(groups.values())
    return {
        name: sha256(canonical_json_bytes(values)).hexdigest()
        for name, values in groups.items()
    }


def assert_independent_parameter_update(row, before, after):
    assert row["parameters_before"] == independent_parameter_pins(before)
    assert row["parameters_after"] == independent_parameter_pins(after)


def independent_buffer_pin(buffer):
    from trade_rl.artifacts import canonical_json_bytes

    arrays = {
        name: buffer[name].tobytes().hex()
        for name in (
            "observations",
            "actions",
            "values",
            "log_probs",
            "rewards",
            "episode_starts",
        )
    }
    return sha256(canonical_json_bytes(arrays)).hexdigest()


@pytest.mark.parametrize(
    "case", ["large_cancellation", "unrounded_carry", "terminal_mask"]
)
def test_native_buffer_mixed_precision_target_bits(case):
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.buffers import RolloutBuffer

    from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
        scheduled_gae,
    )

    if case == "large_cancellation":
        rewards, values, last, done = [0, 1], [0, 2**24], 2**24, False
        expected = "0000803f"
    elif case == "unrounded_carry":
        rewards, values, last, done = [0, 1], [0, 0], 2**-24, False
        expected = "3433733f"
    else:
        rewards, values, last, done = [0, 10], [0, 0], 7, True
        expected = "0000184100002041"
    native = RolloutBuffer(
        2,
        spaces.Box(-1, 1, (1,), np.float32),
        spaces.Discrete(4),
        device="cpu",
        gamma=1.0,
        gae_lambda=0.95,
    )
    native.rewards[:] = np.array(rewards, "<f4")[:, None]
    native.values[:] = np.array(values, "<f4")[:, None]
    native.episode_starts[:] = np.array([[1], [0]], "<f4")
    native.compute_returns_and_advantage(
        torch.tensor([[last]], dtype=torch.float32), np.array([done], bool)
    )
    rows = [
        {
            "done": done if i == 1 else False,
            "episode_start": i == 0,
            "actor": {
                "reward": np.array([reward], "<f4").tobytes().hex(),
                "value": np.array([[value]], "<f4").tobytes().hex(),
            },
        }
        for i, (reward, value) in enumerate(zip(rewards, values, strict=True))
    ]
    advantages, returns = scheduled_gae(
        rows, np.array([[last]], "<f4").tobytes().hex(), gamma=1.0, gae_lambda=0.95
    )
    assert advantages == native.advantages.tobytes().hex()
    assert returns == native.returns.tobytes().hex()
    if case == "large_cancellation":
        assert native.advantages[-1].tobytes().hex() == expected
    elif case == "unrounded_carry":
        assert native.advantages[0].tobytes().hex() == expected
    else:
        assert advantages == expected


def fit_capture(
    monkeypatch, *, mode="native_ppo_update_v1", failure=None, kl_stop=False
):
    from tests.evaluation.test_allocation_scheduled_transition_trace import (
        scheduled_trace_env,
    )
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.evaluation.rl_allocation import scheduled_training as producer
    from trade_rl.evaluation.rl_allocation.scheduled_transition_trace import (
        ScheduledAllocationTransitionRecorder,
    )

    env = scheduled_trace_env()
    trace = ScheduledAllocationTransitionRecorder(env, learner_diagnostics=mode)
    actual, buffers, updates, models, bootstraps, native_steps = [], [], [], [], [], []
    originals = []
    foreign_hook = []
    step = env.step

    def native_step(action):
        result = step(action)
        native_steps.append((np.float32(result[1]), bool(result[2])))
        return result

    env.step = native_step
    original = producer.construct_protocol_ppo

    def construct(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)
        forward, compute, train, predict = (
            model.policy.forward,
            model.rollout_buffer.compute_returns_and_advantage,
            model.train,
            model.policy.predict_values,
        )
        originals.append((model, forward, compute, train, predict))
        if failure is not None:
            foreign_hook.append(
                model.policy.optimizer.register_step_post_hook(lambda *args: None)
            )
        if kl_stop:
            evaluate = model.policy.evaluate_actions

            def stopped_evaluate(*args, **kwargs):
                value, logpi, entropy = evaluate(*args, **kwargs)
                return value, logpi + 1, entropy

            model.policy.evaluate_actions = stopped_evaluate

        def actor(observation, *a, **kw):
            action, value, logpi = forward(observation, *a, **kw)
            actual.append(
                (
                    observation.detach().cpu().numpy().copy(),
                    action.detach().cpu().numpy().copy(),
                    value.detach().cpu().numpy().copy(),
                    logpi.detach().cpu().numpy().copy(),
                    model._last_episode_starts.astype("<f4").copy(),
                )
            )
            return action, value, logpi

        def bootstrap(observation):
            value = predict(observation)
            bootstraps.append(
                (
                    observation.detach().cpu().numpy().copy(),
                    value.detach().cpu().numpy().copy(),
                )
            )
            return value

        def targets(*a, **kw):
            result = compute(*a, **kw)
            buffer = model.rollout_buffer
            buffers.append(
                {
                    name: getattr(buffer, name).copy()
                    for name in (
                        "observations",
                        "actions",
                        "values",
                        "log_probs",
                        "rewards",
                        "episode_starts",
                        "advantages",
                        "returns",
                    )
                }
            )
            return result

        def learn_update():
            if failure == "train":
                raise RuntimeError("native train injected failure")
            before = {
                name: value.detach().cpu().numpy().copy()
                for name, value in model.policy.state_dict().items()
            }
            result = train()
            if failure == "logger_missing":
                model.logger.name_to_value.pop("train/value_loss")
            elif failure == "logger_stale":
                model.logger.name_to_value["train/n_updates"] = -1
            elif failure == "logger_ordinal_bool" and model._n_updates == 1:
                model.logger.name_to_value["train/n_updates"] = True
            elif failure == "logger_bool":
                model.logger.name_to_value["train/value_loss"] = True
            elif failure == "logger_nonfinite":
                model.logger.name_to_value["train/value_loss"] = float("inf")
            elif failure == "mode":
                trace._learner_diagnostics = "none"
            after = {
                name: value.detach().cpu().numpy().copy()
                for name, value in model.policy.state_dict().items()
            }
            updates.append((before, after, deepcopy(model.logger.name_to_value)))
            return result

        model.policy.forward = actor
        model.policy.predict_values = bootstrap
        model.rollout_buffer.compute_returns_and_advantage = targets
        model.train = learn_update
        if failure == "optimizer":

            def broken_step(*args, **kwargs):
                raise RuntimeError("native optimizer injected failure")

            monkeypatch.setattr(model.policy.optimizer, "step", broken_step)
        if failure == "last_false":
            import stable_baselines3.common.callbacks as cb

            learn = model.learn

            class Stop(cb.BaseCallback):
                def _on_step(self):
                    return self.model.num_timesteps < 12

            def stopped(**kwargs):
                kwargs["callback"] = cb.CallbackList([kwargs["callback"], Stop()])
                return learn(**kwargs)

            model.learn = stopped
        if failure in ("source", "model"):
            learn = model.learn

            def altered(**kwargs):
                result = learn(**kwargs)
                if failure == "model":
                    model.num_timesteps += 1
                else:
                    child = env.training_environments[1]
                    features = child.dataset.features.copy()
                    features[8, 0, 0] += 1
                    child.dataset = child.executor.dataset = replace(
                        child.dataset, features=features
                    )
                return result

            model.learn = altered
        return model

    monkeypatch.setattr(producer, "construct_protocol_ppo", construct)
    if failure == "setup":
        module = producer.importlib.import_module

        def broken_import(name):
            if name == "stable_baselines3.common.callbacks":
                raise RuntimeError("callbacks setup injected failure")
            return module(name)

        monkeypatch.setattr(producer.importlib, "import_module", broken_import)
    if failure in ("observer", "finish"):

        def broken(*args, **kwargs):
            raise RuntimeError("observer/finish injected failure")

        monkeypatch.setattr(
            trace, "observe_update" if failure == "observer" else "finish", broken
        )
    if failure == "partial_attach":
        from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv

        setter = AllocationTradingEnv.__setattr__
        second = env.training_environments[1]

        def broken_slot(child, name, value):
            if child is second and name == "_transition_recorder" and value is not None:
                raise RuntimeError("second child attach injected failure")
            return setter(child, name, value)

        monkeypatch.setattr(AllocationTradingEnv, "__setattr__", broken_slot)

    def restore_witnesses():
        # SB3 saves instance data. Test closures must not enter ordinary policy.zip.
        env.__dict__.pop("step", None)
        for model, *_ in originals:
            for name in ("train", "learn"):
                model.__dict__.pop(name, None)
            for name in ("forward", "predict_values"):
                model.policy.__dict__.pop(name, None)
            model.policy.__dict__.pop("evaluate_actions", None)
            model.rollout_buffer.__dict__.pop("compute_returns_and_advantage", None)
        for handle in foreign_hook:
            handle.remove()

    if failure is not None:
        with pytest.raises((ValueError, RuntimeError)):
            producer.fit_allocation_ppo_schedule(
                env,
                total_timesteps=12,
                seed=7,
                training_protocol=protocol(n_steps=6, batch_size=2, n_epochs=1),
                transition_trace=trace,
            )
        assert not trace.complete
        if failure == "logger_ordinal_bool":
            assert models[0]._n_updates == 1 and len(updates) == 1
        assert all(
            child._transition_recorder is None for child in env.training_environments
        )
        assert set(models[0].policy.optimizer._optimizer_step_post_hooks) == {
            foreign_hook[0].id
        }
        # The observer restored the independent original installed by this test.
        assert models[0].policy.predict_values.__name__ == "bootstrap"
        restore_witnesses()
        assert "predict_values" not in models[0].policy.__dict__
        return
    fit = producer.fit_allocation_ppo_schedule(
        env,
        total_timesteps=12,
        seed=7,
        training_protocol=protocol(
            n_steps=6, batch_size=2, n_epochs=1, target_kl=0.1 if kl_stop else None
        ),
        transition_trace=trace,
    )
    restore_witnesses()
    return env, fit, trace, actual, buffers, updates, bootstraps, native_steps


def test_actual_scheduled_producer_actor_buffer_diagnostics_save_sidecar_reload(
    monkeypatch, tmp_path
):
    from tests.evaluation.test_allocation_preprocessing_runtime import bind_frozen
    from tests.evaluation.test_allocation_scheduled_transition_trace import dated_args
    from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
    from trade_rl.evaluation.rl_allocation.scheduled_transition_trace_io import (
        publish_scheduled_allocation_transition_trace,
        read_scheduled_allocation_transition_trace,
    )
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    env, fit, trace, actual, buffers, updates, bootstraps, native_steps = fit_capture(
        monkeypatch
    )
    assert trace.complete
    events = [json.loads(row) for row in trace.events]
    rows = [row for row in events if row["kind"] == "transition"]
    boundaries = [row for row in events if row["kind"] == "rollout"]
    diagnostics = [row for row in events if row["kind"] == "update"]
    assert (
        len(actual) == len(rows) == 12
        and len(buffers) == len(updates) == len(diagnostics) == 2
    )
    for row, witness in zip(rows, actual, strict=True):
        for key, value in zip(
            ("observation", "action", "value", "log_prob"), witness[:4], strict=True
        ):
            assert row["actor"][key] == value.tobytes().hex()
    for ordinal, (boundary, buffer) in enumerate(zip(boundaries, buffers, strict=True)):
        batch = actual[ordinal * 6 : (ordinal + 1) * 6]
        expected = {
            "observations": np.stack([w[0] for w in batch]),
            "actions": np.stack([w[1] for w in batch]).reshape(6, 1, 1).astype("<f4"),
            "values": np.stack([w[2] for w in batch]).reshape(6, 1),
            "log_probs": np.stack([w[3] for w in batch]).reshape(6, 1),
            "rewards": np.array(
                [w[0] for w in native_steps[ordinal * 6 : (ordinal + 1) * 6]], "<f4"
            ).reshape(6, 1),
            "episode_starts": np.stack([w[4] for w in batch]).reshape(6, 1),
        }
        for name, value in expected.items():
            assert buffer[name].dtype == value.dtype
            assert buffer[name].shape == value.shape
            assert buffer[name].tobytes() == value.tobytes()
        assert boundary["buffer_digest"] == independent_buffer_pin(buffer)
        assert boundary["advantages"] == buffer["advantages"].tobytes().hex()
        assert boundary["returns"] == buffer["returns"].tobytes().hex()
    assert len(bootstraps) == 2
    for boundary, (observation, value) in zip(boundaries, bootstraps, strict=True):
        assert boundary["bootstrap"] == {
            "observation": observation.tobytes().hex(),
            "value": value.tobytes().hex(),
        }
    for row, (before, after, logged) in zip(diagnostics, updates, strict=True):
        for name, value in row["diagnostics"].items():
            assert value == float(logged["train/" + name])
        assert row["optimizer_steps"] == 3 and row["epochs"] == 1
        assert_independent_parameter_update(row, before, after)
        wrong = deepcopy(row)
        wrong["parameters_before"] = {"actor": "f" * 64, "critic": "e" * 64}
        wrong["parameters_after"] = {"actor": "d" * 64, "critic": "c" * 64}
        with pytest.raises(AssertionError):
            assert_independent_parameter_update(wrong, before, after)
        for prefix in ("mlp_extractor.policy_net.", "mlp_extractor.value_net."):
            assert any(
                not np.array_equal(before[name], after[name])
                for name in before
                if name.startswith(prefix)
            )
    assert (
        boundaries[0]["done"] is False
        and boundaries[0]["current"]["window_ordinal"] == 1
        and boundaries[0]["current"]["index"] == 8
    )
    assert (
        boundaries[-1]["done"] is True
        and boundaries[-1]["current"]["window_ordinal"] == 0
        and boundaries[-1]["current"]["index"] == 6
    )
    assert rows[-1]["old"]["window_ordinal"] == 2 and rows[-1]["next"]["episode"] == 3
    policy = fit.inference_policy()
    saved = tmp_path / "policy"
    pin = save_allocation_policy(saved, policy)
    sidecar = tmp_path / "sidecar"
    trace_pin = publish_scheduled_allocation_transition_trace(
        trace, sidecar, bundle_root=saved, expected_bundle_digest=pin
    )
    manifest = read_scheduled_allocation_transition_trace(
        sidecar,
        expected_digest=trace_pin,
        bundle_root=saved,
        expected_bundle_digest=pin,
        require_learner_diagnostics=True,
    )
    assert [w["actor_count"] for w in manifest["input_summary"]["windows"]] == [4, 4, 4]
    assert [w["reset_count"] for w in manifest["input_summary"]["windows"]] == [2, 1, 1]
    loaded = load_allocation_policy(
        saved,
        expected_digest=pin,
        expected_recipe_digest=env.template_env.recipe_digest,
    )
    args = bind_frozen(dated_args(4), env.template_env.feature_preprocessing)
    before, after = AllocationTradingEnv(**args), AllocationTradingEnv(**args)
    left, _ = before.reset(seed=17)
    right, _ = after.reset(seed=17)
    for _ in range(4):
        a = policy.action(left, runtime_recipe_digest=before.recipe_digest)
        b = loaded.action(right, runtime_recipe_digest=after.recipe_digest)
        assert a == b
        left, _, _, _, _ = before.step(a)
        right, _, _, _, _ = after.step(b)
    assert (
        before.book.cash == after.book.cash
        and before.book.portfolio_value == after.book.portfolio_value
    )
    assert before.book.quantities.tolist() == after.book.quantities.tolist()
    with pytest.raises(FileExistsError):
        publish_scheduled_allocation_transition_trace(
            trace, sidecar, bundle_root=saved, expected_bundle_digest=pin
        )
    with pytest.raises(ValueError):
        publish_scheduled_allocation_transition_trace(
            trace, saved / "inside", bundle_root=saved, expected_bundle_digest=pin
        )
    from trade_rl.evaluation.rl_allocation import scheduled_transition_trace_io as io

    def failed_publication(*args):
        raise OSError("atomic publication injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(io, "atomic_rename_directory", failed_publication)
        with pytest.raises(OSError, match="publication injected"):
            publish_scheduled_allocation_transition_trace(
                trace,
                tmp_path / "staging-failure",
                bundle_root=saved,
                expected_bundle_digest=pin,
            )
    assert not (tmp_path / "staging-failure").exists()
    assert not list(tmp_path.glob(".staging-failure.staging-*"))
    import os

    alias = tmp_path / "bundle-alias"
    try:
        os.symlink(saved, alias, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"parent symlink alias permission unavailable: {error}")
    else:
        with pytest.raises(ValueError, match="separate"):
            publish_scheduled_allocation_transition_trace(
                trace,
                alias / "inside-alias",
                bundle_root=saved,
                expected_bundle_digest=pin,
            )
        assert not (saved / "inside-alias").exists()
    import torch

    with torch.no_grad():
        next(fit.model.policy.parameters()).add_(1)
    with pytest.raises(ValueError, match="parameters changed"):
        publish_scheduled_allocation_transition_trace(
            trace, tmp_path / "mutated", bundle_root=saved, expected_bundle_digest=pin
        )
    assert not (tmp_path / "mutated").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "last_false",
        "train",
        "source",
        "model",
        "setup",
        "optimizer",
        "observer",
        "finish",
        "partial_attach",
        "logger_missing",
        "logger_stale",
        "logger_ordinal_bool",
        "logger_bool",
        "logger_nonfinite",
        "mode",
    ],
)
def test_native_failed_fit_never_completes_and_removes_observers(monkeypatch, failure):
    fit_capture(monkeypatch, failure=failure)


def test_native_kl_stop_records_zero_returned_calls_without_parameter_changes(
    monkeypatch,
):
    _, fit, trace, _, _, updates, _, _ = fit_capture(monkeypatch, kl_stop=True)
    assert trace.complete
    optimization = fit.receipt["optimization"]
    assert optimization["successful_optimizer_step_calls"] == 0
    assert optimization["optimizer_step_events_digest"] == sha256(b"").hexdigest()
    rows = [
        json.loads(raw) for raw in trace.events if json.loads(raw)["kind"] == "update"
    ]
    assert len(rows) == 2
    for row, (before, after, _) in zip(rows, updates, strict=True):
        assert row["optimizer_steps"] == 0 and row["epochs"] == 1
        assert row["parameters_before"] == row["parameters_after"]
        assert_independent_parameter_update(row, before, after)
        assert all(np.array_equal(before[name], after[name]) for name in before)


def test_none_omitted_and_enabled_preserve_native_model_receipts_rng_and_economics():
    import torch

    from tests.evaluation.test_allocation_scheduled_transition_trace import (
        scheduled_trace_env,
    )
    from tests.simulation.test_stateful_execution_characterization import _normalize
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.evaluation.rl_allocation.scheduled_training import (
        fit_allocation_ppo_schedule,
    )
    from trade_rl.evaluation.rl_allocation.scheduled_transition_trace import (
        ScheduledAllocationTransitionRecorder,
    )

    witnesses = []
    for lane in ("omitted", "none", "enabled"):
        env = scheduled_trace_env()
        kwargs = {} if lane == "omitted" else {"transition_trace": None}
        if lane == "enabled":
            kwargs["transition_trace"] = ScheduledAllocationTransitionRecorder(
                env, learner_diagnostics="native_ppo_update_v1"
            )
        fit = fit_allocation_ppo_schedule(
            env,
            total_timesteps=12,
            seed=7,
            training_protocol=protocol(n_steps=6, batch_size=2, n_epochs=1),
            **kwargs,
        )
        witnesses.append(
            (
                fit.inference_policy().manifest,
                {
                    n: p.detach().clone()
                    for n, p in fit.model.policy.state_dict().items()
                },
                deepcopy(fit.model.policy.optimizer.state_dict()),
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state().clone(),
                [_normalize(child.book) for child in env.training_environments],
                [_normalize(child.order_book) for child in env.training_environments],
            )
        )
    first = witnesses[0]
    for other in witnesses[1:]:
        assert first[0] == other[0]
        assert all(torch.equal(first[1][n], other[1][n]) for n in first[1])
        assert first[2]["param_groups"] == other[2]["param_groups"]
        for index, state in first[2]["state"].items():
            assert all(
                torch.equal(v, other[2]["state"][index][n])
                if torch.is_tensor(v)
                else v == other[2]["state"][index][n]
                for n, v in state.items()
            )
        assert first[3] == other[3]
        assert (
            first[4][0] == other[4][0]
            and np.array_equal(first[4][1], other[4][1])
            and first[4][2:] == other[4][2:]
        )
        assert torch.equal(first[5], other[5])
        assert first[6:] == other[6:]
