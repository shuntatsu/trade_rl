"""Fixed synthetic learnability protocol, never market/economic selection."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from tests.simulation.test_stateful_execution_characterization import _normalize
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")


@pytest.mark.parametrize("lane", [3, 4])
@pytest.mark.parametrize("horizon,budget", [(3, 2), (2, 2), (2, 4)])
def test_actual_transition_trace_binds_native_actor_and_completes_only_after_fit(
    monkeypatch,
    tmp_path,
    lane,
    horizon,
    budget,
):
    import json

    from tests.evaluation.test_allocation_preprocessing_runtime import (
        frozen_args,
        raw_args,
    )
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.evaluation.rl_allocation.transition_trace import (
        AllocationTransitionRecorder,
    )

    args = frozen_args(stop=6 + horizon)
    env = AllocationTradingEnv(**(raw_args(args) if lane == 3 else args))
    trace = AllocationTransitionRecorder(env)
    module, build, actual = trainer(), trainer().build_allocation_ppo, []

    def instrument(*a, **k):
        model = build(*a, **k)
        forward = model.policy.forward

        def actor(obs, *args, **kwargs):
            result = forward(obs, *args, **kwargs)
            actual.append(
                tuple(x.detach().cpu().numpy().copy() for x in (obs, *result))
            )
            return result

        model.policy.forward = actor
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", instrument)
    policy = module.fit_allocation_ppo(
        env,
        total_timesteps=budget,
        seed=0,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
        transition_trace=trace,
    )
    rows = [json.loads(raw) for raw in trace.events]
    transitions = [row for row in rows if row["kind"] == "transition"]
    boundaries = [row for row in rows if row["kind"] == "rollout"]
    for row, tensors in zip(transitions, actual):
        for key, values, dtype in zip(
            ("observation", "action", "value", "log_prob"),
            tensors,
            ("<f4", "<i8", "<f4", "<f4"),
        ):
            assert row["actor"][key] == values.astype(dtype).tobytes().hex()
    assert np.frombuffer(bytes.fromhex(rows[0]["actor"]["observation"]), "<f4")[0] == (
        4 if lane == 3 else 3
    )
    assert rows[0]["facts"]["proposal"]["decision"]["feature_values"] == [4.0]
    assert transitions[1]["done"] == (horizon == 2)
    if horizon == 2:
        assert not np.frombuffer(bytes.fromhex(transitions[1]["terminal"]), "<f4").any()
    assert np.frombuffer(bytes.fromhex(boundaries[0]["new_observation"]), "<f4")[0] == (
        (4 if lane == 3 else 3) if horizon == 2 else (8 if lane == 3 else 7)
    )
    assert (
        boundaries[0]["index"] == (6 if horizon == 2 else 8)
        and boundaries[0]["last_transition_index"] == 8
    )
    assert [r["sequence"] for r in transitions] == list(range(budget))
    assert [r["episode"] for r in transitions] == (
        [0, 0, 1, 1] if budget == 4 else [0, 0]
    )
    assert [r["rollout"] for r in boundaries] == list(range(budget // 2))
    assert trace.complete and trace.training_manifest == policy.manifest
    assert env._transition_recorder is None
    from trade_rl.evaluation.rl_allocation.transition_trace_io import (
        publish_allocation_transition_trace,
        read_allocation_transition_trace,
    )
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    bundle_root, trace_root = tmp_path / "bundle", tmp_path / "separate" / "trace"
    bundle_pin = save_allocation_policy(bundle_root, policy)
    bundle_bytes = {p.name: p.read_bytes() for p in bundle_root.iterdir()}
    with pytest.raises(ValueError, match="separate|bundle"):
        publish_allocation_transition_trace(
            trace,
            bundle_root / "trace",
            bundle_root=bundle_root,
            expected_bundle_digest=bundle_pin,
        )
    assert {p.name: p.read_bytes() for p in bundle_root.iterdir()} == bundle_bytes
    assert not (bundle_root / "trace").exists()
    assert not list(bundle_root.glob(".*.staging-*"))
    detour = bundle_root / "unused" / ".." / ".." / "outside"
    detour_pin = publish_allocation_transition_trace(
        trace, detour, bundle_root=bundle_root, expected_bundle_digest=bundle_pin
    )
    assert set(p.name for p in bundle_root.iterdir()) == set(bundle_bytes)
    assert {p.name: p.read_bytes() for p in bundle_root.iterdir()} == bundle_bytes
    assert not (bundle_root / "unused").exists()
    assert (
        read_allocation_transition_trace(
            detour.resolve(),
            expected_digest=detour_pin,
            bundle_root=bundle_root,
            expected_bundle_digest=bundle_pin,
        )["status"]
        == "COMPLETE"
    )
    pin = publish_allocation_transition_trace(
        trace, trace_root, bundle_root=bundle_root, expected_bundle_digest=bundle_pin
    )
    receipt = read_allocation_transition_trace(
        trace_root,
        expected_digest=pin,
        bundle_root=bundle_root,
        expected_bundle_digest=bundle_pin,
    )
    assert receipt["status"] == "COMPLETE" and receipt["transition_count"] == budget
    assert set(p.name for p in bundle_root.iterdir()) == {"manifest.json", "policy.zip"}
    from copy import deepcopy
    from hashlib import sha256

    from trade_rl.artifacts import canonical_json_bytes

    original_lines = (trace_root / "transitions.jsonl").read_bytes()
    for change in (
        "ordinal",
        "bool",
        "action",
        "reward",
        "feature",
        "book",
        "vector",
        "extra",
        "terminal",
        "boundary",
        "missing_admission",
    ):
        bad = deepcopy(rows)
        first = bad[0]
        if change == "ordinal":
            first["sequence"] = 1
        elif change == "bool":
            first["sequence"] = False
        elif change == "action":
            first["actor"]["action_code"] = (first["actor"]["action_code"] + 1) % 4
        elif change == "reward":
            first["actor"]["reward"] = np.array([1], "<f4").tobytes().hex()
        elif change == "feature":
            first["actor"]["observation"] = (
                np.ones((1, receipt["width"]), "<f4").tobytes().hex()
            )
        elif change == "book":
            first["facts"]["book"]["cash"] += 1
        elif change == "vector":
            first["facts"]["execution"]["cost_by_symbol"] = [[0.0]]
        elif change == "extra":
            first["facts"]["execution"]["extra"] = 0
        elif change == "terminal":
            bad[1]["terminal"] = np.ones(receipt["width"], "<f4").tobytes().hex()
        elif change == "boundary":
            bad[2]["buffer_digest"] = "0" * 64
        else:
            bad.pop()
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in bad)
        changed = receipt | {"transitions_sha256": sha256(raw).hexdigest()}
        (trace_root / "transitions.jsonl").write_bytes(raw)
        (trace_root / "manifest.json").write_bytes(canonical_json_bytes(changed))
        with pytest.raises(ValueError):
            read_allocation_transition_trace(
                trace_root,
                expected_digest=content_digest(changed),
                bundle_root=bundle_root,
                expected_bundle_digest=bundle_pin,
            )
    (trace_root / "transitions.jsonl").write_bytes(original_lines)
    (trace_root / "manifest.json").write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(FileExistsError):
        publish_allocation_transition_trace(
            trace,
            trace_root,
            bundle_root=bundle_root,
            expected_bundle_digest=bundle_pin,
        )
    import os
    import shutil
    from pathlib import Path

    export = os.environ.get("ALLOCATION_TRACE_EXPORT_ROOT")
    if export and horizon == 2 and budget == 2:
        destination = Path(export) / f"bundle-v{lane}"
        destination.mkdir(parents=True)
        shutil.copytree(bundle_root, destination / "bundle")
        shutil.copytree(trace_root, destination / "trace")
        (destination / "pins.json").write_bytes(
            canonical_json_bytes(
                {
                    "bundle_digest": bundle_pin,
                    "trace_digest": pin,
                    "finite_protocol": "synthetic_T2_no_market_v1",
                }
            )
        )


@pytest.mark.parametrize(
    "failure", ["last_callback", "optimizer", "post_source", "post_protocol"]
)
def test_transition_trace_never_completes_stopped_or_invalid_fit(monkeypatch, failure):
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.evaluation.rl_allocation.transition_trace import (
        AllocationTransitionRecorder,
    )

    env = AllocationTradingEnv(**frozen_args(stop=8))
    trace = AllocationTransitionRecorder(env)
    module, build, models = trainer(), trainer().build_allocation_ppo, []
    if failure == "last_callback":
        factory = trace.callback

        def stopping(base):
            child = factory(base)
            step = child._on_step

            def reject():
                result = step()
                return result and child.model.num_timesteps != 2

            child._on_step = reject
            return child

        monkeypatch.setattr(trace, "callback", stopping)

    def instrument(*a, **k):
        model = build(*a, **k)
        models.append(model)
        if failure == "optimizer":

            def explode(*a, **k):
                raise RuntimeError("optimizer after admission")

            model.policy.optimizer.step = explode
        elif failure.startswith("post_"):
            learn = model.learn

            def changed(**kwargs):
                result = learn(**kwargs)
                if failure == "post_source":
                    features = env.dataset.features.copy()
                    features[0, 0, 0] = 9
                    env.dataset = env.executor.dataset = replace(
                        env.dataset, features=features
                    )
                else:
                    model.gamma = 0.5
                return result

            model.learn = changed
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", instrument)
    with pytest.raises((ValueError, RuntimeError)):
        module.fit_allocation_ppo(
            env,
            total_timesteps=2,
            training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
            transition_trace=trace,
        )
    assert not trace.complete and env._transition_recorder is None
    assert not models[0].policy.optimizer._optimizer_step_post_hooks
    with pytest.raises(ValueError, match="successful final"):
        _ = trace.training_manifest
    if failure == "last_callback":
        assert models[0].num_timesteps == 2 and models[0].rollout_buffer.pos == 1


def test_optional_trace_preserves_actual_training_bytes_native_economics_and_rng(
    monkeypatch,
):
    import json
    import random
    from types import SimpleNamespace

    import torch

    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.artifacts import canonical_json_bytes
    from trade_rl.evaluation.rl_allocation.transition_trace import (
        AllocationTransitionRecorder,
    )

    args = frozen_args(stop=8)
    declaration = protocol(n_steps=2, batch_size=2, n_epochs=1)
    first = AllocationTradingEnv(**args)
    ordinary = trainer().fit_allocation_ppo(
        first,
        total_timesteps=2,
        seed=7,
        training_protocol=declaration,
        transition_trace=None,
    )
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    second = AllocationTradingEnv(**args)
    trace = AllocationTransitionRecorder(second)
    observed = trainer().fit_allocation_ppo(
        second,
        total_timesteps=2,
        seed=7,
        training_protocol=declaration,
        transition_trace=trace,
    )
    assert canonical_json_bytes(ordinary.manifest) == canonical_json_bytes(
        observed.manifest
    )
    assert torch.equal(rng[0], torch.get_rng_state())
    current = np.random.get_state()
    assert (
        rng[1][0] == current[0]
        and np.array_equal(rng[1][1], current[1])
        and rng[1][2:] == current[2:]
    )
    assert rng[2] == random.getstate()
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            ordinary.model.policy.parameters(), observed.model.policy.parameters()
        )
    )
    raw = json.loads(trace.events[0])
    assert raw["facts"]["proposal"]["raw_action"] == raw["actor"]["action_code"]
    # Fixed identical actions compare the whole native execution/book/order trace.
    left, right = AllocationTradingEnv(**args), AllocationTradingEnv(**args)
    left.reset()
    right.reset()
    # Use the pure freeze helper without installing a collector for replay.
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        freeze_allocation_execution,
    )

    right._transition_recorder = SimpleNamespace(
        freeze_execution=lambda result: freeze_allocation_execution(right, result)
    )
    for action in (3, 0):
        a, b = left.step(action), right.step(action)
        b[4].pop("transition_trace")
        assert _normalize(a) == _normalize(b)


@pytest.mark.parametrize("horizon,budget", [(3, 2), (2, 2), (2, 4)])
def test_actual_frozen_features_reach_actor_critic_autoreset_and_v4_bundle(
    tmp_path, monkeypatch, horizon, budget
):
    import hashlib
    import json

    from tests.evaluation.test_allocation_preprocessing_runtime import (
        bind_frozen,
        frozen_args,
    )
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    args = frozen_args(stop=6 + horizon)
    args["observation_schema"] = replace(
        args["observation_schema"], max_active_orders=1
    )
    args = bind_frozen(args, args["feature_preprocessing"])
    env = AllocationTradingEnv(**args)
    module, actors, boundaries = trainer(), [], []
    build = module.build_allocation_ppo

    def traced(*a, **k):
        model = build(*a, **k)
        forward, critic = model.policy.forward, model.policy.predict_values

        def actor(obs, *a, **k):
            assert obs[0, 0].item() == 3 + 2 * (env.index - 6)
            actors.append(
                (
                    dict(
                        phase="actor",
                        index=env.index,
                        episode_start=bool(model._last_episode_starts[0]),
                    ),
                    obs.detach().cpu().numpy().astype("<f4").tobytes(),
                )
            )
            return forward(obs, *a, **k)

        def boundary(obs):
            done = bool(model._last_episode_starts[0])
            assert obs[0, 0].item() == (3 if done else 7)
            boundaries.append(
                (
                    dict(
                        phase="rollout_boundary",
                        index=env.index,
                        last_transition_index=8,
                        done=done,
                    ),
                    obs.detach().cpu().numpy().astype("<f4").tobytes(),
                )
            )
            return critic(obs)

        model.policy.forward, model.policy.predict_values = actor, boundary
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", traced)
    policy = module.fit_allocation_ppo(
        env,
        total_timesteps=budget,
        seed=0,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=2),
    )
    raw = policy.manifest
    receipt = raw["training"]["observation_consumption"]
    assert raw["schema"] == "allocation_ppo_inference_bundle_v4"
    assert raw["training"]["schema"] == "allocation_ppo_training_receipt_v4"
    assert receipt["schema"] == "allocation_ppo_observation_consumption_v3"
    assert receipt["preprocessing_digest"] == args["feature_preprocessing"].digest

    def digest(events):
        result = hashlib.sha256()
        for facts, values in events:
            header = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
            result.update(
                len(header).to_bytes(8, "big")
                + header
                + len(values).to_bytes(8, "big")
                + values
            )
        return result.hexdigest()

    terminal = [
        (dict(phase="terminal_sentinel", index=8), np.zeros(47, "<f4").tobytes())
    ] * (budget // 2 if horizon == 2 else 0)
    assert (
        receipt["actor_digest"] == digest(actors) and receipt["actor_count"] == budget
    )
    assert receipt["rollout_boundary_digest"] == digest(boundaries)
    assert receipt["terminal_digest"] == digest(terminal) and receipt[
        "terminal_count"
    ] == len(terminal)
    pin = save_allocation_policy(tmp_path / "v4", policy)
    loaded = load_allocation_policy(
        tmp_path / "v4", expected_digest=pin, expected_recipe_digest=env.recipe_digest
    )
    assert loaded.manifest == policy.manifest | {
        "policy_sha256": hashlib.sha256(
            (tmp_path / "v4" / "policy.zip").read_bytes()
        ).hexdigest()
    }
    assert rollout(
        AllocationTradingEnv(**args),
        lambda o: policy.action(o, runtime_recipe_digest=env.recipe_digest),
    ) == rollout(
        AllocationTradingEnv(**args),
        lambda o: loaded.action(o, runtime_recipe_digest=env.recipe_digest),
    )
    loaded.model.policy.optimizer.param_groups[0]["eps"] = 1e-8
    with pytest.raises(ValueError, match="protocol"):
        loaded.action(env.reset()[0], runtime_recipe_digest=env.recipe_digest)
    loaded.model.policy.optimizer.param_groups[0]["eps"] = 1e-5
    loaded.model._n_updates += 1
    with pytest.raises(ValueError, match="epoch iterations"):
        loaded.action(env.reset()[0], runtime_recipe_digest=env.recipe_digest)


def test_actual_training_detects_changed_unseen_prefix_after_learning(monkeypatch):
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args
    from tests.strategies.test_allocation_protocol_receipt import protocol

    env = AllocationTradingEnv(**frozen_args())
    module, build = trainer(), trainer().build_allocation_ppo

    def altered(*a, **k):
        model = build(*a, **k)
        learn = model.learn

        def learns(**kwargs):
            result = learn(**kwargs)
            changed = env.dataset.features.copy()
            changed[0, 0, 0] = 9
            env.dataset = env.executor.dataset = replace(env.dataset, features=changed)
            return result

        model.learn = learns
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", altered)
    with pytest.raises(ValueError, match="prefix"):
        module.fit_allocation_ppo(
            env,
            total_timesteps=2,
            training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=2),
        )


@pytest.mark.parametrize("steps,budget,expected", [(2, 2, 2), (2, 4, 4), (4, 4, 4)])
def test_explicit_protocol_counts_actual_adam_calls_and_final_update(
    tmp_path, monkeypatch, steps, budget, expected
):
    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    module = trainer()
    env = protocol_env(steps=steps)
    declaration = protocol(n_steps=steps, batch_size=2, n_epochs=2)
    original = module.build_allocation_ppo
    calls, built = [], []

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        built.append(model)
        step = model.policy.optimizer.step

        def returned_step(*a, **k):
            result = step(*a, **k)
            calls.append(model.num_timesteps)
            return result

        monkeypatch.setattr(model.policy.optimizer, "step", returned_step)
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", build)
    policy = module.fit_allocation_ppo(
        env, total_timesteps=budget, seed=7, training_protocol=declaration
    )
    raw = policy.manifest
    assert raw["schema"] == "allocation_ppo_inference_bundle_v3"
    assert raw["recipe"] == env.recipe and raw["recipe_digest"] == env.recipe_digest
    assert raw["training"]["protocol"] == declaration.payload()
    receipt = raw["training"]["optimization"]
    assert len(calls) == receipt["successful_optimizer_step_calls"] == expected
    assert receipt["rollout_update_count"] == budget // steps
    assert receipt["epoch_iterations"] == budget // steps * 2
    assert (
        calls[-1] == budget and not built[0].policy.optimizer._optimizer_step_post_hooks
    )
    destination = tmp_path / "protocol"
    digest = save_allocation_policy(destination, policy)
    loaded = load_allocation_policy(
        destination, expected_digest=digest, expected_recipe_digest=env.recipe_digest
    )
    assert loaded.manifest["training"] == raw["training"]
    assert rollout(
        protocol_env(steps=steps),
        lambda o: policy.action(o, runtime_recipe_digest=env.recipe_digest),
    ) == rollout(
        protocol_env(steps=steps),
        lambda o: loaded.action(o, runtime_recipe_digest=env.recipe_digest),
    )
    loaded.model.policy.optimizer.param_groups[0]["eps"] = 1e-8
    with pytest.raises(ValueError, match="protocol"):
        loaded.action(env.reset()[0], runtime_recipe_digest=env.recipe_digest)


def test_explicit_protocol_resolves_all_constructor_kwargs(monkeypatch):
    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol

    module = trainer()
    sb3 = import_module("stable_baselines3")
    original, captured = sb3.PPO, []

    def construct(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sb3, "PPO", construct)
    declaration = protocol(
        n_steps=2,
        batch_size=2,
        n_epochs=2,
        learning_rate=0.003,
        clip_range=0.17,
        clip_range_vf=0.12,
        normalize_advantage=False,
        ent_coef=0.01,
        vf_coef=0.6,
        max_grad_norm=0.7,
        pi_layers=(8,),
        vf_layers=(4,),
        adam_betas=(0.8, 0.95),
        adam_eps=2e-5,
        adam_weight_decay=0.01,
        adam_amsgrad=True,
    )
    model = module.build_allocation_ppo(protocol_env(), training_protocol=declaration)
    p, k = declaration.payload(), captured[0]
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "learning_rate",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "sde_sample_freq",
        "rollout_buffer_kwargs",
        "stats_window_size",
        "tensorboard_log",
        "verbose",
        "_init_setup_model",
    ):
        assert k[name] == p["ppo"][name]
    assert k["policy_kwargs"]["net_arch"] == {"pi": [8], "vf": [4]}
    expected = {
        name: value
        for name, value in p["optimizer"].items()
        if name not in {"class", "learning_rate_source"}
    }
    expected["betas"] = (0.8, 0.95)
    assert k["policy_kwargs"]["optimizer_kwargs"] == expected
    assert model.policy.optimizer.param_groups[0]["eps"] == 2e-5


def test_native_kl_stop_skips_adam_but_enters_epoch_and_keeps_state(monkeypatch):
    import torch

    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol

    module = trainer()
    original, snapshots = module.build_allocation_ppo, []

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        snapshots.append([p.detach().clone() for p in model.policy.parameters()])
        forward, evaluate = model.policy.forward, model.policy.evaluate_actions

        def actor(*a, **k):
            actions, values, logprob = forward(*a, **k)
            return actions, values, torch.zeros_like(logprob)

        def likelihood(*a, **k):
            values, logprob, entropy = evaluate(*a, **k)
            return values, torch.ones_like(logprob), entropy

        monkeypatch.setattr(model.policy, "forward", actor)
        monkeypatch.setattr(model.policy, "evaluate_actions", likelihood)
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", build)
    policy = module.fit_allocation_ppo(
        protocol_env(),
        total_timesteps=2,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=2, target_kl=0.1),
    )
    receipt = policy.manifest["training"]["optimization"]
    assert receipt["successful_optimizer_step_calls"] == 0
    assert receipt["epoch_iterations"] == policy.model._n_updates == 1
    assert not policy.model.policy.optimizer.state
    assert all(
        torch.equal(a, b)
        for a, b in zip(snapshots[0], policy.model.policy.parameters())
    )
    from trade_rl.strategies.rl.allocation_model import validate_allocation_model

    policy.model._n_updates = True
    with pytest.raises(ValueError, match="epoch iterations"):
        validate_allocation_model(policy.model, policy.manifest)


def test_explicit_protocol_hook_is_removed_on_learning_exception(monkeypatch):
    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol

    module = trainer()
    original, models = module.build_allocation_ppo, []

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)

        def fail(**_):
            assert len(model.policy.optimizer._optimizer_step_post_hooks) == 1
            raise RuntimeError("synthetic train exception")

        monkeypatch.setattr(model, "learn", fail)
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", build)
    with pytest.raises(RuntimeError, match="synthetic"):
        module.fit_allocation_ppo(
            protocol_env(),
            total_timesteps=2,
            training_protocol=protocol(n_steps=2, batch_size=2),
        )
    assert not models[0].policy.optimizer._optimizer_step_post_hooks


def test_protocol_checks_built_activation_and_loaded_backend_without_optional_imports(
    monkeypatch,
):
    import torch

    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.strategies.rl import allocation_model as validator

    declaration = protocol(n_steps=2, batch_size=2, n_epochs=2)
    model = trainer().build_allocation_ppo(
        protocol_env(), training_protocol=declaration
    )
    old = model.policy.mlp_extractor.policy_net[1]
    model.policy.mlp_extractor.policy_net[1] = torch.nn.ReLU()
    with pytest.raises(ValueError, match="protocol"):
        validator.validate_allocation_protocol_model(model, declaration)
    model.policy.mlp_extractor.policy_net[1] = old
    monkeypatch.setattr(validator, "version", lambda _: "9.0.0")
    with pytest.raises(ValueError, match="release"):
        validator.validate_allocation_protocol_model(model, declaration)


@pytest.mark.parametrize("corruption", ["child", "dimension", "width"])
def test_protocol_checks_actual_flatten_extractor_child(corruption):
    import torch

    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.strategies.rl.allocation_model import (
        validate_allocation_protocol_model,
    )

    declaration = protocol(n_steps=2, batch_size=2, n_epochs=2)
    model = trainer().build_allocation_ppo(
        protocol_env(), training_protocol=declaration
    )
    extractor = model.policy.features_extractor
    if corruption == "child":
        extractor.flatten = torch.nn.Tanh()
    elif corruption == "dimension":
        extractor.flatten = torch.nn.Flatten(start_dim=0)
    else:
        extractor._features_dim = 70
    with pytest.raises(ValueError, match="protocol"):
        validate_allocation_protocol_model(model, declaration)


def test_protocol_rejects_duplicate_actual_optimizer_parameter():
    from tests.evaluation.test_allocation_training_source import protocol_env
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.strategies.rl.allocation_model import (
        validate_allocation_protocol_model,
    )

    declaration = protocol(n_steps=2, batch_size=2, n_epochs=2)
    model = trainer().build_allocation_ppo(
        protocol_env(), training_protocol=declaration
    )
    parameters = model.policy.optimizer.param_groups[0]["params"]
    parameters.append(parameters[0])
    with pytest.raises(ValueError, match="protocol"):
        validate_allocation_protocol_model(model, declaration)


@pytest.mark.parametrize("v2", [False, True])
def test_none_protocol_retains_legacy_bytes_sources_and_actions(v2):
    from tests.evaluation.test_allocation_training_source import protocol_env
    from trade_rl.artifacts import canonical_json_bytes

    def env():
        if v2:
            return protocol_env()
        args = parameters(stop=10)
        args["bound"] = replace(
            args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
        )
        return AllocationTradingEnv(**args)

    module = trainer()
    old = module.fit_allocation_ppo(env(), total_timesteps=2, seed=7)
    explicit = module.fit_allocation_ppo(
        env(), total_timesteps=2, seed=7, training_protocol=None
    )
    assert canonical_json_bytes(old.manifest) == canonical_json_bytes(explicit.manifest)
    assert old.manifest["training"]["ppo"] == {
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "n_steps": 2,
        "batch_size": 2,
        "n_epochs": 10,
        "learning_rate": 0.002,
        "net_arch": {"pi": [32, 32], "vf": [32, 32]},
        "device": "cpu",
    }
    assert set(old.manifest["training"]) == {
        "seed",
        "requested_timesteps",
        "actual_timesteps",
        "clock",
        "objective",
        "source",
        "ppo",
    } | ({"observation_consumption"} if v2 else set())
    first, second = env(), env()
    assert rollout(
        first, lambda o: old.action(o, runtime_recipe_digest=first.recipe_digest)
    ) == rollout(
        second, lambda o: explicit.action(o, runtime_recipe_digest=second.recipe_digest)
    )


def trainer():
    try:
        return import_module("trade_rl.evaluation.rl_allocation.training")
    except ModuleNotFoundError as error:
        if error.name == "trade_rl.evaluation.rl_allocation.training":
            pytest.fail("The actual allocation PPO training capability is missing")
        raise


def synthetic_episode(schedule_seed, *, price_response=0.01, fee=0.0005):
    # G2 protocol fixed before fitting: 16 causal sign decisions, .01 price
    # response, .0005 actual fee; four flat mature prefix labels fit a cash
    # forecast. Held-out schedules 100/101/102 are not used by RL fitting.
    rng = np.random.default_rng(schedule_seed)
    cue = rng.choice([-1.0, 1.0], size=23)
    close = np.full((23, 1), 100.0)
    for i in range(6, 22):
        close[i + 1, 0] = close[i, 0] * (1.0 + price_response * cue[i])
    opens = close.copy()
    opens[7:, 0] = close[6:-1, 0]
    shape = close.shape
    dataset = MarketDataset(
        dataset_id=content_digest({"synthetic_cue_episode_v1": schedule_seed}),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.timedelta64(schedule_seed if schedule_seed >= 100 else 0, "D")
        + np.arange(23) * np.timedelta64(1, "h"),
        features=np.stack([np.ones(23), cue], axis=1)[:, None, :].astype(np.float32),
        feature_names=("constant", "current_cue"),
        feature_available=np.ones((23, 1, 2), dtype=bool),
        global_features=np.zeros(shape),
        global_feature_names=("regime",),
        symbols=("S0",),
        volume_units=(VolumeUnit.BASE_ASSET,),
        periods_per_year=8760,
        open=opens,
        close=close,
        mark_price=close,
        index_price=close,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        volume=np.full(shape, 100_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        split_factor=np.ones(shape),
    )
    times = dataset.timestamps
    block = ForecastBlock(
        times[5], times[5] + np.timedelta64(15, "m"), times[6], times[22]
    )
    stream = fit_prequential_simple_ridge(
        dataset, blocks=(block,), feature_indices=(0,), horizon_hours=1
    )
    assert all(packet.expected_simple_return == 0.0 for packet in stream.packets)
    costs = tuple(
        HorizonCostEstimates(
            "S0",
            times[i],
            times[i],
            times[i + 1],
            "declared-synthetic-fee",
            buy_cost=fee,
            sell_cost=fee,
        )
        for i in range(6, 22)
    )
    action = AllocationActionContract("direct", 0.5)
    allocator = AfterCostTargetAllocator()
    cost = replace(ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0)
    risk = PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=None)
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        datetime(2026, 1, 1, 6, tzinfo=UTC)
        + timedelta(days=schedule_seed if schedule_seed >= 100 else 0),
        datetime(2026, 1, 1, 22, tzinfo=UTC)
        + timedelta(days=schedule_seed if schedule_seed >= 100 else 0),
        "marked_continuation",
        MarketExecutor(
            dataset, cost, insolvency_valuation="retain_debt"
        ).execution_policy_digest,
        content_digest(risk),
        allocation_recipe_digest(
            action,
            ("current_cue",),
            allocator=allocator,
            expected_horizon_seconds=3600,
            runtime_profile=AllocationRuntimeProfile(
                MarketExecutor(
                    dataset, cost, insolvency_valuation="retain_debt"
                ).execution_policy_digest,
                content_digest(risk),
                1000.0,
                "USD",
                3600,
                16 * 3600,
            ),
        ),
    )
    clock = FinancialClockContract(
        3600, 3600, 3600, 16 * 3600, 32, 1.0, 0.95, "equity_delta_v1"
    )
    return AllocationTradingEnv(
        dataset,
        stream=stream,
        estimates=costs,
        bound=BoundObjectiveClock(objective, clock),
        action_contract=action,
        allocator=allocator,
        execution_cost=cost,
        risk_config=risk,
        feature_indices=(1,),
        symbol_index=0,
        start_index=6,
        stop_index=22,
        account_id="synthetic-independent-S0",
    )


def rollout(env, action):
    observation, _ = env.reset(seed=41)
    trace, reward, aligned = [], 0.0, 0
    while True:
        cue = (
            float(env.dataset.features[env.index, 0, 1])
            if env.dataset.n_features == 2
            else 0.0
        )
        observation, increment, done, truncated, info = env.step(action(observation))
        assert not truncated
        reward += increment
        aligned += int(float(env.book.weights[0]) * cue > 0)
        trace.append(_normalize(info))
        if done:
            return reward, aligned / len(trace), trace


@pytest.mark.parametrize("seed", [0, 7, 17])
def test_actual_ppo_learns_current_cue_and_frozen_reload_preserves_full_execution(
    seed, tmp_path
):
    module = trainer()
    env = synthetic_episode(10)
    untrained = module.build_allocation_ppo(env, seed=seed)
    initial = [
        rollout(
            synthetic_episode(s),
            lambda x: int(untrained.predict(x, deterministic=True)[0]),
        )[0]
        for s in (100, 101, 102)
    ]
    trained = module.fit_allocation_ppo(env, total_timesteps=4096, seed=seed)
    assert trained.manifest["training"]["actual_timesteps"] == 4096
    assert trained.manifest["training"]["requested_timesteps"] == 4096
    assert trained.manifest["training"]["clock"]["gamma"] == 1.0
    assert trained.manifest["training"]["source"]["decision_indices"] == list(
        range(6, 22)
    )
    assert trained.manifest["training"]["source"]["decision_counts"] == [256] * 16
    scores = []
    for schedule, initial_reward in zip((100, 101, 102), initial):
        check = synthetic_episode(schedule)
        result = rollout(
            check,
            lambda x: trained.action(x, runtime_recipe_digest=check.recipe_digest),
        )
        # Fixed acceptance before fitting: cue-conditioned EFFECTIVE exposure,
        # positive after-cost gain over cash, and improvement over initialization.
        assert result[1] >= 0.8
        assert result[0] > 0.025
        assert result[0] - initial_reward > 0.02
        scores.append(result)
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    digest = save_allocation_policy(tmp_path / "policy", trained)
    loaded = load_allocation_policy(
        tmp_path / "policy",
        expected_digest=digest,
        expected_recipe_digest=env.recipe_digest,
    )
    for schedule, expected in zip((100, 101, 102), scores):
        check = synthetic_episode(schedule)
        assert (
            rollout(
                check,
                lambda x: loaded.action(x, runtime_recipe_digest=check.recipe_digest),
            )
            == expected
        )
    with pytest.raises(ValueError, match="recipe"):
        loaded.action(
            np.zeros(env.observation_space.shape), runtime_recipe_digest="0" * 64
        )


def test_rollout_boundary_bootstraps_but_declared_horizon_is_terminal():
    module = trainer()
    kwargs = parameters(stop=10)
    kwargs["bound"] = replace(
        kwargs["bound"], clock=replace(kwargs["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**kwargs)
    model = module.build_allocation_ppo(env, seed=0)
    calls = []
    original = model.rollout_buffer.compute_returns_and_advantage

    def capture(last_values, dones):
        calls.append((last_values.detach().cpu().numpy().copy(), dones.copy()))
        original(last_values, dones)

    model.rollout_buffer.compute_returns_and_advantage = capture
    _, callback = model._setup_learn(4)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert env.index == 8
    assert calls[-1][1].tolist() == [False]
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert calls[-1][1].tolist() == [True]
    assert (
        env.index == 6
    )  # genuine terminal auto-reset by DummyVecEnv, not buffer reset


def test_training_budget_cannot_silently_round_to_more_transitions():
    with pytest.raises(ValueError, match="rollout|multiple"):
        trainer().fit_allocation_ppo(synthetic_episode(10), total_timesteps=33, seed=0)


@pytest.mark.parametrize("seed", [0, 7, 17])
def test_actual_ppo_avoids_fee_only_trading_on_held_out_cues(seed):
    # Separately fixed G2 protocol before its first fit: no price response,
    # .005 fee, unchanged 4096-step budget/network and all three policy seeds.
    trained = trainer().fit_allocation_ppo(
        synthetic_episode(10, price_response=0.0, fee=0.005),
        total_timesteps=4096,
        seed=seed,
    )
    for schedule in (100, 101, 102):
        check = synthetic_episode(schedule, price_response=0.0, fee=0.005)
        result = rollout(
            check,
            lambda x: trained.action(x, runtime_recipe_digest=check.recipe_digest),
        )
        assert result[0] == 0.0
        assert check.book.fill_count == 0


def test_observed_source_counts_cover_actual_terminal_resets_only():
    env = AllocationTradingEnv(**parameters(stop=7))
    policy = trainer().fit_allocation_ppo(env, total_timesteps=4, seed=0)
    source = policy.manifest["training"]["source"]
    assert source["start_index"] == 6 and source["stop_index"] == 7
    assert source["decision_indices"] == [6]
    assert source["decision_counts"] == [4]
    assert source["observation_indices"] == [6]


def test_actual_short_rollout_receipt_includes_nonterminal_critic_bootstrap():
    args = parameters(stop=10)
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**args)
    policy = trainer().fit_allocation_ppo(env, total_timesteps=2, seed=0)
    source = policy.manifest["training"]["source"]
    assert source["decision_indices"] == [6, 7]
    assert source["decision_counts"] == [1, 1]
    assert source["observation_indices"] == [6, 7, 8]
    assert not env._terminated and env.index == 8


def test_v2_true_terminal_masks_nonzero_critic_and_live_rollout_bootstraps():
    import torch

    from tests.evaluation.test_allocation_rl_observation_v2 import opt_in

    args = parameters(stop=10)
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**opt_in(args))
    model = trainer().build_allocation_ppo(env, seed=0)
    critic_inputs, results = [], []
    forward = model.policy.forward

    def actor(obs):
        actions, values, log_probs = forward(obs)
        # A binary-exact current value isolates the terminal mask from GAE's
        # ordinary advantage/value cancellation roundoff.
        return actions, torch.zeros_like(values), log_probs

    def critic(obs):
        critic_inputs.append(obs.detach().cpu().numpy().copy())
        return torch.full((1, 1), 7.0)

    original = model.rollout_buffer.compute_returns_and_advantage

    def capture(last_values, dones):
        original(last_values, dones)
        results.append(
            (
                bool(dones[0]),
                float(model.rollout_buffer.returns[-1, 0]),
                float(model.rollout_buffer.rewards[-1, 0]),
            )
        )

    model.policy.predict_values = critic
    model.policy.forward = actor
    model.rollout_buffer.compute_returns_and_advantage = capture
    _, callback = model._setup_learn(4)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert env.index == 8 and results[-1][0] is False
    assert results[-1][1] == pytest.approx(results[-1][2] + 7.0)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert results[-1][0] is True and results[-1][1] == results[-1][2]
    assert (
        env.index == 6
        and len(critic_inputs) == 2
        and all(x.any() for x in critic_inputs)
    )
    infos = model.env.buf_infos[0]
    assert not infos["TimeLimit.truncated"] and not infos["terminal_observation"].any()
    assert infos["terminal_observation"].shape == env.observation_space.shape


@pytest.mark.parametrize("budget", [2, 4])
def test_actual_v2_consumed_tensors_receipt_and_save_load_trace(
    tmp_path, monkeypatch, budget
):
    import hashlib
    import json

    from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    args = parameters(stop=10)
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**opt_in(args))
    module, actor_events, boundary_events = trainer(), [], []
    build = module.build_allocation_ppo

    def traced(*args, **kwargs):
        model = build(*args, **kwargs)
        forward, critic = model.policy.forward, model.policy.predict_values

        def actor(obs, *a, **k):
            actor_events.append(
                (
                    dict(
                        phase="actor",
                        index=env.index,
                        episode_start=bool(model._last_episode_starts[0]),
                    ),
                    obs.detach().cpu().numpy().astype("<f4").tobytes(),
                )
            )
            return forward(obs, *a, **k)

        def boundary(obs):
            done = bool(model._last_episode_starts[0])
            boundary_events.append(
                (
                    dict(
                        phase="rollout_boundary",
                        index=env.index,
                        last_transition_index=8 if len(boundary_events) == 0 else 10,
                        done=done,
                    ),
                    obs.detach().cpu().numpy().astype("<f4").tobytes(),
                )
            )
            return critic(obs)

        model.policy.forward, model.policy.predict_values = actor, boundary
        return model

    monkeypatch.setattr(module, "build_allocation_ppo", traced)
    policy = module.fit_allocation_ppo(env, total_timesteps=budget, seed=0)
    assert policy.manifest["schema"] == "allocation_ppo_inference_bundle_v2"
    receipt = policy.manifest["training"]["observation_consumption"]

    def digest(events):
        result = hashlib.sha256()
        for metadata, raw in events:
            header = json.dumps(
                metadata, separators=(",", ":"), sort_keys=True
            ).encode()
            result.update(
                len(header).to_bytes(8, "big")
                + header
                + len(raw).to_bytes(8, "big")
                + raw
            )
        return result.hexdigest()

    assert receipt["actor_count"] == len(actor_events) == budget
    assert receipt["rollout_boundary_count"] == len(boundary_events) == budget // 2
    assert receipt["actor_digest"] == digest(actor_events)
    assert receipt["rollout_boundary_digest"] == digest(boundary_events)
    terminal = (
        [(dict(phase="terminal_sentinel", index=10), np.zeros((71,), "<f4").tobytes())]
        if budget == 4
        else []
    )
    assert receipt["terminal_count"] == len(terminal) and receipt[
        "terminal_digest"
    ] == digest(terminal)
    assert policy.manifest["training"]["source"]["observation_indices"] == (
        [6, 7, 8, 9] if budget == 4 else [6, 7, 8]
    )
    pin = save_allocation_policy(tmp_path / "v2", policy)
    loaded = load_allocation_policy(
        tmp_path / "v2", expected_digest=pin, expected_recipe_digest=env.recipe_digest
    )
    assert loaded.manifest["training"]["observation_consumption"] == receipt
    assert rollout(
        AllocationTradingEnv(**opt_in(args)),
        lambda x: policy.action(x, runtime_recipe_digest=env.recipe_digest),
    ) == rollout(
        AllocationTradingEnv(**opt_in(args)),
        lambda x: loaded.action(x, runtime_recipe_digest=env.recipe_digest),
    )


def delayed_v4_episode(
    schedule_seed,
    *,
    frozen=None,
    delay=8,
    price_response=0.04,
    fee=0.0005,
    gae_lambda=0.95,
    rollout_steps=32,
):
    """Four causal cue decisions whose economic payoff arrives eight bars later."""
    from tests.evaluation.test_allocation_preprocessing_runtime import bind_frozen
    from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
    from trade_rl.evaluation.rl_allocation import preprocessing

    start, segments = 6, 4
    stop = start + segments * delay
    n_bars = stop + 1
    rng = np.random.default_rng(schedule_seed)
    signs = np.asarray([-1.0, -1.0, 1.0, 1.0])
    rng.shuffle(signs)
    cue = np.zeros(n_bars, dtype=np.float32)
    close = np.full((n_bars, 1), 100.0)
    level = 100.0
    for segment, sign in enumerate(signs):
        decision = start + segment * delay
        cue[decision : decision + delay] = sign
        close[decision : decision + delay, 0] = level
        level *= 1.0 + price_response * sign
        close[decision + delay, 0] = level
    opens = close.copy()
    opens[1:, 0] = close[:-1, 0]
    shape = close.shape
    offset = schedule_seed if schedule_seed >= 100 else 2 * max(schedule_seed - 10, 0)
    dataset = MarketDataset(
        dataset_id=content_digest(
            {
                "synthetic_delayed_credit_v1": schedule_seed,
                "delay": delay,
                "price_response": price_response,
            }
        ),
        timestamps=np.datetime64("2026-02-01", "ns")
        + np.timedelta64(offset, "D")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.stack([np.ones(n_bars), cue], axis=1)[:, None, :].astype(
            np.float32
        ),
        feature_names=("constant", "signal"),
        feature_available=np.ones((n_bars, 1, 2), dtype=bool),
        global_features=np.zeros(shape),
        global_feature_names=("regime",),
        symbols=("S0",),
        volume_units=(VolumeUnit.BASE_ASSET,),
        periods_per_year=8760,
        open=opens,
        close=close,
        mark_price=close,
        index_price=close,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        volume=np.full(shape, 100_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        split_factor=np.ones(shape),
    )
    times = dataset.timestamps
    block = ForecastBlock(
        times[5], times[5] + np.timedelta64(15, "m"), times[start], times[stop]
    )
    stream = fit_prequential_simple_ridge(
        dataset,
        blocks=(block,),
        feature_indices=(0,),
        horizon_hours=1,
    )
    assert all(packet.expected_simple_return == 0.0 for packet in stream.packets)
    estimates = tuple(
        HorizonCostEstimates(
            "S0",
            times[i],
            times[i],
            times[i + 1],
            "declared-delayed-credit-fee",
            buy_cost=fee,
            sell_cost=fee,
        )
        for i in range(start, stop)
    )
    action = AllocationActionContract("direct", 0.5)
    allocator = AfterCostTargetAllocator()
    cost = replace(ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0)
    risk = PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=None)
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    profile = AllocationRuntimeProfile(
        economics,
        content_digest(risk),
        1000.0,
        "USD",
        3600,
        (stop - start) * 3600,
    )
    recipe = allocation_recipe_digest(
        action,
        ("signal",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        runtime_profile=profile,
    )
    base = datetime(2026, 2, 1, tzinfo=UTC) + timedelta(days=offset)
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        base + timedelta(hours=start),
        base + timedelta(hours=stop),
        "marked_continuation",
        economics,
        content_digest(risk),
        recipe,
    )
    clock = FinancialClockContract(
        3600,
        3600,
        3600,
        (stop - start) * 3600,
        rollout_steps,
        1.0,
        gae_lambda,
        "equity_delta_v1",
    )
    args = opt_in(
        dict(
            dataset=dataset,
            stream=stream,
            estimates=estimates,
            bound=BoundObjectiveClock(objective, clock),
            action_contract=action,
            allocator=allocator,
            execution_cost=cost,
            risk_config=risk,
            feature_indices=(1,),
            symbol_index=0,
            start_index=start,
            stop_index=stop,
            account_id="synthetic-delayed-credit-S0",
        )
    )
    if frozen is None:
        frozen = preprocessing.fit_allocation_feature_preprocessing(
            dataset,
            feature_indices=(1,),
            fit_symbol_indices=(0,),
            fit_start=0,
            fit_stop=5,
            fit_as_of=times[5],
            first_decision_index=start,
        )
    args = bind_frozen(args, frozen)
    return AllocationTradingEnv(**args), frozen


def delayed_rollout(env, action):
    observation, _ = env.reset(seed=41)
    reward, correct, cue_count, actions = 0.0, 0, 0, []
    while True:
        cue = float(env.dataset.features[env.index, 0, 1])
        code = int(action(observation))
        observation, increment, done, truncated, _ = env.step(code)
        assert not truncated
        reward += increment
        actions.append(code)
        if cue:
            cue_count += 1
            correct += int(float(env.book.weights[0]) * cue > 0)
        if done:
            return reward, correct / cue_count, tuple(actions), env.book.fill_count


def delayed_training_schedule(
    *,
    gae_lambda=0.95,
    rollout_steps=32,
    delay=8,
):
    from trade_rl.evaluation.rl_allocation.training_schedule import (
        AllocationTrainingScheduleEnv,
        allocation_training_window,
    )
    from trade_rl.strategies.rl.allocation_training_schedule import (
        AllocationTrainingSchedule,
    )

    seeds = (10, 11, 12, 16, 17, 24)
    first, frozen = delayed_v4_episode(
        seeds[0],
        gae_lambda=gae_lambda,
        rollout_steps=rollout_steps,
        delay=delay,
    )
    children = [first]
    for seed in seeds[1:]:
        child, _ = delayed_v4_episode(
            seed,
            frozen=frozen,
            gae_lambda=gae_lambda,
            rollout_steps=rollout_steps,
            delay=delay,
        )
        children.append(child)
    windows = tuple(allocation_training_window(child) for child in children)
    schedule = AllocationTrainingSchedule(
        windows,
        train_window_ids=tuple(window.window_id for window in windows),
    )
    return AllocationTrainingScheduleEnv(schedule, tuple(children)), frozen


def test_delayed_credit_longer_rollout_is_an_isolated_protocol_factor():
    from tests.strategies.test_allocation_protocol_receipt import protocol

    short, _ = delayed_training_schedule(gae_lambda=0.95, rollout_steps=32)
    long, _ = delayed_training_schedule(gae_lambda=0.95, rollout_steps=64)
    short_protocol = protocol(
        n_steps=32, batch_size=32, n_epochs=10, gae_lambda=0.95
    ).payload()
    long_protocol = protocol(
        n_steps=64, batch_size=32, n_epochs=10, gae_lambda=0.95
    ).payload()

    assert short.schedule.digest == long.schedule.digest
    assert short.template_env.recipe_digest == long.template_env.recipe_digest
    short_clock = short.template_env.bound.clock.payload()
    long_clock = long.template_env.bound.clock.payload()
    assert short_clock | {"rollout_steps": 64} == long_clock

    short_ppo, long_ppo = dict(short_protocol["ppo"]), dict(long_protocol["ppo"])
    assert short_ppo.pop("n_steps") == 32
    assert long_ppo.pop("n_steps") == 64
    assert short_ppo == long_ppo
    assert short_protocol["policy"] == long_protocol["policy"]
    assert short_protocol["optimizer"] == long_protocol["optimizer"]


def test_v4_delayed_credit_oracle_has_no_positive_immediate_cue_reward():
    env, _ = delayed_v4_episode(10)
    observation, _ = env.reset(seed=41)
    cue = float(env.dataset.features[env.index, 0, 1])
    action = 3 if cue > 0 else 1
    rewards = []
    observation, reward, done, truncated, _ = env.step(action)
    assert not done and not truncated
    rewards.append(reward)
    for _ in range(7):
        observation, reward, done, truncated, _ = env.step(0)
        assert not truncated
        rewards.append(reward)
    assert rewards[0] < 0.0
    assert max(rewards[:-1]) <= 0.0
    assert rewards[-1] > 0.015
    assert sum(rewards) > 0.015


@pytest.mark.parametrize("seed", [0, 7, 17])
def test_actual_v4_schedule_ppo_learns_delayed_payoff_with_long_rollout_on_held_out_schedules(
    seed,
):
    from tests.strategies.test_allocation_protocol_receipt import protocol
    from trade_rl.evaluation.rl_allocation.scheduled_training import (
        fit_allocation_ppo_schedule,
    )

    env, frozen = delayed_training_schedule(gae_lambda=0.95, rollout_steps=64)
    declaration = protocol(
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gae_lambda=0.95,
    )
    untrained = trainer().build_allocation_ppo(
        env.template_env, seed=seed, training_protocol=declaration
    )
    initial = []
    for schedule in (100, 101, 102):
        check, _ = delayed_v4_episode(
            schedule,
            frozen=frozen,
            gae_lambda=0.95,
            rollout_steps=64,
        )
        initial.append(
            delayed_rollout(
                check,
                lambda x: int(untrained.predict(x, deterministic=True)[0]),
            )[0]
        )

    trained = fit_allocation_ppo_schedule(
        env,
        total_timesteps=4096,
        seed=seed,
        training_protocol=declaration,
    )
    receipt = trained.receipt
    assert receipt["protocol"]["ppo"]["gae_lambda"] == 0.95
    assert receipt["protocol"]["ppo"]["n_steps"] == 64
    assert len(receipt["consumption"]["windows"]) == 6
    assert (
        sum(sum(row["decision_counts"]) for row in receipt["consumption"]["windows"])
        == 4096
    )

    for schedule, initial_reward in zip((100, 101, 102), initial):
        check, _ = delayed_v4_episode(
            schedule,
            frozen=frozen,
            gae_lambda=0.95,
            rollout_steps=64,
        )
        result = delayed_rollout(
            check,
            lambda x: int(trained.model.predict(x, deterministic=True)[0]),
        )
        assert result[1] >= 0.75
        assert result[0] > 0.03
        assert result[0] - initial_reward > 0.02
