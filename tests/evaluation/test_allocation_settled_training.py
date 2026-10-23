"""Fixed synthetic actual-CPU oracles; no market/economic selection."""

import json
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_settled_env import environment
from tests.strategies.test_allocation_protocol_receipt import protocol


def owner(name):
    assert util.find_spec(name), "distinct settled training owner is missing"
    return import_module(name)


def declaration():
    return protocol(
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        pi_layers=(8,),
        vf_layers=(8,),
        learning_rate=1e-12,
    )


def records(trace):
    return [json.loads(raw) for raw in trace.events]


def tensor(record):
    return np.frombuffer(bytes.fromhex(record["hex"]), dtype=record["dtype"]).reshape(
        record["shape"]
    )


def install_seven(monkeypatch, module, *, action=0):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    original = module.build_settled_allocation_ppo

    def actual(*args, **kwargs):
        model = original(*args, **kwargs)
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(7.0)
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-50.0)
            model.policy.action_net.bias[action] = 50.0
        return model

    monkeypatch.setattr(module, "build_settled_allocation_ppo", actual)


@pytest.mark.parametrize(
    "arguments,action",
    [
        ({"lot_size": 1e-20}, 3),
        ({"lot_size": 0.2, "cost_changes": {"lot_size": 0.3}}, 3),
        ({"actor_volume": 1.0}, 3),
        (
            {
                "side": -1,
                "lot_size": 0.1,
                "borrow_rate": 0.1,
                "funding": True,
                "cash_rate": -0.2,
                "dividend": 0.1,
                "cost_changes": {"borrow_rate_multiplier": 1.0},
            },
            1,
        ),
        ({"funding": True, "cash_rate": 0.2, "dividend": 0.1}, 3),
    ],
)
def test_actual_cpu_native_partial_lots_and_signed_carry(
    monkeypatch, arguments, action
):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module, action=action)
    env = environment(**arguments)
    trace = owner(
        "trade_rl.evaluation.rl_allocation.settled_transition_trace"
    ).SettledTransitionRecorder(env)
    if arguments.get("lot_size") == 1e-20:
        with pytest.raises(ValueError, match="eligible settled episodes"):
            module.fit_settled_allocation_ppo(
                env,
                total_timesteps=2,
                training_protocol=declaration(),
                transition_trace=trace,
            )
        events = records(trace)
        returned = [
            json.loads(bytes.fromhex(row["settled_transition_hex"]))
            for row in events
            if row["kind"] == "env_transition"
        ]
        assert len(returned) == 2 and events[-1]["stage"] == "post_fit_admission"
        assert not returned[-1]["settlement_witness"]["settled"]
        assert any(
            value != "0"
            for value in returned[-1]["rows"][-1]["native"]["book"]["exact_quantities"]
        )
        assert not trace.complete
        return
    policy = module.fit_settled_allocation_ppo(
        env,
        total_timesteps=2,
        training_protocol=declaration(),
        transition_trace=trace,
    )
    assert (
        trace.complete
        and policy.manifest["schema"] == "allocation_ppo_inference_bundle_v6"
    )
    transitions = [
        json.loads(bytes.fromhex(row["settled_transition_hex"]))
        for row in records(trace)
        if row["kind"] == "env_transition"
    ]
    assert transitions[-1]["settlement_witness"]["settled"]
    assert all(
        value == "0"
        for value in transitions[-1]["rows"][-1]["native"]["book"]["exact_quantities"]
    )
    native_rows = [
        row["native"] for transition in transitions for row in transition["rows"]
    ]
    assert any(row["execution"]["fill_count"] for row in native_rows)
    if arguments.get("actor_volume"):
        assert any(
            event["event_type"] == "partial_fill"
            for row in native_rows
            for event in row["order_events"]
        )
    if arguments.get("borrow_rate"):
        assert sum(row["execution"]["interval_borrow_cost"] for row in native_rows) > 0
        assert sum(row["execution"]["interval_funding"] for row in native_rows) > 0
    elif arguments.get("funding"):
        assert sum(row["execution"]["interval_funding"] for row in native_rows) < 0


def test_actual_postlearn_weight_change_cannot_publish(monkeypatch):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module)
    torch = pytest.importorskip("torch")
    original = module.build_settled_allocation_ppo

    def changed(*args, **kwargs):
        model = original(*args, **kwargs)
        learn = model.learn

        def mutate_after_actual_fit(*fit_args, **fit_kwargs):
            returned = learn(*fit_args, **fit_kwargs)
            with torch.no_grad():
                model.policy.value_net.bias.add_(1.0)
            return returned

        model.learn = mutate_after_actual_fit
        return model

    monkeypatch.setattr(module, "build_settled_allocation_ppo", changed)
    env = environment(fee=0.0)
    trace = owner(
        "trade_rl.evaluation.rl_allocation.settled_transition_trace"
    ).SettledTransitionRecorder(env)
    with pytest.raises(ValueError, match="model|parameter"):
        module.fit_settled_allocation_ppo(
            env,
            total_timesteps=2,
            training_protocol=declaration(),
            transition_trace=trace,
        )
    assert not trace.complete
    assert records(trace)[-1]["kind"] == "failure"


@pytest.mark.parametrize("failure", ["callback_stop", "add_after_write", "autoreset"])
def test_actual_collector_failure_retains_native_stage_without_successful_append(
    monkeypatch, failure
):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module)
    env = environment(actor_steps=1, fee=0.0)
    trace = owner(
        "trade_rl.evaluation.rl_allocation.settled_transition_trace"
    ).SettledTransitionRecorder(env)
    if failure == "callback_stop":
        original = trace.callback

        def stopped(base):
            callback = original(base)
            step = callback._on_step

            def stop_after_stage():
                step()
                return False

            callback._on_step = stop_after_stage
            return callback

        monkeypatch.setattr(trace, "callback", stopped)
    elif failure == "add_after_write":
        original = module.build_settled_allocation_ppo

        def failing_add(*args, **kwargs):
            model = original(*args, **kwargs)
            add = model.rollout_buffer.add

            def written_then_failed(*add_args, **add_kwargs):
                add(*add_args, **add_kwargs)
                raise RuntimeError("synthetic append return failure")

            model.rollout_buffer.add = written_then_failed
            return model

        monkeypatch.setattr(module, "build_settled_allocation_ppo", failing_add)
    else:
        reset = env.reset
        calls = 0

        def failed_autoreset(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic autoreset failure")
            return reset(*args, **kwargs)

        monkeypatch.setattr(env, "reset", failed_autoreset)
    with pytest.raises((RuntimeError, ValueError)):
        module.fit_settled_allocation_ppo(
            env,
            total_timesteps=2,
            training_protocol=declaration(),
            transition_trace=trace,
        )
    events = records(trace)
    native = [row for row in events if row["kind"] == "env_transition"]
    assert len(native) == 1 and native[0]["terminated"]
    facts = json.loads(bytes.fromhex(native[0]["settled_transition_hex"]))
    assert len(facts["rows"]) == 3 and facts["rows"][-1]["processing_index"] == 9
    assert facts["settlement_witness"]["settled"]
    assert not any(row["kind"] == "append" for row in events)
    assert not trace.complete
    diagnostic = events[-1]
    assert (
        diagnostic["kind"] == "failure" and diagnostic["successful_append_count"] == 0
    )
    assert diagnostic["actual_model_timesteps"] == (0 if failure == "autoreset" else 1)
    assert diagnostic["buffer_pos"] == (1 if failure == "add_after_write" else 0)


@pytest.mark.parametrize("actor,reserve,budget", [(1, 2, 2), (2, 2, 2), (3, 1, 4)])
def test_actual_cpu_buffer_masks_signed_native_horizon_and_updates(
    monkeypatch, actor, reserve, budget
):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module)
    env = environment(actor_steps=actor, reserve_steps=reserve, fee=0.0)
    trace = owner(
        "trade_rl.evaluation.rl_allocation.settled_transition_trace"
    ).SettledTransitionRecorder(env)
    policy = module.fit_settled_allocation_ppo(
        env,
        total_timesteps=budget,
        seed=0,
        training_protocol=declaration(),
        transition_trace=trace,
    )
    events = records(trace)
    appends = [row for row in events if row["kind"] == "append"]
    transitions = [row for row in events if row["kind"] == "env_transition"]
    rollouts = [row for row in events if row["kind"] == "rollout"]
    critics = [row for row in events if row["kind"] == "critic"]
    updates = [row for row in events if row["kind"] == "update"]
    assert len(appends) == len(transitions) == budget
    assert len(rollouts) == len(critics) == len(updates) == budget // 2
    assert [row["sequence"] for row in appends] == list(range(budget))
    assert all(row["pos_after"] == row["pos_before"] + 1 for row in appends)
    assert all(row["optimizer_calls"] == row["epochs"] == 1 for row in updates)
    assert all(np.array_equal(tensor(row["result"]), [[7.0]]) for row in critics)
    first = rollouts[0]["chronological_buffer"]
    assert np.array_equal(tensor(first["values"]), [[7.0], [7.0]])
    if actor < 3:
        assert critics[0]["role"] == "prepared_reset_boundary_critic"
        if actor == 1:
            assert np.array_equal(tensor(first["advantages"]), [[-7.0], [-7.0]])
            assert not tensor(first["returns"]).any()
        else:
            # lambda=.95 applies once between the two actor transitions.
            assert tensor(first["advantages"]) == pytest.approx(
                np.array([[-6.65], [-7.0]])
            )
            assert tensor(first["returns"]) == pytest.approx(
                np.array([[0.35], [0.0]]), abs=1e-6
            )
    else:
        assert critics[0]["role"] == "live_boundary_critic"
        assert not tensor(first["advantages"]).any()
        assert np.array_equal(tensor(first["returns"]), [[7.0], [7.0]])
        second = rollouts[1]["chronological_buffer"]
        assert np.array_equal(tensor(second["episode_starts"]), [[0.0], [1.0]])
        assert np.array_equal(tensor(second["advantages"]), [[-7.0], [0.0]])
        assert np.array_equal(tensor(second["returns"]), [[0.0], [7.0]])
        assert env.actor_step == 1 and not env._terminated
    payloads = [
        json.loads(bytes.fromhex(row["settled_transition_hex"])) for row in transitions
    ]
    assert [row["actor_index"] for row in payloads] == [
        6 + i % actor for i in range(budget)
    ]
    assert all(row["actor_step"] == i % actor for i, row in enumerate(payloads))
    for row in payloads:
        expected = 1 + reserve if row["terminated"] else 1
        assert len(row["rows"]) == expected
        assert row["reward"] == 0.0
    assert trace.complete and trace.training_manifest == policy.manifest
    manifest = policy.manifest
    assert manifest["schema"] == "allocation_ppo_inference_bundle_v6"
    assert manifest["recipe"]["schema"] == "allocation_ppo_recipe_v4"
    assert manifest["training"]["actual_timesteps"] == budget
    assert manifest["training"]["clock"]["physical_horizon_steps"] == actor + reserve
    assert (
        manifest["training"]["optimization"]["successful_optimizer_step_calls"]
        == budget // 2
    )


def test_incomplete_fit_keeps_live_frontier_and_never_publishes(monkeypatch):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module)
    env = environment(actor_steps=3, reserve_steps=1, fee=0.0)
    trace = owner(
        "trade_rl.evaluation.rl_allocation.settled_transition_trace"
    ).SettledTransitionRecorder(env)
    with pytest.raises(ValueError, match="eligible|completed|settled"):
        module.fit_settled_allocation_ppo(
            env,
            total_timesteps=2,
            training_protocol=declaration(),
            transition_trace=trace,
        )
    assert env.actor_step == 2 and env.index == 8 and not env._terminated
    assert not trace.complete
    with pytest.raises(ValueError, match="publication"):
        _ = trace.training_manifest
    assert len([r for r in records(trace) if r["kind"] == "append"]) == 2


@pytest.mark.parametrize("budget", [True, 0, 1, -2])
def test_invalid_budget_rejects_before_backend_or_first_reset(monkeypatch, budget):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    env = environment()
    monkeypatch.setattr(
        env, "reset", lambda **kwargs: pytest.fail("invalid declaration reset")
    )
    monkeypatch.setattr(
        module,
        "construct_protocol_ppo",
        lambda *args, **kwargs: pytest.fail("invalid declaration imported backend"),
    )
    with pytest.raises(ValueError):
        module.fit_settled_allocation_ppo(
            env, total_timesteps=budget, training_protocol=declaration()
        )


def test_actual_verified_bundle_and_same_h_inference(monkeypatch, tmp_path):
    module = owner("trade_rl.evaluation.rl_allocation.settled_training")
    install_seven(monkeypatch, module)
    env = environment(actor_steps=1, reserve_steps=2, fee=0.0)
    policy = module.fit_settled_allocation_ppo(
        env, total_timesteps=2, training_protocol=declaration()
    )
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    root = tmp_path / "policy"
    pin = save_allocation_policy(root, policy)
    assert {p.name for p in root.iterdir()} == {"manifest.json", "policy.zip"}
    inference = owner("trade_rl.evaluation.rl_allocation.settled_inference")
    fresh = environment(actor_steps=1, reserve_steps=2, fee=0.0)
    loaded = inference.load_settled_allocation_policy(
        root, env=fresh, expected_digest=pin
    )
    assert loaded.manifest["recipe_digest"] == policy.manifest["recipe_digest"]
    result = inference.run_settled_allocation_policy(fresh, loaded)
    assert result["settlement_witness"]["settled"]
    assert (
        fresh.index == 9
        and fresh.actor_step == 1
        and fresh.book.portfolio_value == 1000.0
    )
    loaded.model.num_timesteps += 2
    untouched = environment(actor_steps=1, reserve_steps=2, fee=0.0)
    monkeypatch.setattr(
        untouched, "reset", lambda **kw: pytest.fail("invalid model reset")
    )
    with pytest.raises(ValueError, match="runtime|receipt"):
        inference.run_settled_allocation_policy(untouched, loaded)
    assert untouched._episode_sequence == 0
    different = environment(actor_steps=2, reserve_steps=1, fee=0.0)
    monkeypatch.setattr(
        different, "reset", lambda **kw: pytest.fail("different clock reset")
    )
    from trade_rl.strategies.rl import allocation_artifact

    monkeypatch.setattr(
        allocation_artifact,
        "_load_policy",
        lambda *a: pytest.fail("different clock deserialized"),
    )
    with pytest.raises(ValueError, match="recipe|settled|clock"):
        inference.load_settled_allocation_policy(
            root, env=different, expected_digest=pin
        )
