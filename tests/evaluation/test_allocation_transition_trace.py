"""Literal native economics and pre-overwrite observation oracles."""

import json
from dataclasses import fields, replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.test_allocation_preprocessing_runtime import finite_forecast
from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.simulation.test_stateful_execution_characterization import _normalize
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.simulation.accounting import BookState


def literal_args(*, volume=1000.0):
    args = parameters()
    dataset = args["dataset"]
    marks = np.full_like(dataset.close, 128.0)
    marks[7:] = 144.0
    args["dataset"] = replace(
        dataset,
        open=np.full_like(dataset.open, 128.0),
        high=np.maximum(marks, 128),
        low=np.minimum(marks, 128),
        close=marks,
        mark_price=marks,
        index_price=marks,
        volume=np.full_like(dataset.volume, volume),
    )
    args["execution_cost"] = replace(
        args["execution_cost"],
        fee_rate=1 / 512,
        max_participation_rate=1.0,
        processing_bar_volume_capacity=True,
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            args["bound"].objective,
            capital=replace(
                args["bound"].objective.capital, initial_equities=(1024.0,)
            ),
        ),
        clock=replace(args["bound"].clock, rollout_steps=2),
    )
    args["stream"] = finite_forecast(args["dataset"])
    return opt_in(args)


@pytest.mark.parametrize(
    "volume,quantity,cash,equity,cost,remainder",
    [
        (1000.0, "4", 511.0, 1087.0, 1.0, None),
        (0.0, "0", 1024.0, 1024.0, 0.0, 4.0),
        (1.0, "1", 895.75, 1039.75, 0.25, 3.0),
    ],
)
def test_pre_overwrite_capture_binds_literal_request_risk_and_native_cash(
    volume, quantity, cash, equity, cost, remainder
):
    module = import_module("trade_rl.evaluation.rl_allocation.transition_facts")
    env = AllocationTradingEnv(**literal_args(volume=volume))
    env.reset()
    observed = []

    def freeze(result):
        assert env.index == 6 and env.book.cash == 1024.0
        observed.append(module.freeze_allocation_execution(env, result))
        return observed[-1]

    env._transition_recorder = SimpleNamespace(freeze_execution=freeze)
    _, reward, _, _, info = env.step(3)
    facts = json.loads(observed[0])
    assert info["transition_trace"] == observed[0]
    assert (facts["decision_index"], facts["processing_index"]) == (6, 7)
    assert facts["proposal"]["raw_action"] == 3
    assert facts["risk"]["proposal_weights"] == [0.5]
    assert facts["risk"]["weights"] == [0.5]
    assert facts["book"]["exact_quantities"] == [quantity]
    assert facts["book"]["cash"] == cash and facts["book"]["equity"] == equity
    assert facts["execution"]["interval_cost"] == cost
    assert reward == (equity - 1024) / 1024
    assert [o["remaining_quantity"] for o in facts["active_orders"]] == (
        [] if remainder is None else [remainder]
    )


def test_detached_book_reads_never_touch_original_exact_cache(monkeypatch):
    module = import_module("trade_rl.evaluation.rl_allocation.transition_facts")
    env = AllocationTradingEnv(**literal_args())
    env.reset()

    def freeze(result):
        book = result.execution.book
        before = _normalize(book)
        original = BookState.exact_quantities.fget
        clone = BookState.clone

        def checked(observed):
            assert observed is not book, "observer read the original exact cache"
            return original(observed)

        def checked_clone(observed):
            assert observed is not book, "observer cloned the original exact cache"
            return clone(observed)

        with monkeypatch.context() as patch:
            patch.setattr(BookState, "exact_quantities", property(checked))
            patch.setattr(BookState, "clone", checked_clone)
            raw = module.freeze_allocation_execution(env, result)
        assert _normalize(book) == before
        book.cash = -99
        assert json.loads(raw)["book"]["cash"] == 511.0
        book.cash = 511
        return raw

    env._transition_recorder = SimpleNamespace(freeze_execution=freeze)
    env.step(3)


def recorder_module():
    return import_module("trade_rl.evaluation.rl_allocation.transition_trace")


class CallbackBase:
    pass


class DetachedTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def collected_rows(*, volume=1000, args=None, actions=(3, 0), input_recording=False):
    env = AllocationTradingEnv(
        **(literal_args(volume=volume) if args is None else args)
    )
    recorder = recorder_module().AllocationTransitionRecorder(env)
    recorder.attach()
    observation, _ = env.reset()
    callback = recorder.callback(CallbackBase)
    if input_recording:
        from trade_rl.evaluation.rl_allocation.input_receipt import (
            AllocationInputRecorder,
        )

        inputs = AllocationInputRecorder(env)
        input_callback = inputs.callback(CallbackBase, lambda *_: True)
        input_callback.globals = {}
        recorder._test_inputs = inputs
    model = SimpleNamespace(_last_episode_starts=np.array([True]))
    rows = []
    for action in actions:
        old = observation.copy()
        observation, reward, done, _, info = env.step(action)
        if done:
            info.update(
                terminal_observation=observation.copy(),
                **{"TimeLimit.truncated": False},
            )
            observation, _ = env.reset()
        values = dict(
            self=model,
            obs_tensor=DetachedTensor(old[None]),
            actions=np.array([action], np.int64),
            values=DetachedTensor(np.array([[2.0]], np.float32)),
            log_probs=DetachedTensor(np.array([-0.5], np.float32)),
            rewards=np.array([reward], np.float32),
            dones=np.array([done]),
            infos=[info],
            new_obs=observation[None].copy(),
        )
        callback.locals = values
        if input_recording:
            input_callback.locals = values
            assert input_callback._on_step()
        assert callback._on_step()
        rows.append((old.copy(), action, reward, bool(model._last_episode_starts[0])))
        model._last_episode_starts[:] = done
    model.rollout_buffer = SimpleNamespace(
        full=True,
        pos=2,
        generator_ready=False,
        observations=np.array([r[0][None] for r in rows], np.float32),
        actions=np.array([[[r[1]]] for r in rows], np.float32),
        values=np.full((2, 1), 2.0, np.float32),
        log_probs=np.full((2, 1), -0.5, np.float32),
        rewards=np.array([[r[2]] for r in rows], np.float32),
        episode_starts=np.array([[r[3]] for r in rows], np.float32),
    )
    callback.locals = values
    if input_recording:
        input_callback._on_rollout_end()
    return env, recorder, callback, model


def test_actual_tensor_copies_match_buffer_admission_and_do_not_follow_aliases():
    env, recorder, callback, model = collected_rows()
    callback._on_rollout_end()
    rows = [json.loads(raw) for raw in recorder.events]
    assert [r["actor"]["action_code"] for r in rows[:2]] == [3, 0]
    assert rows[0]["actor"]["value"] == np.array([[2]], "<f4").tobytes().hex()
    before = recorder.events
    callback.locals["obs_tensor"].value[:] = -99
    model.rollout_buffer.observations[:] = -99
    assert recorder.events == before and not recorder.complete
    recorder.detach()
    assert env._transition_recorder is None


@pytest.mark.parametrize(
    "change", ["full", "pos", "generator_ready", "actions", "observations", "shape"]
)
def test_callback_count_is_not_admission_and_changed_native_buffer_is_rejected(change):
    _, recorder, callback, model = collected_rows()
    if change == "full":
        model.rollout_buffer.full = False
    elif change == "pos":
        model.rollout_buffer.pos = 1
    elif change == "generator_ready":
        model.rollout_buffer.generator_ready = True
    elif change == "shape":
        model.rollout_buffer.observations = model.rollout_buffer.observations.reshape(
            -1
        )
    else:
        getattr(model.rollout_buffer, change).flat[0] += 1
    with pytest.raises(ValueError, match="buffer|admi"):
        callback._on_rollout_end()
    assert not recorder.complete
    recorder.detach()


def test_existing_recorder_slot_is_never_overwritten():
    env = AllocationTradingEnv(**literal_args())
    recorder = recorder_module().AllocationTransitionRecorder(env)
    occupant = object()
    env._transition_recorder = occupant
    with pytest.raises(ValueError, match="occupied"):
        recorder.attach()
    assert env._transition_recorder is occupant


def test_trace_requires_explicit_protocol_before_optional_import(monkeypatch):
    from trade_rl.evaluation.rl_allocation import training

    env = AllocationTradingEnv(**literal_args())
    recorder = recorder_module().AllocationTransitionRecorder(env)
    monkeypatch.setattr(
        training.importlib, "import_module", lambda _: pytest.fail("backend imported")
    )
    with pytest.raises(ValueError, match="explicit.*protocol"):
        training.fit_allocation_ppo(env, total_timesteps=2, transition_trace=recorder)
    assert env._transition_recorder is None and not recorder.complete


@pytest.mark.parametrize(
    "change",
    [
        None,
        "action",
        "cash",
        "extra",
        "vector",
        "risk",
        "clock",
        "bool",
        "scalar",
        "integerbool",
        "ratio_negative",
        "ratio_large",
        "requested_negative",
        "filled_negative",
        "capacity_volume",
        "capacity_price",
        "capacity_multiplier",
        "capacity_limit",
        "capacity_consumed",
    ],
)
def test_closed_economic_reader_checks_native_mapping_without_reexecution(change):
    from trade_rl.evaluation.rl_allocation import transition_facts

    env, recorder, callback, _ = collected_rows()
    callback._on_rollout_end()
    facts = json.loads(recorder.events[0])["facts"]
    recorder.detach()
    if change == "action":
        facts["proposal"]["raw_action"] = 2
    elif change == "cash":
        facts["book"]["cash"] += 1
    elif change == "extra":
        facts["execution"]["extra"] = 0
    elif change == "vector":
        facts["risk"]["weights"] = [[0.5]]
    elif change == "risk":
        facts["risk"]["weights"] = [0.4]
    elif change == "clock":
        facts["processing_index"] += 1
    elif change == "bool":
        facts["execution"]["bars_advanced"] = True
    elif change == "scalar":
        facts["execution"]["interval_cost"] = True
    elif change == "integerbool":
        facts["execution"]["completed_fill_count"] = True
    elif change in ("ratio_negative", "ratio_large"):
        facts["execution"]["fill_ratio"] = -1.0 if change == "ratio_negative" else 2.0
    elif change in ("requested_negative", "filled_negative"):
        name = (
            "requested_notional"
            if change == "requested_negative"
            else "filled_notional"
        )
        facts["execution"][name] = -1.0
    elif change is not None and change.startswith("capacity_"):
        field, value = {
            "capacity_volume": ("processing_volume", -1.0),
            "capacity_price": ("capacity_reference_price", 0.0),
            "capacity_multiplier": ("contract_multiplier", 0.0),
            "capacity_limit": ("participation_limit", 2.0),
            "capacity_consumed": ("consumed_capacity_notional", -1.0),
        }[change]
        facts["execution"]["capacity_evidence"][0][field] = value
    if change is None:
        reward = transition_facts.validate_allocation_execution_facts(
            facts, env.recipe, dataset_id=env.dataset.dataset_id, action_code=3
        )
        assert reward == 63 / 1024
    else:
        with pytest.raises(ValueError):
            transition_facts.validate_allocation_execution_facts(
                facts, env.recipe, dataset_id=env.dataset.dataset_id, action_code=3
            )


def test_hold_keeps_partial_quantity_but_cancels_residual_and_records_carry():
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        freeze_allocation_execution,
    )

    args = literal_args(volume=1)
    rates = np.zeros(args["dataset"].n_bars)
    rates[8] = 8760
    args["dataset"] = replace(args["dataset"], cash_rate=rates)
    env = AllocationTradingEnv(**args)
    env.reset()
    rows = []
    env._transition_recorder = SimpleNamespace(
        freeze_execution=lambda result: (
            rows.append(freeze_allocation_execution(env, result)) or rows[-1]
        )
    )
    env.step(3)
    _, reward, _, _, _ = env.step(0)
    facts = json.loads(rows[1])
    assert facts["proposal"]["raw_action"] == 0
    assert facts["book"]["exact_quantities"] == ["1"] and not facts["active_orders"]
    assert facts["execution"]["interval_cost"] == 0
    assert facts["execution"]["interval_cash_interest"] == 895.75
    assert facts["book"]["cash"] == 1791.5 and reward == 895.75 / 1024
    assert any(e["new_status"] == "cancelled" for e in facts["order_events"])


@pytest.mark.parametrize("action,target", [(1, -0.5), (3, 0.5)])
def test_distinct_raw_actions_survive_many_to_one_hard_risk(action, target):
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        freeze_allocation_execution,
    )

    args = literal_args()
    args["risk_config"] = replace(args["risk_config"], max_turnover=0)
    args["allocator"] = replace(args["allocator"], lower_weight=-0.5)
    env = AllocationTradingEnv(**opt_in(args))
    env.reset()
    rows = []
    env._transition_recorder = SimpleNamespace(
        freeze_execution=lambda result: (
            rows.append(freeze_allocation_execution(env, result)) or rows[-1]
        )
    )
    _, reward, _, _, _ = env.step(action)
    facts = json.loads(rows[0])
    assert (
        facts["proposal"]["raw_action"] == action
        and facts["proposal"]["target_weight"] == target
    )
    assert facts["risk"]["weights"] == [0] and facts["book"]["exact_quantities"] == [
        "0"
    ]
    assert facts["execution"]["filled_notional"] == 0 and reward == 0


@pytest.mark.parametrize(
    "change",
    [
        None,
        "zero_notional_progress",
        "event_clock",
        "completed_active",
        "tiny_progress",
        "unknown",
        "foreign",
        "limit",
        "stop_market",
        "event_kind",
        "market_trigger_index",
        "market_triggered_sequence",
    ],
)
def test_active_order_reader_preserves_native_progress_and_rejects_impossible_states(
    change,
):
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        validate_allocation_execution_facts,
    )

    env, recorder, _, _ = collected_rows(
        volume=0 if change == "market_triggered_sequence" else 1
    )
    facts = json.loads(recorder.events[0])["facts"]
    recorder.detach()
    order = facts["active_orders"][0]
    if change == "zero_notional_progress":
        order.update(
            cumulative_filled_quantity=0.0,
            exact_cumulative_filled_quantity="0",
            remaining_quantity=4.0,
            status="triggered",
        )
    elif change == "event_clock":
        order["last_processed_index"] = 6
    elif change == "completed_active":
        order.update(
            cumulative_filled_quantity=4.0,
            exact_cumulative_filled_quantity="4",
            remaining_quantity=0.0,
        )
    elif change == "tiny_progress":
        order.update(
            cumulative_filled_quantity=0.0,
            exact_cumulative_filled_quantity=f"1/{10**325}",
            remaining_quantity=4.0,
        )
    elif change == "unknown":
        order["extra"] = 0
    elif change == "foreign":
        order["intent"]["dataset_id"] = "f" * 64
    elif change in ("limit", "stop_market"):
        from trade_rl.simulation.orders.model import OrderIntent, OrderType

        intent = OrderIntent.from_mapping(order["intent"])
        values = {
            f.name: getattr(intent, f.name)
            for f in fields(intent)
            if f.name != "order_id"
        }
        values.update(
            order_type=OrderType(change),
            limit_price=128.0 if change == "limit" else None,
            stop_price=128.0 if change == "stop_market" else None,
        )
        replacement = OrderIntent.create(**values)
        order["intent"] = json.loads(json.dumps(replacement.canonical_payload()))
        for event in facts["order_events"]:
            if event["order_id"] == intent.order_id:
                event["order_id"] = replacement.order_id
    elif change == "event_kind":
        facts["order_events"][-1]["event_type"] = "submitted"
    elif change == "market_trigger_index":
        order["trigger_index"] = order["last_processed_index"]
    elif change == "market_triggered_sequence":
        order.update(status="triggered", trigger_index=7)
        no_fill = facts["order_events"][-1]
        trigger = dict(no_fill)
        trigger.update(event_type="triggered", new_status="triggered")
        no_fill.update(previous_status="triggered", new_status="triggered", sequence=3)
        facts["order_events"].insert(2, trigger)
    if change is None:
        assert (
            validate_allocation_execution_facts(
                facts, env.recipe, dataset_id=env.dataset.dataset_id, action_code=3
            )
            == 15.75 / 1024
        )
    else:
        with pytest.raises(ValueError):
            validate_allocation_execution_facts(
                facts, env.recipe, dataset_id=env.dataset.dataset_id, action_code=3
            )


@pytest.mark.parametrize("selected,drift", [(0, False), (1, False), (0, True)])
def test_trace_preserves_fixed_selected_coordinate_for_full_two_symbol_book(
    selected, drift
):
    from tests.strategies.test_allocation_protocol_receipt import manifest_v3, protocol
    from trade_rl.evaluation.rl_allocation.training_source import (
        allocation_training_source,
    )
    from trade_rl.evaluation.rl_allocation.transition_validation import (
        validate_transition_events,
    )

    args = literal_args()
    dataset = args["dataset"]
    duplicated = {}
    for field in fields(dataset):
        value = getattr(dataset, field.name)
        if (
            field.init
            and not field.name.startswith("global_")
            and isinstance(value, np.ndarray)
        ):
            if value.ndim >= 2 and value.shape[:2] == (dataset.n_bars, 1):
                duplicated[field.name] = np.repeat(value, 2, axis=1)
            elif value.shape == (1,):
                duplicated[field.name] = np.repeat(value, 2)
    args["dataset"] = replace(
        dataset,
        **duplicated,
        symbols=(dataset.symbols[0], "OTHER"),
        volume_units=dataset.volume_units * 2,
    )
    args["stream"] = finite_forecast(args["dataset"])
    args["symbol_index"] = selected
    args["estimates"] = tuple(
        replace(item, symbol=args["dataset"].symbols[selected])
        for item in args["estimates"]
    )
    env, recorder, callback, _ = collected_rows(
        args=opt_in(args),
        actions=(3, 0) if selected == 1 else (0, 0),
        input_recording=True,
    )
    callback._on_rollout_end()
    recorder.detach()
    events = [json.loads(event) for event in recorder.events]
    if selected == 1:
        assert events[0]["facts"]["risk"]["weights"] == [0, 0.5]
        assert events[0]["facts"]["current_weights"] == [0, 0]
        assert events[1]["facts"]["current_weights"] == [0, 576 / 1087]
    else:
        assert all(row["facts"]["risk"]["weights"] == [0, 0] for row in events[:2])
        assert all(row["facts"]["current_weights"] == [0, 0] for row in events[:2])
    bundle = manifest_v3()
    declaration = protocol(n_steps=2, batch_size=2, n_epochs=2)
    bundle.update(recipe=env.recipe, recipe_digest=env.recipe_digest)
    bundle["training"].update(
        actual_timesteps=2,
        requested_timesteps=2,
        protocol=declaration.payload(),
        protocol_digest=declaration.digest,
        clock=env.bound.clock.payload(),
        objective=env.bound.objective.payload(),
        source=allocation_training_source(
            env, decision_counts={6: 1, 7: 1}, observation_indices=(6, 7)
        ),
        observation_consumption=recorder._test_inputs.payload(),
    )
    ppo = bundle["training"]["ppo"]
    ppo.update(
        {
            key: value
            for key, value in declaration.payload()["ppo"].items()
            if key in ppo
        }
    )
    bundle["training"]["optimization"].update(
        rollout_update_count=1,
        epoch_iterations=2,
        successful_optimizer_step_calls=2,
    )
    from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest

    validate_allocation_manifest(bundle)
    if drift:
        events[1]["facts"]["symbol_index"] = 1
        with pytest.raises(ValueError, match="symbol|coordinate"):
            validate_transition_events(events, bundle)
    else:
        assert validate_transition_events(events, bundle)["symbol_count"] == 2
