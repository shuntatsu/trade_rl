"""Native no-fill attempt clocks do not advance immutable order state."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.test_allocation_preprocessing_runtime import finite_forecast
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.evaluation.test_allocation_transition_trace import literal_args
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.transition_facts import (
    freeze_allocation_execution,
    validate_allocation_execution_facts,
)


def native_carried_facts(first_volume):
    args = literal_args(volume=0)
    dataset = args["dataset"]
    prices = np.full_like(dataset.close, 128.0)
    volumes = np.zeros_like(dataset.volume)
    volumes[7] = first_volume
    args["dataset"] = replace(
        dataset,
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        mark_price=prices.copy(),
        index_price=prices.copy(),
        volume=volumes,
    )
    # Zero fee keeps the repeated target exactly unchanged after partial fill;
    # the production fixture and native order/account implementation are intact.
    args["execution_cost"] = replace(args["execution_cost"], fee_rate=0)
    args["stream"] = finite_forecast(args["dataset"])
    env = AllocationTradingEnv(**opt_in(args))
    env.reset()
    rows = []
    env._transition_recorder = SimpleNamespace(
        freeze_execution=lambda result: (
            rows.append(freeze_allocation_execution(env, result)) or rows[-1]
        )
    )
    rewards = []
    for _ in range(2):
        _, reward, _, _, _ = env.step(3)
        rewards.append(reward)
    return env, [json.loads(raw) for raw in rows], rewards


def validate(env, facts):
    # Canonical detach/rehash prevents mutation checks relying on stale bytes.
    return validate_allocation_execution_facts(
        json.loads(canonical_json_bytes(facts)),
        env.recipe,
        dataset_id=env.dataset.dataset_id,
        action_code=3,
    )


@pytest.mark.parametrize(
    "first_volume,status,quantity", [(0, "eligible", "0"), (1, "partially_filled", "1")]
)
def test_native_carried_no_fill_preserves_mutation_clock_and_whole_order(
    first_volume, status, quantity
):
    env, rows, rewards = native_carried_facts(first_volume)
    initial, carried = rows
    assert initial["active_orders"] == carried["active_orders"]
    order = carried["active_orders"][0]
    assert order["status"] == status and order["last_processed_index"] == 7
    assert carried["book"]["exact_quantities"] == [quantity]
    assert (carried["decision_index"], carried["processing_index"]) == (7, 8)
    assert [event["event_type"] for event in carried["order_events"]] == ["no_fill"]
    assert carried["order_events"][0]["processing_index"] == 8
    assert validate(env, initial) == rewards[0] == 0
    assert validate(env, carried) == rewards[1] == 0


@pytest.mark.parametrize("first_volume", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["current_attempt_clock", "missing_clock", "status", "remaining"]
)
def test_rehashed_carried_no_fill_rejects_impossible_final_state(
    first_volume, mutation
):
    env, rows, _ = native_carried_facts(first_volume)
    facts = rows[1]
    order, event = facts["active_orders"][0], facts["order_events"][0]
    if mutation == "current_attempt_clock":
        order["last_processed_index"] = facts["processing_index"]
    elif mutation == "missing_clock":
        order["last_processed_index"] = None
    elif mutation == "status":
        event.update(previous_status="latency_wait", new_status="latency_wait")
    else:
        event["remaining_quantity"] -= 0.5
    with pytest.raises(ValueError):
        validate(env, facts)


@pytest.mark.parametrize("first_volume", [0, 1])
def test_same_bar_state_change_then_no_fill_requires_actual_mutation_clock(
    first_volume,
):
    env, rows, _ = native_carried_facts(first_volume)
    facts = rows[0]
    if first_volume:
        # Append a same-bar no-fill attempt to the genuine partial-fill state.
        event = dict(facts["order_events"][-1])
        event.update(
            event_type="no_fill",
            previous_status="partially_filled",
            filled_quantity=0.0,
            filled_notional=0.0,
            sequence=len(facts["order_events"]),
        )
        facts["order_events"].append(event)
    assert facts["order_events"][-1]["event_type"] == "no_fill"
    facts["active_orders"][0]["last_processed_index"] = facts["decision_index"]
    with pytest.raises(ValueError, match="last native event"):
        validate(env, facts)


def test_submitted_plus_no_fill_cannot_skip_native_eligible_transition():
    env, rows, _ = native_carried_facts(0)
    facts = rows[0]
    facts["order_events"] = [
        event for event in facts["order_events"] if event["event_type"] != "eligible"
    ]
    for sequence, event in enumerate(facts["order_events"]):
        event["sequence"] = sequence
    assert [event["event_type"] for event in facts["order_events"]] == [
        "submitted",
        "no_fill",
    ]
    with pytest.raises(ValueError):
        validate(env, facts)
