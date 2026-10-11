"""Literal closed-witness algebra and admission, never execution authenticity."""

from copy import deepcopy
from hashlib import sha256
from importlib import import_module, util

import pytest

from tests.evaluation.test_allocation_settled_env import environment
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import canonical_json_bytes


def receipt_type():
    name = "trade_rl.strategies.rl.allocation_settled_receipt"
    assert util.find_spec(name), "distinct lower settled receipt is missing"
    return import_module(name).SettledAllocationTrainingReceipt


def literal(*, actor=2, reserve=2, budget=2):
    env = environment(actor_steps=actor, reserve_steps=reserve)
    declaration = protocol(n_steps=2, batch_size=2, n_epochs=1)
    horizon = actor + reserve
    episode = {
        "schema": "allocation_settled_episode_witness_v1",
        "episode_index": 0,
        "actor_transition_indices": list(range(actor)),
        "processing_rows": [
            {
                "kind": "actor" if i < actor else "reserve",
                "start_index": 6 + i,
                "processing_index": 7 + i,
                "bars_advanced": 1,
                "returned_clock_types": {"next_index": "int", "bars_advanced": "int"},
            }
            for i in range(horizon)
        ],
        "final_book": {
            "as_of_index": 6 + horizon,
            "as_of_dataset_id": env.dataset.dataset_id,
            "cash": 998.0,
            "equity": 998.0,
            "peak_value": 1000.0,
            "exact_quantities": ["0"],
            "max_drawdown": 0.0020000000000000018,
            "insolvent": False,
            "termination_reason": None,
        },
        "final_active_orders": [],
        "failure": None,
    }
    source = dict(
        dataset_id=env.dataset.dataset_id,
        symbol=env.dataset.symbols[0],
        symbol_count=1,
        symbol_index=0,
        start_index=6,
        stop_index=6 + horizon,
        start_time_ns=int(
            env.dataset.timestamps[6].astype("datetime64[ns]").astype("int64")
        ),
        terminal_time_ns=int(
            env.dataset.timestamps[6 + horizon].astype("datetime64[ns]").astype("int64")
        ),
    )
    events = b""
    for i in range(budget // 2):
        event = canonical_json_bytes(
            dict(step=i + 1, rollout=i + 1, timesteps=(i + 1) * 2)
        )
        events += len(event).to_bytes(8, "big") + event
    raw = dict(
        schema="allocation_settled_training_receipt_v1",
        seed=0,
        requested_timesteps=budget,
        actual_timesteps=budget,
        ppo=dict(
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=0.95,
            learning_rate=declaration.learning_rate,
            net_arch=declaration.payload()["policy"]["net_arch"],
            device="cpu",
        ),
        protocol=declaration.payload(),
        protocol_digest=declaration.digest,
        clock=env.bound.clock.payload(),
        objective=env.bound.objective.payload(),
        source=source,
        episodes=[episode],
        bindings=dict(
            schema="allocation_settled_upper_proof_bindings_v1",
            source_consumption_digest="a" * 64,
            observation_consumption_digest="b" * 64,
            transition_trace_digest="c" * 64,
            buffer_digest="d" * 64,
            append_ledger_digest="e" * 64,
        ),
        optimization=dict(
            schema="allocation_ppo_optimization_receipt_v1",
            rollout_update_count=budget // 2,
            successful_optimizer_step_calls=budget // 2,
            epoch_iterations=budget // 2,
            optimizer_step_events_digest=sha256(events).hexdigest(),
            final_capture="after_final_train_v1",
        ),
        failure=None,
    )
    return env, raw


def read(env, raw):
    return receipt_type().from_payload(
        raw, recipe=env.recipe, recipe_digest=env.recipe_digest
    )


def test_literal_native_witness_has_derived_counts_and_is_detached():
    env, raw = literal()
    receipt = read(env, raw)
    assert (
        receipt.completed_episode_count
        == receipt.structurally_settled_episode_count
        == receipt.eligible_episode_count
        == 1
    )
    assert receipt.live_episode_count == 0
    receipt.require_admission()
    before = receipt.payload()
    raw["episodes"][0]["final_book"]["cash"] = -1.0
    assert receipt.payload() == before
    with pytest.raises((AttributeError, TypeError)):
        receipt.eligible_episode_count = 9


@pytest.mark.parametrize(
    "change", ["missing_reserve", "residual", "economic_stop", "dd", "failure"]
)
def test_diagnostic_economics_cannot_become_successful_bundle(change):
    env, raw = literal()
    episode = raw["episodes"][0]
    if change == "missing_reserve":
        episode["processing_rows"].pop()
        episode["final_book"]["as_of_index"] -= 1
    elif change == "residual":
        episode["final_book"]["exact_quantities"] = ["3"]
    elif change == "economic_stop":
        episode["final_book"].update(
            cash=-504.5,
            equity=-504.5,
            max_drawdown=1.5045,
            insolvent=True,
            termination_reason="insolvency",
        )
    elif change == "dd":
        episode["final_book"]["max_drawdown"] = 0.21
    else:
        episode["failure"] = dict(
            stage="reserve_postflight",
            attempted_start_index=9,
            error_type="ValueError",
            error_message="returned-clock mismatch",
        )
    receipt = read(env, raw)
    assert receipt.payload() == raw
    with pytest.raises(ValueError):
        receipt.require_admission()
    if change == "dd":
        assert (
            receipt.structurally_settled_episode_count == 1
            and receipt.eligible_episode_count == 0
        )


@pytest.mark.parametrize(
    "change",
    [
        "summary",
        "bool_clock",
        "extra",
        "source_span",
        "false_native_type",
        "noncanonical_quantity",
    ],
)
def test_closed_reader_rejects_rehashed_relational_or_type_tamper(change):
    env, raw = literal()
    if change == "summary":
        raw["episodes"][0]["settled"] = True
    elif change == "bool_clock":
        raw["episodes"][0]["processing_rows"][0]["bars_advanced"] = True
    elif change == "extra":
        raw["bindings"]["unknown"] = "f" * 64
    elif change == "source_span":
        raw["source"]["stop_index"] -= 1
    elif change == "false_native_type":
        raw["episodes"][0]["processing_rows"][0]["returned_clock_types"][
            "next_index"
        ] = "numpy.int64"
    else:
        raw["episodes"][0]["final_book"]["exact_quantities"] = ["0/1"]
    with pytest.raises(ValueError):
        read(env, raw)


def test_live_budget_tail_is_preserved_without_actor_divisibility():
    env, raw = literal(actor=3, reserve=1, budget=4)
    tail = deepcopy(raw["episodes"][0])
    tail["episode_index"] = 1
    tail["actor_transition_indices"] = [0]
    tail["processing_rows"] = tail["processing_rows"][:1]
    tail["final_book"]["as_of_index"] = 7
    raw["episodes"].append(tail)
    receipt = read(env, raw)
    assert receipt.completed_episode_count == receipt.eligible_episode_count == 1
    assert receipt.live_episode_count == 1
    receipt.require_admission()
