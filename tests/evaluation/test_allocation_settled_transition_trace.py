"""Failure cleanup preserves the original error and foreign method ownership."""

import json
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from tests.evaluation.test_allocation_settled_env import environment
from tests.evaluation.test_allocation_settled_training import declaration, install_seven
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation import settled_training
from trade_rl.evaluation.rl_allocation.settled_transition_trace import (
    SettledTransitionRecorder,
    _InstanceLease,
)
from trade_rl.evaluation.rl_allocation.settled_transition_validation import (
    _account,
    _native_sources,
    validate_settled_transition_capture,
)
from trade_rl.evaluation.rl_allocation.transition_facts import _pending
from trade_rl.simulation.orders.model import OrderIntent, PendingOrder


@pytest.fixture(scope="module")
def observed_fit():
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    protocol = replace(declaration(), n_epochs=2)
    with pytest.MonkeyPatch.context() as patch:
        install_seven(patch, settled_training, action=3)
        env = environment()
        trace = SettledTransitionRecorder(env)
        policy = settled_training.fit_settled_allocation_ppo(
            env, total_timesteps=4, training_protocol=protocol, transition_trace=trace
        )
    return env, trace.payload(), policy.manifest["training"]["optimization"], protocol


@pytest.mark.parametrize(
    "mutation", ["initial_bool", "fill_count", "target_identity", "update_epochs"]
)
def test_rehashed_actual_capture_cannot_erase_native_relationships(
    observed_fit, mutation
):
    env, original, optimization, protocol = observed_fit
    capture = deepcopy(original)
    if mutation == "initial_bool":
        next(event for event in capture["events"] if event["kind"] == "training_start")[
            "actual_timesteps"
        ] = False
    elif mutation == "update_epochs":
        updates = [event for event in capture["events"] if event["kind"] == "update"]
        updates[0]["epochs"], updates[0]["n_updates"] = 1, 1
        updates[1]["epochs"], updates[1]["n_updates"] = 3, 4
    else:
        event = next(
            event for event in capture["events"] if event["kind"] == "env_transition"
        )
        facts = json.loads(bytes.fromhex(event["settled_transition_hex"]))
        if mutation == "fill_count":
            row = next(row for row in facts["rows"] if row["kind"] == "actor")
            row["native"]["execution"]["fill_count"] += 1
        else:
            row = facts["rows"][0]
            accepted = PendingOrder.from_mapping(row["accepted_orders"][0])
            original_id = accepted.order_id
            intent = OrderIntent.create(
                **{
                    field.name: getattr(accepted.intent, field.name)
                    for field in fields(OrderIntent)
                    if field.name != "order_id"
                }
                | {"target_identity": "f" * 64}
            )
            row["accepted_orders"][0] = _pending(replace(accepted, intent=intent))
            for evidence in (
                row["native"]["order_events"] + row["native"]["terminal_order_reasons"]
            ):
                if evidence["order_id"] == original_id:
                    evidence["order_id"] = intent.order_id
        event["settled_transition_hex"] = canonical_json_bytes(facts).hex()
    with pytest.raises(ValueError):
        validate_settled_transition_capture(
            env,
            capture,
            requested_timesteps=4,
            optimization=optimization,
            training_protocol=protocol,
        )


@pytest.mark.parametrize("malformed_notes", [False, True])
def test_detach_diagnostic_failure_cannot_replace_supplied_primary(
    monkeypatch, malformed_notes
):
    env = environment()
    trace = SettledTransitionRecorder(env)
    lease = _InstanceLease(
        env, "step", lambda original: lambda action: original(action)
    )

    def foreign(action):
        return None

    env.step = foreign
    trace._leases.append(lease)

    def unavailable_diagnostic():
        raise RuntimeError("diagnostic serialization unavailable")

    monkeypatch.setattr(trace, "payload", unavailable_diagnostic)
    original = ValueError("original execution failure")
    if malformed_notes:
        original.__notes__ = "malformed foreign notes"
    trace.detach(primary_error=original)
    assert env.step is foreign
    if not malformed_notes:
        assert any("ownership" in note for note in original.__notes__)
    assert not trace.complete


def test_returned_account_cannot_hide_its_current_drawdown(observed_fit):
    env, capture, _, _ = observed_fit
    reset = next(event for event in capture["events"] if event["kind"] == "reset")
    account = deepcopy(reset["native_account"])
    account["book"]["peak_value"] = 1200.0
    with pytest.raises(ValueError, match="drawdown"):
        _account(env, account, env.start_index)


def test_native_signed_insolvency_keeps_its_capped_drawdown_diagnostic():
    env = environment(side=-1, terminal_price=400.0)
    env.reset()
    env.step(1)
    _, reward, terminated, _, info = env.step(0)
    facts = json.loads(info["settled_transition"])
    row = facts["rows"][-1]
    book = row["native"]["book"]
    assert terminated and reward < -1 and book["equity"] < 0
    assert book["max_drawdown"] == 1.0 and book["termination_reason"] is not None
    actual = _account(
        env,
        {"book": book, "active_orders": row["native"]["active_orders"]},
        row["processing_index"],
    )
    assert actual.portfolio_value == book["equity"]


def test_absent_funding_cannot_be_offset_by_invented_interest(observed_fit):
    env, capture, _, _ = observed_fit
    before = next(event for event in capture["events"] if event["kind"] == "reset")[
        "native_account"
    ]
    event = next(
        event for event in capture["events"] if event["kind"] == "env_transition"
    )
    row = json.loads(bytes.fromhex(event["settled_transition_hex"]))["rows"][0]
    row["native"]["execution"]["interval_funding"] = 99.0
    row["native"]["execution"]["interval_cash_interest"] = -99.0
    row["native"]["book"]["funding_pnl"] = 99.0
    with pytest.raises(ValueError, match="funding|interest"):
        _native_sources(env, row, before)
