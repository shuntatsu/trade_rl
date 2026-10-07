"""Complete native records distinguish first decisions and recorded target bounds."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from importlib import import_module
from math import fsum

import numpy as np
import pytest

from tests.evaluation.test_allocation_scheduled_transition_trace import (
    fake_bundle,
    fake_capture,
    native_update_fixture,
)
from trade_rl.artifacts import content_digest


def capability():
    try:
        return import_module(
            "trade_rl.evaluation.rl_allocation.scheduled_credit_diagnostics"
        ).diagnose_scheduled_allocation_credit
    except ModuleNotFoundError:
        pytest.fail("complete scheduled credit diagnostics are missing")


@pytest.fixture(scope="module")
def cash_trace():
    env, _, events, _ = fake_capture(actions=(2,) * 12)
    return events, fake_bundle(env)


def test_literal_targets_keep_true_episode_phase_and_unequal_groups(cash_trace):
    events, bundle = cash_trace
    report = capability()(events, bundle).payload()
    assert report["transition_count"] == 12
    assert report["rollout_count"] == 2
    assert report["phase_counts"] == {"first_decision": 3, "later_decision": 9}
    assert report["raw_action_counts"] == [0, 0, 12, 0]
    # Independent literal cash rewards, collector values and pinned GAE targets.
    advantages = np.asarray(
        [
            0.0175234375,
            -0.04734375,
            -0.115625,
            -0.1875,
            6.415625,
            6.6875,
            -0.353125,
            -0.4375,
            -0.4111640625,
            -0.49859375,
            -0.590625,
            -0.6875,
        ],
        dtype="<f4",
    ).astype(float)
    values = np.arange(12, dtype=float) / 16
    returns = (advantages.astype("<f4") + values.astype("<f4")).astype(float)
    summary = report["summary"]
    assert summary["count"] == 12
    assert summary["advantage_counts"] == {"negative": 9, "zero": 0, "positive": 3}
    for name, expected in (
        ("raw_advantage", advantages),
        ("raw_return", returns),
        ("critic_value", values),
        ("collector_reward", np.zeros(12)),
    ):
        assert summary[name]["mean"] == pytest.approx(fsum(expected) / 12)
        assert summary[name]["min"] == pytest.approx(min(expected))
        assert summary[name]["max"] == pytest.approx(max(expected))
    first = next(row for row in report["phases"] if row["phase"] == "first_decision")
    assert first["advantage_counts"] == {"negative": 1, "zero": 0, "positive": 2}
    assert first["raw_advantage"]["mean"] == pytest.approx(
        fsum(advantages[[0, 4, 8]]) / 3
    )
    groups = report["groups"]
    assert sum(row["count"] for row in groups) == 12
    assert any(row["count"] == 3 for row in groups)
    assert summary["raw_advantage"]["mean"] != pytest.approx(
        fsum(row["raw_advantage"]["mean"] for row in groups) / len(groups)
    )
    assert report["actual_minibatch_exposure"] == "NOT_ESTABLISHED"
    assert report["gradient_attribution"] == "NOT_ESTABLISHED"


def test_terminal_inside_live_ended_rollout_is_its_own_target_boundary(cash_trace):
    events, bundle = cash_trace
    report = capability()(events, bundle).payload()
    boundaries = [row for row in events if row["kind"] == "rollout"]
    assert boundaries[0]["done"] is False
    assert boundaries[1]["done"] is True
    windows = bundle["training"]["schedule"]["train_window_ids"]
    assert [row["window_id"] for row in report["windows"]] == windows
    assert [row["count"] for row in report["windows"]] == [4, 4, 4]
    pairs = {
        (row["window_id"], row["phase"], row["target_boundary"]): row["count"]
        for row in report["groups"]
    }
    assert pairs == {
        (windows[0], "first_decision", "same_episode_terminal"): 1,
        (windows[0], "later_decision", "same_episode_terminal"): 3,
        (windows[1], "first_decision", "live_bootstrap"): 1,
        (windows[1], "later_decision", "live_bootstrap"): 1,
        (windows[1], "later_decision", "same_episode_terminal"): 2,
        (windows[2], "first_decision", "same_episode_terminal"): 1,
        (windows[2], "later_decision", "same_episode_terminal"): 3,
    }
    assert report["boundary_counts"] == {
        "same_episode_terminal": 10,
        "live_bootstrap": 2,
    }


def test_absent_actions_have_zero_coverage_and_no_invented_means(cash_trace):
    events, bundle = cash_trace
    actions = capability()(events, bundle).payload()["raw_actions"]
    assert [row["raw_action_code"] for row in actions] == [0, 1, 2, 3]
    for code in (0, 1, 3):
        row = actions[code]
        assert row["count"] == 0
        assert row["advantage_counts"] == {"negative": 0, "zero": 0, "positive": 0}
        assert row["raw_advantage"] is None
        assert row["raw_return"] is None
        assert row["critic_value"] is None
        assert row["collector_reward"] is None
        assert row["filled_transition_count"] == 0
        assert row["filled_notional"] is None
        assert row["interval_cost"] is None


def test_hold_raw_action_does_not_mean_flat_post_transition_quantity():
    env, _, events, _ = fake_capture(actions=(3, 0, 0, 0) * 3, zero_values=True)
    report = capability()(events, fake_bundle(env)).payload()
    assert report["raw_action_counts"] == [9, 0, 0, 3]
    assert all(row["post_held_sign"] == 1 for row in report["groups"])
    holds = [row for row in report["groups"] if row["raw_action_code"] == 0]
    assert sum(row["count"] for row in holds) == 9
    assert all(row["phase"] == "later_decision" for row in holds)
    # A held quantity is not evidence of a new fill on this decision.
    rows = [row for row in events if row["kind"] == "transition"]
    assert any(
        row["actor"]["action_code"] == 0
        and row["facts"]["execution"]["filled_notional"] == 0
        for row in rows
    )


def test_all_actions_and_signed_post_groups_remain_distinct_and_sorted():
    env, _, events, _ = fake_capture(actions=(1, 0, 2, 3) * 3)
    report = capability()(events, fake_bundle(env)).payload()
    assert report["raw_action_counts"] == [3, 3, 3, 3]
    groups = report["groups"]
    # This native fixture's long-only allocator clamps negative code 1 to flat.
    assert {row["post_held_sign"] for row in groups} == {0, 1}
    keys = [
        (
            row["window_id"],
            row["phase"],
            row["raw_action_code"],
            row["post_held_sign"],
            row["target_boundary"],
        )
        for row in groups
    ]
    assert keys == sorted(keys)
    assert all(
        row["post_held_sign"] == 0 for row in groups if row["raw_action_code"] in (0, 1)
    )


def test_recorded_fills_and_cost_distinguish_held_positions_from_new_execution():
    env, _, events, _ = fake_capture(actions=(3, 0, 0, 0) * 3)
    report = capability()(events, fake_bundle(env)).payload()
    assert "filled_transition_count" in report["summary"], (
        "native new-fill coverage is missing"
    )
    assert report["summary"]["filled_transition_count"] == 3
    assert report["summary"]["filled_notional"] == {
        "mean": 125.0,
        "min": 0.0,
        "max": 500.0,
    }
    assert report["summary"]["interval_cost"] == {"mean": 0.25, "min": 0.0, "max": 1.0}
    first, later = report["phases"]
    assert first["count"] == 3 and first["filled_transition_count"] == 3
    assert first["filled_notional"]["mean"] == 500.0
    assert later["count"] == 9 and later["filled_transition_count"] == 0
    assert later["interval_cost"]["mean"] == 0.0
    assert sum(row["filled_transition_count"] for row in report["groups"]) == 3


@pytest.mark.parametrize("mutation", ["missing", "reordered", "gae", "source", "mode"])
def test_closed_admission_rejects_inconsistent_evidence_before_summary(
    cash_trace, mutation, monkeypatch
):
    events, bundle = deepcopy(cash_trace)
    module = import_module(
        "trade_rl.evaluation.rl_allocation.scheduled_credit_diagnostics"
    )
    if mutation == "missing":
        events.pop(2)
    elif mutation == "reordered":
        events[2], events[3] = events[3], events[2]
    elif mutation == "gae":
        boundary = next(row for row in events if row["kind"] == "rollout")
        targets = np.frombuffer(bytes.fromhex(boundary["advantages"]), "<f4").copy()
        targets[0] = np.nextafter(targets[0], np.float32(np.inf))
        boundary["advantages"] = targets.tobytes().hex()
    elif mutation == "source":
        events[1]["old"]["source_scope_digest"] = "a" * 64
    kwargs = (
        {"learner_diagnostics": "native_ppo_update_v1"} if mutation == "mode" else {}
    )

    def forbidden_summary(*args, **kwargs):
        raise AssertionError("aggregation ran before closed admission")

    monkeypatch.setattr(module, "_summary", forbidden_summary)
    with pytest.raises(ValueError):
        capability()(events, bundle, **kwargs)


def test_result_is_detached_immutable_and_binds_complete_inputs(cash_trace):
    events, bundle = deepcopy(cash_trace)
    original_inputs = deepcopy((events, bundle))
    result = capability()(events, bundle)
    assert (events, bundle) == original_inputs
    payload = result.payload()
    identities = payload["identities"]
    assert identities["events_digest"] == content_digest(events)
    assert identities["bundle_digest"] == content_digest(bundle)
    assert identities["training_digest"] == content_digest(bundle["training"])
    assert identities["schedule_digest"] == bundle["training"]["schedule_digest"]
    assert identities["source_records_digest"] == content_digest(
        bundle["training"]["sources"]
    )
    assert identities["protocol_digest"] == bundle["training"]["protocol_digest"]
    assert identities["learner_diagnostics"] == "none"
    assert result.digest == content_digest(payload)
    original_digest = result.digest
    payload["summary"]["raw_advantage"]["mean"] = 999
    events.clear()
    bundle.clear()
    assert result.digest == original_digest
    assert result.payload()["summary"]["count"] == 12
    with pytest.raises(FrozenInstanceError):
        result._payload_json = "{}"


def test_native_update_mode_admits_every_transition_without_counting_updates():
    events, bundle = native_update_fixture()
    report = capability()(
        events, bundle, learner_diagnostics="native_ppo_update_v1"
    ).payload()
    assert report["summary"]["count"] == 12
    assert report["rollout_count"] == 2
    assert report["identities"]["learner_diagnostics"] == "native_ppo_update_v1"
    assert report["identities"]["events_digest"] == content_digest(events)
    assert report["actual_minibatch_exposure"] == "NOT_ESTABLISHED"


def test_zero_advantages_are_counted_without_strict_positive_alias():
    env, _, events, _ = fake_capture(actions=(2,) * 12, zero_values=True)
    report = capability()(events, fake_bundle(env)).payload()
    assert report["summary"]["advantage_counts"] == {
        "negative": 0,
        "zero": 12,
        "positive": 0,
    }
    assert report["summary"]["raw_advantage"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
