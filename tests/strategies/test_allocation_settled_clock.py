"""Literal timing oracles, without execution or learner imports."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from importlib import import_module
from types import MappingProxyType

import pytest

MODULE = "trade_rl.strategies.rl.allocation_settled_clock"
DECLARATION = dict(
    actor_steps=2,
    reserve_steps=2,
    execution_interval_seconds=3600,
    rollout_steps=2,
    gamma=1.0,
    gae_lambda=0.95,
)
EXPECTED = {
    "schema": "allocation_settled_clock_v1",
    **DECLARATION,
    "physical_horizon_steps": 4,
    "policy_horizon_seconds": 7200,
    "economic_horizon_seconds": 14400,
    "final_reward_interval_seconds": 10800,
    "reward_schema": "equity_delta_v1",
    "reward_timing": "last_actor_includes_all_native_reserved_bars_v1",
    "gae_clock": "actor_transition",
    "reserve_execution": "zero_target_through_unchanged_hard_risk",
    "reserve_coverage": "all_declared_bars_even_if_flat",
}


def declaration(**changes):
    return import_module(MODULE).SettledAllocationClockContract(
        **{**DECLARATION, **changes}
    )


def test_two_actor_transitions_cover_four_physical_bars_and_full_h_remaining():
    clock = declaration()
    assert clock.physical_horizon_steps == 4
    assert clock.processing_span(0) == (0, 1)
    assert clock.processing_span(1) == (1, 4)
    assert clock.transition_duration_seconds(0) == 3600
    assert clock.transition_duration_seconds(1) == 10800
    assert clock.remaining_horizon_fraction(0) == 1.0
    assert clock.remaining_horizon_fraction(1) == 0.75
    assert clock.payload() == EXPECTED


def test_one_actor_can_terminate_inside_a_two_transition_rollout():
    clock = declaration(actor_steps=1, reserve_steps=2)
    assert clock.processing_span(0) == (0, 3)
    assert clock.transition_duration_seconds(0) == 10800
    assert clock.remaining_horizon_fraction(0) == 1.0
    assert clock.rollout_steps == 2
    # Actor clocks and rollout boundaries do not have a divisibility relation.
    other = declaration(actor_steps=3, reserve_steps=1)
    assert other.processing_span(2) == (2, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_steps", True),
        ("actor_steps", 0),
        ("actor_steps", 2.0),
        ("reserve_steps", False),
        ("reserve_steps", 0),
        ("reserve_steps", 1.5),
        ("execution_interval_seconds", True),
        ("execution_interval_seconds", 0),
        ("execution_interval_seconds", 3600.0),
        ("rollout_steps", True),
        ("rollout_steps", 1),
        ("rollout_steps", 2.0),
        ("gamma", True),
        ("gamma", 0.99),
        ("gamma", float("nan")),
        ("gae_lambda", False),
        ("gae_lambda", float("inf")),
        ("gae_lambda", -0.01),
        ("gae_lambda", 1.01),
    ],
)
def test_invalid_clock_is_rejected_before_any_runtime(field, value):
    with pytest.raises(ValueError):
        declaration(**{field: value})


def test_actor_indices_cannot_refer_to_reserve_or_terminal_sentinel():
    clock = declaration()
    for index in (-1, 2, 3, True, 0.0):
        for method in (
            clock.processing_span,
            clock.transition_duration_seconds,
            clock.remaining_horizon_fraction,
        ):
            with pytest.raises(ValueError, match="actor"):
                method(index)


def test_closed_payload_digest_and_detached_immutability():
    clock = declaration()
    encoded = json.dumps(EXPECTED, sort_keys=True, separators=(",", ":")).encode()
    assert clock.digest == sha256(encoded).hexdigest()
    cls = type(clock)
    assert cls.from_payload(json.loads(encoded)) == clock
    detached = clock.payload()
    detached["reserve_steps"] = 99
    assert clock.payload() == EXPECTED
    with pytest.raises(FrozenInstanceError):
        clock.reserve_steps = 99
    assert replace(clock, gae_lambda=0.0).gae_lambda == 0.0
    assert replace(clock, gae_lambda=1.0).gae_lambda == 1.0


def test_reader_rejects_relational_tampering_and_noncanonical_json():
    cls = type(declaration())
    mutations = (
        {"schema": "financial_clock_v1"},
        {"physical_horizon_steps": 3},
        {"policy_horizon_seconds": 14400},
        {"economic_horizon_seconds": 7200},
        {"final_reward_interval_seconds": 3600},
        {"reward_schema": "net_log_return_v1"},
        {
            "reward_timing": "last_actor_includes_all_native_reserved_bars_v1 ",
        },
        {"gae_clock": "physical_bar"},
        {"reserve_execution": "perfect_liquidation"},
        {"reserve_coverage": "until_flat"},
        {"actor_steps": True},
        {"gamma": True},
        {"gamma": 1},
        {"unexpected": "ignored"},
    )
    for change in mutations:
        with pytest.raises(ValueError):
            cls.from_payload({**deepcopy(EXPECTED), **change})
    missing = deepcopy(EXPECTED)
    del missing["reserve_coverage"]
    for payload in (missing, MappingProxyType(EXPECTED), None):
        with pytest.raises(ValueError):
            cls.from_payload(payload)


def test_legacy_regular_clock_keeps_its_distinct_schema_and_duration_contract():
    from trade_rl.evaluation.objectives.clock import FinancialClockContract

    legacy = FinancialClockContract(
        3600, 3600, 3600, 14400, 2, 1.0, 0.95, "equity_delta_v1"
    )
    assert legacy.payload()["schema"] == "financial_clock_v1"
    with pytest.raises(ValueError, match="one declared reward interval"):
        replace(legacy, reward_interval_seconds=10800)
    assert declaration().payload()["schema"] != legacy.payload()["schema"]
