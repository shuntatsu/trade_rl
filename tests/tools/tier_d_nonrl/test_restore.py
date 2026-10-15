"""Frozen declaration restoration has no authority to refit or re-declare."""

import copy
import importlib

import pytest

from tests.tools.tier_d_nonrl.plan_fixtures import costs_payload, matrix_payload
from trade_rl.artifacts import content_digest


def api():
    return importlib.import_module("tools.tier_d_nonrl.restore")


def test_exact_public_plan_roundtrip_is_detached():
    raw = matrix_payload()  # The existing public constructors validate the fixture.
    expected = copy.deepcopy(raw)
    plan = api().restore_matrix(raw, expected_digest=content_digest(raw))
    assert plan.payload == expected
    assert plan.digest == content_digest(expected)
    raw["accounts"][0] = "mutated"
    assert plan.payload == expected


@pytest.mark.parametrize(
    "change", ["bound", "redundant_recipe", "objective_schema", "clock", "extra"]
)
def test_rehashed_invalid_plan_payload_is_not_accepted(change):
    raw = matrix_payload()
    terminal = raw["groups"][0]["plans"]["nonrl"]
    if change == "bound":
        terminal["bound_digest"] = "b" * 64
    elif change == "redundant_recipe":
        terminal["recipe"]["target"] = "substituted"
    elif change == "objective_schema":
        terminal["objective"]["tax_treatment"] = "changed"
    elif change == "clock":
        terminal["clock"]["decision_interval_seconds"] = True
    else:
        terminal["unexpected"] = True
    with pytest.raises(ValueError):
        api().restore_matrix(raw, expected_digest=content_digest(raw))


def test_cost_rows_roundtrip_preserves_exact_rates_and_clocks():
    raw = costs_payload()
    table = api().restore_costs(raw, expected_digest=content_digest(raw))
    assert table[24, "BASEL0"][0].payload() == raw["tables"][0]["rows"][0]


@pytest.mark.parametrize(
    "field,value", [("available_at", True), ("buy_cost", -0.1), ("schema", "unknown")]
)
def test_rehashed_invalid_costs_are_rejected(field, value):
    raw = costs_payload()
    raw["tables"][0]["rows"][0][field] = value
    with pytest.raises(ValueError):
        api().restore_costs(raw, expected_digest=content_digest(raw))
