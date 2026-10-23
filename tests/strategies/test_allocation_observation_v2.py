from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema


def schema(**changes):
    values = dict(
        feature_names=("alpha", "beta"),
        max_active_orders=2,
        initial_capital=100.0,
        episode_steps=16,
    )
    return AllocationObservationSchema(**(values | changes))


def test_declared_field_order_and_namespaces():
    # Independent layout specification, without encoder/runtime constants.
    forecast = "expected_simple_return return_variance buy_cost sell_cost exit_cost funding_return borrow_return cash_return baseline_target_weight"
    account = (
        "position_notional_over_initial_capital current_weight cash_over_initial_capital "
        "equity_over_initial_capital peak_over_initial_capital current_drawdown maximum_drawdown "
        "margin_used_over_initial_capital maintenance_margin maintenance_requirement_over_initial_capital "
        "margin_deficit_over_initial_capital observable_tradable remaining_horizon_fraction"
    )
    order = (
        "presence requested_notional_over_initial_capital cumulative_quantity_notional_over_initial_capital "
        "remaining_notional_over_initial_capital cumulative_filled_notional_over_initial_capital "
        "submission_reference_over_current_mark decision_equity_over_initial_capital submit_offset eligible_offset "
        "expiry_presence expiry_offset trigger_presence trigger_offset last_transition_presence last_transition_offset "
        "reduce_only tif_ioc tif_day tif_gtc status_submitted status_latency_wait status_eligible status_triggered status_partially_filled"
    )
    expected = (
        "feature:alpha",
        "feature:beta",
        *(forecast + " " + account).split(),
        *(f"order[{slot}].{field}" for slot in (0, 1) for field in order.split()),
    )
    assert schema().fields == expected
    names = schema(feature_names=("cash_return", "order[0].presence")).fields
    assert len(set(names)) == len(names)


def test_frozen_schema_detached_payload_and_identity():
    contract = schema()
    payload = contract.payload()
    original_digest = contract.digest
    assert payload["schema"] == "allocation_account_observation_v2"
    assert payload["initial_capital"] == 100.0 and payload["episode_steps"] == 16
    assert payload["fields"] == list(contract.fields)
    assert content_digest(payload) == original_digest
    payload["fields"].clear()
    payload["feature_names"].clear()
    payload["initial_capital"] = 0
    assert contract.digest == original_digest and contract.feature_names == (
        "alpha",
        "beta",
    )
    with pytest.raises(FrozenInstanceError):
        contract.max_active_orders = 3
    alternatives = (
        {},
        {"max_active_orders": 1},
        {"initial_capital": 200},
        {"episode_steps": 32},
        {"feature_names": ("beta", "alpha")},
    )
    assert len({schema(**change).digest for change in alternatives}) == 5


@pytest.mark.parametrize(
    "change",
    [
        {"feature_names": []},
        {"feature_names": ()},
        {"feature_names": "alpha"},
        {"feature_names": ("x", "x")},
        {"feature_names": ("",)},
        {"feature_names": (True,)},
        {"max_active_orders": True},
        {"max_active_orders": 0},
        {"max_active_orders": 65},
        {"max_active_orders": 1.5},
        {"initial_capital": True},
        {"initial_capital": 0},
        {"initial_capital": -1},
        {"initial_capital": np.inf},
        {"initial_capital": np.nan},
        {"episode_steps": True},
        {"episode_steps": 0},
        {"episode_steps": 1.5},
    ],
)
def test_schema_rejects_ambiguous_contracts(change):
    with pytest.raises(ValueError):
        schema(**change)


@pytest.mark.parametrize(
    "name", ["feature_names", "max_active_orders", "initial_capital", "episode_steps"]
)
def test_all_four_schema_identity_fields_are_required(name):
    values = dict(
        feature_names=("alpha", "beta"),
        max_active_orders=2,
        initial_capital=100,
        episode_steps=16,
    )
    del values[name]
    with pytest.raises(TypeError):
        AllocationObservationSchema(**values)


@pytest.mark.parametrize("capital", [np.float32(1 / 3), Fraction(1, 3), 100])
def test_native_capital_is_canonical_float(capital):
    contract = schema(initial_capital=capital)
    assert type(contract.initial_capital) is float
    assert contract.initial_capital == float(capital)
    assert contract.digest == schema(initial_capital=float(capital)).digest


@pytest.mark.parametrize("capital", [10**1000, Fraction(1, 10**1000)])
def test_unreportable_native_capital_is_a_declared_value_error(capital):
    with pytest.raises(ValueError):
        schema(initial_capital=capital)


@pytest.mark.parametrize("slots", [1, 64])
def test_order_capacity_boundary_declares_complete_fixed_width(slots):
    contract = schema(max_active_orders=slots)
    assert len(contract.fields) == 2 + 9 + 13 + 24 * slots
    assert contract.fields[-1] == f"order[{slots - 1}].status_partially_filled"
