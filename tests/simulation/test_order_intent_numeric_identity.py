"""Created order IDs must bind the exact numeric representation they persist."""

import json

import pytest

from tests.simulation.test_orders import _intent
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.simulation.orders.model import (
    OrderDomainError,
    OrderIntent,
    OrderType,
)


@pytest.mark.parametrize(
    ("field", "value", "extra"),
    [
        ("requested_quantity", 10, {}),
        ("limit_price", 99, {}),
        ("submission_reference_price", 100, {}),
        ("decision_equity", 1000, {}),
        ("stop_price", 101, {"order_type": OrderType.STOP_MARKET, "limit_price": None}),
    ],
)
def test_integer_input_identity_matches_persisted_float(
    field: str, value: int, extra: dict[str, object]
) -> None:
    intent = _intent(**extra, **{field: value})
    control = _intent(**extra, **{field: float(value)})
    payload = json.loads(canonical_json_bytes(intent.canonical_payload()))
    assert intent == control
    assert intent.order_id == control.order_id
    assert OrderIntent.from_mapping(payload) == intent


@pytest.mark.parametrize(
    ("field", "extra"),
    [
        ("requested_quantity", {}),
        ("limit_price", {}),
        ("submission_reference_price", {}),
        ("decision_equity", {}),
        ("stop_price", {"order_type": OrderType.STOP_MARKET, "limit_price": None}),
    ],
)
def test_create_rejects_boolean_quantity_and_prices(
    field: str, extra: dict[str, object]
) -> None:
    with pytest.raises(OrderDomainError, match=field):
        _intent(**extra, **{field: True})


def test_float_identity_remains_byte_compatible() -> None:
    intent = _intent()
    payload = intent.canonical_payload()
    assert (
        intent.order_id
        == "a76c0a1c140befd5af551c233150a726ff4f4de6eab443a9751314c2f7ef2203"
    )
    assert OrderIntent.from_mapping(json.loads(canonical_json_bytes(payload))) == intent


@pytest.mark.parametrize("value", [2**53 + 1, 10**400], ids=["rounded", "overflow"])
def test_integer_quantity_must_have_exact_finite_float_representation(
    value: int,
) -> None:
    with pytest.raises(OrderDomainError, match="requested_quantity"):
        _intent(requested_quantity=value)


def test_reader_rejects_integer_that_rounds_to_existing_order_identity() -> None:
    intent = _intent(requested_quantity=float(2**53))
    payload = intent.canonical_payload()
    payload["requested_quantity"] = 2**53 + 1
    with pytest.raises(OrderDomainError, match="requested_quantity"):
        OrderIntent.from_mapping(payload)
