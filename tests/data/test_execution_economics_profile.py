from __future__ import annotations

import math

import pytest

import trade_rl.data.build as market_build


def _profile_type():
    profile_type = getattr(market_build, "ExecutionEconomicsProfile", None)
    assert profile_type is not None, "data.build must own ExecutionEconomicsProfile"
    return profile_type


def test_execution_economics_profile_round_trips_exact_payload() -> None:
    profile_type = _profile_type()
    profile = profile_type(
        name="research_v1",
        fee_rate=0.0005,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )

    payload = profile.to_payload()

    assert payload == {
        "schema_version": "execution_economics_profile_v1",
        "name": "research_v1",
        "fee_rate": 0.0005,
        "maker_fee_rate": 0.0,
        "taker_fee_rate": 0.0,
        "spread_rate": 0.0002,
        "max_participation_rate": 0.05,
        "borrow_available": True,
        "borrow_rate": 0.0,
    }
    assert profile_type.from_payload(payload, field="execution_economics") == profile


def test_execution_economics_profile_rejects_unknown_payload_field() -> None:
    profile_type = _profile_type()
    payload = {
        "schema_version": "execution_economics_profile_v1",
        "name": "research_v1",
        "fee_rate": 0.0005,
        "maker_fee_rate": 0.0,
        "taker_fee_rate": 0.0,
        "spread_rate": 0.0002,
        "max_participation_rate": 0.05,
        "borrow_available": True,
        "borrow_rate": 0.0,
        "unexpected": 1,
    }

    with pytest.raises(ValueError, match="execution_economics contains unknown fields"):
        profile_type.from_payload(payload, field="execution_economics")


@pytest.mark.parametrize("field", ["maker_fee_rate", "taker_fee_rate"])
def test_execution_economics_profile_rejects_fee_double_count(field: str) -> None:
    profile_type = _profile_type()
    kwargs = {
        "name": "invalid",
        "fee_rate": 0.0005,
        "maker_fee_rate": 0.0,
        "taker_fee_rate": 0.0,
    }
    kwargs[field] = 0.0001

    with pytest.raises(ValueError, match="generic fee_rate cannot be combined"):
        profile_type(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fee_rate", -0.1),
        ("maker_fee_rate", -0.1),
        ("taker_fee_rate", -0.1),
        ("spread_rate", -0.1),
        ("borrow_rate", -0.1),
        ("fee_rate", math.inf),
    ],
)
def test_execution_economics_profile_rejects_invalid_rates(
    field: str,
    value: float,
) -> None:
    profile_type = _profile_type()
    kwargs = {"name": "invalid", field: value}

    with pytest.raises(ValueError, match=field):
        profile_type(**kwargs)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, math.inf])
def test_execution_economics_profile_rejects_invalid_participation(value: float) -> None:
    profile_type = _profile_type()

    with pytest.raises(ValueError, match="max_participation_rate"):
        profile_type(name="invalid", max_participation_rate=value)


def test_execution_economics_profile_rejects_borrow_rate_when_unavailable() -> None:
    profile_type = _profile_type()

    with pytest.raises(ValueError, match="borrow_rate must be zero"):
        profile_type(name="invalid", borrow_available=False, borrow_rate=0.05)
