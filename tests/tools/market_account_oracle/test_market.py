from dataclasses import FrozenInstanceError
from decimal import Decimal, getcontext, localcontext

import pytest

from tools.market_account_oracle.market import MarketRow, market_order_facts


def _row(
    index: int = 1,
    open_price: float = 120.0,
    close_price: float = 124.0,
    volume: float = 10.0,
) -> MarketRow:
    return MarketRow(index, open_price, close_price, volume)


def _facts(**overrides: object):
    arguments: dict[str, object] = {
        "current": _row(),
        "previous": MarketRow(0, 99.0, 100.0, 10.0),
        "volume_unit": "base_asset",
        "contract_multiplier": 2.0,
        "direction": 1,
    }
    arguments.update(overrides)
    return market_order_facts(**arguments)  # type: ignore[arg-type]


def test_prior_close_capacity_is_separate_from_processing_open_price() -> None:
    previous = MarketRow(0, 98.0, 100.0, 10.0)
    current = MarketRow(1, 120.0, 124.0, 10.0)

    prior = market_order_facts(
        current,
        previous=previous,
        volume_unit="base_asset",
        contract_multiplier=2.0,
        direction=1,
    )
    current_bar = market_order_facts(
        current,
        volume_unit="base_asset",
        contract_multiplier=2.0,
        direction=1,
        processing_bar_volume_capacity=True,
    )

    assert prior.execution_price == Decimal("120.0")
    assert prior.reference_index == 0
    assert prior.reference_price == Decimal("100.0")
    assert prior.raw_volume == Decimal("10.0")
    assert prior.market_notional == Decimal("1000.00")
    assert prior.quantity_capacity == Decimal("5.0")
    assert current_bar.reference_index == 1
    assert current_bar.reference_price == Decimal("120.0")
    assert current_bar.market_notional == Decimal("1200.00")
    assert current_bar.quantity_capacity == Decimal("5.0")


@pytest.mark.parametrize(
    ("unit", "expected_notional", "expected_quantity"),
    [
        ("contracts", Decimal("2000.00"), Decimal("10.0")),
        ("quote_notional", Decimal("10.0"), None),
    ],
)
def test_capacity_conversion_uses_the_declared_market_volume_unit(
    unit: str,
    expected_notional: Decimal,
    expected_quantity: Decimal | None,
) -> None:
    facts = _facts(volume_unit=unit)

    assert facts.market_notional == expected_notional
    assert facts.quantity_capacity == expected_quantity


def test_first_prior_capacity_row_has_no_fabricated_history() -> None:
    facts = market_order_facts(
        _row(index=0, open_price=12.5, volume=9.0),
        volume_unit="base_asset",
        contract_multiplier=3.0,
        direction=-1,
    )

    assert facts.execution_price == Decimal("12.5")
    assert facts.reference_index is None
    assert facts.reference_price == Decimal("12.5")
    assert facts.raw_volume == Decimal("0")
    assert facts.market_notional == Decimal("0")
    assert facts.quantity_capacity is None


def test_tick_rounding_preserves_binary64_boundary_and_adverse_side() -> None:
    boundary = market_order_facts(
        _row(index=0, open_price=0.35),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.1,
        direction=1,
    )
    buy_stress = market_order_facts(
        _row(index=0, open_price=1.3),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.5,
        direction=1,
        adverse_tick_rounding=True,
    )
    sell_stress = market_order_facts(
        _row(index=0, open_price=1.3),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.5,
        direction=-1,
        adverse_tick_rounding=True,
    )
    tie = market_order_facts(
        _row(index=0, open_price=1.25),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.5,
        direction=1,
    )

    assert boundary.execution_price == Decimal("0.30000000000000004")
    assert buy_stress.execution_price == Decimal("1.5")
    assert sell_stress.execution_price == Decimal("1.0")
    assert tie.execution_price == Decimal("1.0")


def test_tick_rounding_uses_at_least_one_tick_when_rounded_to_zero() -> None:
    ordinary = market_order_facts(
        _row(index=0, open_price=0.01),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.5,
        direction=1,
    )
    adverse_sell = market_order_facts(
        _row(index=0, open_price=0.01),
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.5,
        direction=-1,
        adverse_tick_rounding=True,
    )

    assert ordinary.execution_price == Decimal("0.5")
    assert adverse_sell.execution_price == Decimal("0.5")


def test_tick_scale_overflow_fails_closed() -> None:
    with pytest.raises(ValueError):
        market_order_facts(
            _row(index=0, open_price=1e308),
            volume_unit="quote_notional",
            contract_multiplier=1.0,
            tick_size=1e-308,
            direction=1,
        )


@pytest.mark.parametrize(
    "row_values",
    [
        (True, 1.0, 1.0, 1.0),
        (-1, 1.0, 1.0, 1.0),
        (0, True, 1.0, 1.0),
        (0, float("nan"), 1.0, 1.0),
        (0, 0.0, 1.0, 1.0),
        (0, 1.0, float("inf"), 1.0),
        (0, Decimal("1e-400"), 1.0, 1.0),
        (0, 1.0, Decimal("Infinity"), 1.0),
        (0, 1.0, 1.0, Decimal("NaN")),
        (0, 1.0, 1.0, True),
        (0, 1.0, 1.0, -1.0),
    ],
)
def test_market_row_rejects_invalid_indices_and_nonfinite_or_boolean_values(
    row_values: tuple[object, object, object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MarketRow(*row_values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"volume_unit": "BASE_ASSET"},
        {"contract_multiplier": True},
        {"contract_multiplier": 0.0},
        {"contract_multiplier": float("inf")},
        {"contract_multiplier": Decimal("1e-400")},
        {"tick_size": -0.1},
        {"tick_size": float("nan")},
        {"tick_size": float("inf")},
        {"tick_size": Decimal("1e-400")},
        {"direction": True},
        {"direction": 0},
        {"adverse_tick_rounding": 1},
        {"processing_bar_volume_capacity": 0},
    ],
)
def test_market_order_facts_rejects_invalid_units_and_controls(
    arguments: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _facts(**arguments)


def test_capacity_modes_require_their_exact_adjacent_row_contract() -> None:
    with pytest.raises((TypeError, ValueError)):
        _facts(previous=None)
    with pytest.raises((TypeError, ValueError)):
        _facts(previous=MarketRow(2, 99.0, 100.0, 10.0))
    with pytest.raises((TypeError, ValueError)):
        _facts(processing_bar_volume_capacity=True, previous=MarketRow(0, 99, 100, 10))


def test_decimal_capacity_is_independent_of_callers_context_precision() -> None:
    row = _row(open_price=12.34)
    previous = MarketRow(0, 9.0, 9.87, 3.21)
    normal = market_order_facts(
        row,
        previous=previous,
        volume_unit="base_asset",
        contract_multiplier=1.1,
        direction=1,
    )

    with localcontext() as context:
        context.prec = 2
        low_precision = market_order_facts(
            row,
            previous=previous,
            volume_unit="base_asset",
            contract_multiplier=1.1,
            direction=1,
        )
        assert getcontext().prec == 2

    assert normal.market_notional == Decimal("31.6827")
    assert low_precision.market_notional == normal.market_notional
    assert low_precision.quantity_capacity == normal.quantity_capacity


def test_market_rows_and_facts_are_immutable() -> None:
    row = _row()
    facts = _facts()

    with pytest.raises(FrozenInstanceError):
        row.index = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        facts.execution_price = Decimal("1")  # type: ignore[misc]
