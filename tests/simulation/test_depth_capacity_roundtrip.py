"""Exact capacity composition with the public bounded-notional domain."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from trade_rl.simulation.depth_calibration import (
    DepthBand,
    DepthBandSnapshot,
    OrderCostBounds,
    Side,
    calibrate_depth_costs,
    depth_capacity,
    order_cost_bounds,
)

TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _book(p1: Fraction, p2: Fraction, *, offset: int = 0) -> DepthBandSnapshot:
    bands = (DepthBand(p1, Decimal(1)), DepthBand(p2, Decimal(10)))
    return DepthBandSnapshot(TIME + timedelta(seconds=offset), bands, bands)


def _complex_book() -> DepthBandSnapshot:
    p1 = 1 + Fraction(1, 2**4000)
    p2 = 2 + Fraction(1, 3**2400)
    # Independent two-band algebra: first unit at 100*p1 and the rest at
    # 100*p2; solve (100*p1 + (Q-1)*100*p2)/Q == 150.
    exact = (p2 - p1) / (p2 - Fraction(3, 2))
    assert 1 < exact < 10
    assert max(exact.numerator.bit_length(), exact.denominator.bit_length()) > 4096
    return _book(p1, p2)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_capacity_refuses_unqueryable_exact_result(side: Side) -> None:
    with pytest.raises(ValueError, match="capacity_notional"):
        depth_capacity(
            [_complex_book()], side=side, quantile=1, max_average_cost_bps=150
        )


@pytest.mark.parametrize(
    ("p1", "p2"), [(Fraction(1), Fraction(2)), (Fraction(9, 8), Fraction(19, 9))]
)
def test_representable_capacity_round_trips_exactly(p1: Fraction, p2: Fraction) -> None:
    snapshot = _book(p1, p2)
    result = depth_capacity(
        [snapshot], side="buy", quantile=1, max_average_cost_bps=150
    )
    expected = (p2 - p1) / (p2 - Fraction(3, 2))
    assert result.capacity_notional == expected
    bound = order_cost_bounds(snapshot, side="buy", notional=result.capacity_notional)
    assert isinstance(bound, OrderCostBounds)
    assert bound.average_upper_bps == 150
    cal = calibrate_depth_costs(
        [snapshot], notionals=[result.capacity_notional], quantiles=[1], sides=["buy"]
    )
    assert cal.cell(side="buy", notional=expected).bounds_at(1).average_upper_bps == 150
    assert cal.input_digest == result.input_digest


def test_only_selected_quantile_capacity_must_be_representable() -> None:
    snapshots = [_complex_book(), _book(Fraction(1), Fraction(3), offset=1)]
    result = depth_capacity(snapshots, side="buy", quantile=1, max_average_cost_bps=150)
    assert result.capacity_notional == Fraction(4, 3)
    cal = calibrate_depth_costs(
        snapshots, notionals=[result.capacity_notional], quantiles=[1], sides=["buy"]
    )
    assert cal.cells[0].bounds_at(1).average_upper_bps == 150
    with pytest.raises(ValueError, match="capacity_notional"):
        depth_capacity(
            snapshots, side="buy", quantile=Fraction(1, 2), max_average_cost_bps=150
        )


@pytest.mark.parametrize("value", [Decimal("1e-308"), Decimal("1e308")])
def test_positive_domain_endpoints_are_queryable(value: Decimal) -> None:
    bands = (DepthBand(Fraction(1), value),)
    snapshot = DepthBandSnapshot(TIME, bands, bands)
    result = depth_capacity(
        [snapshot], side="sell", quantile=1, max_average_cost_bps=100
    )
    assert result.capacity_notional == Fraction(value)
    bound = order_cost_bounds(snapshot, side="sell", notional=result.capacity_notional)
    assert isinstance(bound, OrderCostBounds)
    assert bound.average_upper_bps == 100


def test_zero_capacity_remains_explicit() -> None:
    result = depth_capacity(
        [_complex_book()], side="buy", quantile=1, max_average_cost_bps=0
    )
    assert result.capacity_notional == 0


def test_digest_identifies_snapshots_not_query_grid() -> None:
    snapshots = [_book(Fraction(1), Fraction(2))]
    a = calibrate_depth_costs(snapshots, notionals=[1], quantiles=[1], sides=["buy"])
    b = calibrate_depth_costs(
        snapshots, notionals=[2], quantiles=[Fraction(1, 2)], sides=["sell"]
    )
    assert a.input_digest == b.input_digest
    assert (a.notionals, a.quantiles, a.sides) != (b.notionals, b.quantiles, b.sides)
