"""Synthetic admission controls for immutable depth snapshots and queries."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction

import pytest

from trade_rl.simulation.depth_calibration import (
    CostAssumptionCheck,
    DepthBand,
    DepthBandSnapshot,
    OrderCostBounds,
    calibrate_depth_costs,
    classify_assumed_cost,
    depth_capacity,
    order_cost_bounds,
)

TIME = datetime(2024, 1, 1, tzinfo=UTC)


class _ChangingDecimal(Decimal):
    amount: int = 1

    def as_integer_ratio(self) -> tuple[int, int]:
        return self.amount, 1


class _ChangingPair(tuple):
    calls: int = 0

    def __getitem__(self, index):
        self.calls += 1
        if self.calls > 2:
            return Decimal(-1)
        return super().__getitem__(index)


class _ForeignFraction(Fraction):
    pass


class _ForeignInt(int):
    pass


class _ForeignFloat(float):
    pass


def test_snapshot_refuses_mutable_decimal_subclass() -> None:
    percentage = _ChangingDecimal("1")
    bands = (DepthBand(percentage, Decimal(10)),)
    with pytest.raises(ValueError, match="percentage"):
        DepthBandSnapshot(TIME, bands, bands)


def test_snapshot_refuses_foreign_pair_before_indexing() -> None:
    pair = _ChangingPair((Decimal(1), Decimal(10)))
    native = (DepthBand(Decimal(1), Decimal(10)),)
    with pytest.raises(ValueError, match="bands"):
        DepthBandSnapshot(TIME, (pair,), native)  # type: ignore[arg-type]
    assert pair.calls == 0


@pytest.mark.parametrize(
    "value", [_ForeignFraction(1, 2), _ForeignInt(1), _ForeignFloat(1.0)]
)
def test_classifier_refuses_foreign_numeric_types(value) -> None:
    with pytest.raises(ValueError, match="assumed_cost_bps"):
        classify_assumed_cost(value, lower_bound_bps=0, upper_bound_bps=100)


@pytest.mark.parametrize("percentage", [1, Decimal("1"), Fraction(1)])
def test_native_numeric_types_and_plain_pair_preserved(percentage) -> None:
    bands = ((percentage, 10),)
    snapshot = DepthBandSnapshot(TIME, bands, bands)  # type: ignore[arg-type]
    bound = order_cost_bounds(snapshot, side="buy", notional=1)
    assert isinstance(bound, OrderCostBounds)
    assert bound.average_upper_bps == 100
    assert snapshot.ask_bands == (DepthBand(percentage, 10),)


def test_native_float_threshold_still_supported() -> None:
    assert (
        classify_assumed_cost(0.5, lower_bound_bps=0, upper_bound_bps=100)
        is CostAssumptionCheck.UNRESOLVABLE
    )


def test_native_depth_band_still_supported() -> None:
    bands = (DepthBand(Decimal(1), Decimal(10)),)
    snapshot = DepthBandSnapshot(TIME, bands, bands)
    bound = order_cost_bounds(snapshot, side="sell", notional=Fraction(1))
    assert isinstance(bound, OrderCostBounds)
    assert bound.average_upper_bps == 100


class _ForeignSnapshot(DepthBandSnapshot):
    def bands_for(self, side):
        return (DepthBand(Decimal(-1), Decimal(10)),)


def test_order_refuses_foreign_snapshot() -> None:
    bands = (DepthBand(Decimal(1), Decimal(10)),)
    snapshot = _ForeignSnapshot(TIME, bands, bands)
    with pytest.raises(ValueError, match="DepthBandSnapshot"):
        order_cost_bounds(snapshot, side="buy", notional=1)


def test_calibration_refuses_foreign_snapshot() -> None:
    bands = (DepthBand(Decimal(1), Decimal(10)),)
    snapshot = _ForeignSnapshot(TIME, bands, bands)
    with pytest.raises(ValueError, match="DepthBandSnapshot"):
        calibrate_depth_costs([snapshot], notionals=[1], quantiles=[1])


def test_capacity_refuses_foreign_snapshot() -> None:
    bands = (DepthBand(Decimal(1), Decimal(10)),)
    snapshot = _ForeignSnapshot(TIME, bands, bands)
    with pytest.raises(ValueError, match="DepthBandSnapshot"):
        depth_capacity([snapshot], side="buy", quantile=1, max_average_cost_bps=100)


class _ForeignSide(str):
    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


@pytest.mark.parametrize("side", [_ForeignSide("hold"), []], ids=["foreign", "list"])
def test_query_refuses_invalid_side(side) -> None:
    bands = (DepthBand(1, 10),)
    snapshot = DepthBandSnapshot(TIME, bands, bands)
    with pytest.raises(ValueError, match="side"):
        order_cost_bounds(snapshot, side=side, notional=1)
    with pytest.raises(ValueError, match="side"):
        calibrate_depth_costs([snapshot], notionals=[1], quantiles=[1], sides=[side])
    with pytest.raises(ValueError, match="side"):
        depth_capacity([snapshot], side=side, quantile=1, max_average_cost_bps=100)
    cal = calibrate_depth_costs([snapshot], notionals=[1], quantiles=[1])
    with pytest.raises(ValueError, match="side"):
        cal.cell(side=side, notional=1)


@pytest.mark.parametrize("hour", [True, 1.0, _ForeignInt(1)])
def test_query_refuses_non_native_hour(hour) -> None:
    bands = (DepthBand(1, 10),)
    snapshot = DepthBandSnapshot(TIME.replace(hour=1), bands, bands)
    cal = calibrate_depth_costs(
        [snapshot], notionals=[1], quantiles=[1], by_utc_hour=True
    )
    with pytest.raises(ValueError, match="hour_utc"):
        cal.cell(side="buy", notional=1, hour_utc=hour)
    with pytest.raises(ValueError, match="hour_utc"):
        depth_capacity(
            [snapshot],
            side="buy",
            quantile=1,
            max_average_cost_bps=100,
            hour_utc=hour,
        )


def test_native_query_keys_still_work() -> None:
    bands = (DepthBand(1, 10),)
    snapshot = DepthBandSnapshot(TIME.replace(hour=1), bands, bands)
    cal = calibrate_depth_costs(
        [snapshot], notionals=[1], quantiles=[1], by_utc_hour=True
    )
    assert cal.cell(side="buy", notional=1, hour_utc=1).snapshot_count == 1
    assert cal.cell(side="sell", notional=1).snapshot_count == 1
    assert (
        depth_capacity(
            [snapshot], side="buy", quantile=1, max_average_cost_bps=100, hour_utc=1
        ).capacity_notional
        == 10
    )


class _ForeignBool:
    @property
    def __class__(self):
        return bool


def test_query_refuses_foreign_grouping_flag() -> None:
    bands = (DepthBand(1, 10),)
    snapshot = DepthBandSnapshot(TIME, bands, bands)
    with pytest.raises(ValueError, match="boolean"):
        calibrate_depth_costs(
            [snapshot],
            notionals=[1],
            quantiles=[1],
            by_utc_hour=_ForeignBool(),  # type: ignore[arg-type]
        )
