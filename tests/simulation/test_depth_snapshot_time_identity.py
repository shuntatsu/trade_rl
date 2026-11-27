"""Synthetic snapshot time/digest identity and numeric-domain controls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from fractions import Fraction

import pytest

from trade_rl.simulation.depth_calibration import (
    CostAssumptionCheck,
    DepthBand,
    DepthBandSnapshot,
    calibrate_depth_costs,
    classify_assumed_cost,
    depth_capacity,
)


class _ChangingZone(tzinfo):
    def __init__(self) -> None:
        self.offset = timedelta(0)
        self.calls = 0

    def utcoffset(self, dt: datetime | None) -> timedelta:
        self.calls += 1
        return self.offset


def _snapshot(stamp: datetime) -> DepthBandSnapshot:
    bands = (DepthBand(Decimal(1), Decimal(10)),)
    return DepthBandSnapshot(stamp, bands, bands)


def test_snapshot_does_not_retain_mutable_timezone() -> None:
    zone = _ChangingZone()
    snapshot = _snapshot(datetime(2024, 1, 1, 0, tzinfo=zone))
    before = calibrate_depth_costs(
        [snapshot], notionals=[1], quantiles=[1], by_utc_hour=True
    )
    zone.offset = timedelta(hours=1)
    after = calibrate_depth_costs(
        [snapshot], notionals=[1], quantiles=[1], by_utc_hour=True
    )
    assert snapshot.time.tzinfo is UTC
    assert before == after
    assert before.input_digest == after.input_digest
    assert {cell.hour_utc for cell in after.cells} == {None, 0}
    capacity = depth_capacity(
        [snapshot], side="buy", quantile=1, max_average_cost_bps=100, hour_utc=0
    )
    assert capacity.input_digest == before.input_digest
    assert zone.calls == 1


def test_zero_offset_is_sampled_only_once() -> None:
    zone = _ChangingZone()
    snapshot = _snapshot(datetime(2024, 1, 1, tzinfo=zone))
    assert zone.calls == 1
    assert snapshot.time == datetime(2024, 1, 1, tzinfo=UTC)
    assert snapshot.time.tzinfo is UTC


def test_named_utc_zone_normalizes_without_changing_instant() -> None:
    stamp = datetime(2024, 2, 29, 12, 34, 56, 123456, tzinfo=UTC)
    named = timezone(timedelta(0), "zero")
    actual = _snapshot(stamp.replace(tzinfo=named))
    expected = _snapshot(stamp)
    assert actual == expected
    assert actual.time.tzinfo is UTC
    actual_result = calibrate_depth_costs([actual], notionals=[1], quantiles=[1])
    expected_result = calibrate_depth_costs([expected], notionals=[1], quantiles=[1])
    assert actual_result.input_digest == expected_result.input_digest


@pytest.mark.parametrize(
    "stamp",
    [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=9))),
    ],
)
def test_non_utc_and_naive_time_still_refused(stamp: datetime) -> None:
    with pytest.raises(ValueError):
        _snapshot(stamp)


def test_foreign_datetime_subclass_refused() -> None:
    class ForeignDatetime(datetime):
        pass

    with pytest.raises(ValueError, match="native"):
        _snapshot(ForeignDatetime(2024, 1, 1, tzinfo=UTC))


def test_classifier_keeps_exact_wide_range_comparison() -> None:
    assert (
        classify_assumed_cost(
            Fraction(1, 10**320),
            lower_bound_bps=Fraction(1, 10**400),
            upper_bound_bps=Fraction(10**400),
        )
        is CostAssumptionCheck.WITHIN_BOUNDS
    )
