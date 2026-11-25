"""Exact UTC scalar timestamps for external research consumers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import numpy as np
import pytest

from trade_rl.data.timestamps import utc_timestamp_ns


@pytest.mark.parametrize(
    "instant, expected",
    [
        (datetime(1970, 1, 1, tzinfo=UTC), 0),
        (datetime(1970, 1, 1, 9, tzinfo=timezone(timedelta(hours=9))), 0),
        (datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC), -1000),
        (datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=UTC), 1000),
        (datetime(2024, 2, 29, 0, 0, tzinfo=UTC), 1709164800000000000),
        (np.datetime64("1970-01-01T00:00:00.000000001", "ns"), 1),
        (np.datetime64("1969-12-31T23:59:59.999999999", "ns"), -1),
        (np.datetime64(7, "3ms"), 21000000),
        (np.datetime64(2000, "ps"), 2),
        (np.datetime64("1970-02", "M"), 2678400000000000),
        (np.datetime64("1677-09-21T00:12:43.145224193", "ns"), -(2**63) + 1),
        (np.datetime64("2262-04-11T23:47:16.854775807", "ns"), 2**63 - 1),
    ],
)
def test_exact_utc_nanoseconds(instant, expected):
    actual = utc_timestamp_ns(instant)
    assert type(actual) is int
    assert actual == expected


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, tzinfo=NoOffset()),
        datetime(1500, 1, 1, tzinfo=UTC),
        datetime(2500, 1, 1, tzinfo=UTC),
        np.datetime64("NaT"),
        np.datetime64("2500-01-01", "D"),
        np.datetime64(1, "ps"),
        np.datetime64(-1, "ps"),
        np.datetime64("0001-01", "M"),
    ],
)
def test_ambiguous_or_unrepresentable_times_fail_closed(value):
    with pytest.raises(ValueError):
        utc_timestamp_ns(value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        1,
        1.5,
        "2024-01-01T00:00:00Z",
        np.array("2024-01-01", dtype="datetime64[D]"),
        np.ma.array(np.datetime64("2024-01-01"), mask=True),
    ],
)
def test_scalar_types_only(value):
    with pytest.raises(TypeError):
        utc_timestamp_ns(value)


def test_datetime_subclass_cannot_discard_finer_precision():
    class Extended(datetime):
        @property
        def nanosecond(self):
            return 123

    with pytest.raises(TypeError):
        utc_timestamp_ns(Extended(2024, 1, 1, tzinfo=UTC))
