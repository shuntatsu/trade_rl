"""Point-in-time rules: close stamps, publication lags, holidays and as-of joins."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest

from trade_rl.integrations.cross_asset.pit import (
    ReleaseRule,
    add_business_days,
    align_asof,
    is_business_day,
    knowable_at_array,
    knowable_at_utc,
    publication_date,
    session_close_utc,
    us_publication_holidays,
)

NY = "America/New_York"
DAILY_1 = ReleaseRule(lag_business_days=1)
WEEKLY_MONDAY = ReleaseRule(lag_business_days=1, weekly_release_weekday=0)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_us_publication_holidays_cover_federal_and_exchange_closures() -> None:
    holidays_2025 = us_publication_holidays(2025)
    assert date(2025, 11, 27) in holidays_2025  # Thanksgiving
    assert date(2025, 4, 18) in holidays_2025  # Good Friday (exchange only)
    assert date(2025, 6, 19) in holidays_2025  # Juneteenth
    assert date(2025, 1, 20) in holidays_2025  # MLK
    assert date(2025, 12, 25) in holidays_2025
    assert date(2025, 11, 11) in holidays_2025  # federal, conservative union
    # Saturday holidays are also observed on the preceding Friday.
    holidays_2026 = us_publication_holidays(2026)
    assert date(2026, 7, 4) in holidays_2026
    assert date(2026, 7, 3) in holidays_2026
    assert date(2026, 4, 3) in holidays_2026  # Good Friday 2026
    # Sunday holidays are observed on the following Monday.
    assert date(2027, 7, 5) in us_publication_holidays(2027)
    assert date(2027, 12, 24) in us_publication_holidays(2027)


def test_business_day_arithmetic_skips_weekends_and_holidays() -> None:
    assert is_business_day(date(2025, 11, 26))
    assert not is_business_day(date(2025, 11, 27))
    assert not is_business_day(date(2025, 11, 29))
    assert add_business_days(date(2025, 11, 26), 1) == date(2025, 11, 28)
    assert add_business_days(date(2025, 11, 7), 1) == date(2025, 11, 10)  # Fri->Mon
    assert add_business_days(date(2025, 11, 7), 0) == date(2025, 11, 7)
    assert add_business_days(date(2025, 11, 8), 0) == date(2025, 11, 10)
    with pytest.raises(ValueError):
        add_business_days(date(2025, 11, 7), -1)


def test_daily_release_is_knowable_only_after_the_whole_publication_day() -> None:
    # Thursday observation -> published Friday -> knowable Saturday 00:00 New York.
    thursday = knowable_at_utc(
        date(2025, 1, 16), venue_tz=NY, close_local=time(17, 0), rule=DAILY_1
    )
    assert thursday == _utc(2025, 1, 18, 5)  # Saturday 00:00 EST
    friday = knowable_at_utc(
        date(2025, 11, 14), venue_tz=NY, close_local=time(16, 0), rule=DAILY_1
    )
    assert friday == _utc(2025, 11, 18, 5)  # Mon 11-17 publication, Tue 00:00 EST


def test_thanksgiving_week_delays_publication_instead_of_leaking() -> None:
    wednesday = date(2025, 11, 26)
    assert publication_date(wednesday, DAILY_1) == date(2025, 11, 28)
    stamp = knowable_at_utc(
        wednesday, venue_tz=NY, close_local=time(16, 0), rule=DAILY_1
    )
    assert stamp == _utc(2025, 11, 29, 5)


def test_weekend_observation_dates_publish_after_the_next_full_business_day() -> None:
    # ICE BofA spreads carry calendar month-end rows on weekends (2023-12-31 is
    # a Sunday): never released before Monday plus the lag, i.e. Tuesday.
    sunday = date(2023, 12, 31)
    assert publication_date(sunday, DAILY_1) == date(2024, 1, 3)  # Jan 1 holiday
    saturday = date(2024, 8, 31)
    assert publication_date(saturday, DAILY_1) == date(2024, 9, 4)  # Sep 2 is Labor Day
    stamp = knowable_at_utc(sunday, venue_tz=NY, close_local=time(17, 0), rule=DAILY_1)
    assert stamp == _utc(2024, 1, 4, 5)


def test_weekly_monday_release_waits_for_the_next_monday() -> None:
    # Friday value is released the following Monday plus one business day buffer.
    assert publication_date(date(2025, 11, 14), WEEKLY_MONDAY) == date(2025, 11, 18)
    # A Monday value is not in that Monday's release.
    assert publication_date(date(2025, 11, 17), WEEKLY_MONDAY) == date(2025, 11, 25)
    # Holiday Monday rolls to Tuesday before the buffer is added.
    assert publication_date(date(2025, 1, 17), WEEKLY_MONDAY) == date(2025, 1, 22)


def test_dst_transitions_move_utc_close_but_not_local_close() -> None:
    before = session_close_utc(date(2025, 3, 7), NY, time(16, 0))
    after = session_close_utc(date(2025, 3, 10), NY, time(16, 0))
    assert before == _utc(2025, 3, 7, 21)  # EST
    assert after == _utc(2025, 3, 10, 20)  # EDT after the 2025-03-09 change
    autumn_before = session_close_utc(date(2025, 10, 31), NY, time(16, 0))
    autumn_after = session_close_utc(date(2025, 11, 3), NY, time(16, 0))
    assert autumn_before == _utc(2025, 10, 31, 20)
    assert autumn_after == _utc(2025, 11, 3, 21)
    # US and UK change clocks on different dates: the gap is 4h, not 5h, between
    # 2025-03-09 and 2025-03-30.
    ny = session_close_utc(date(2025, 3, 17), NY, time(16, 30))
    london = session_close_utc(date(2025, 3, 17), "Europe/London", time(16, 30))
    assert ny - london == timedelta(hours=4)
    ny_winter = session_close_utc(date(2025, 3, 7), NY, time(16, 30))
    london_winter = session_close_utc(date(2025, 3, 7), "Europe/London", time(16, 30))
    assert ny_winter - london_winter == timedelta(hours=5)


def test_us_and_asia_close_ordering_is_resolved_in_utc() -> None:
    day = date(2025, 1, 14)
    tokyo = session_close_utc(day, "Asia/Tokyo", time(15, 30))
    hong_kong = session_close_utc(day, "Asia/Hong_Kong", time(16, 0))
    london = session_close_utc(day, "Europe/London", time(16, 30))
    new_york = session_close_utc(day, NY, time(16, 0))
    assert tokyo < hong_kong < london < new_york
    # Tokyo's next session closes after New York's previous close.
    assert session_close_utc(day, NY, time(16, 0)) < session_close_utc(
        day + timedelta(days=1), "Asia/Tokyo", time(15, 30)
    )
    # Even with identical lags a later venue date never becomes knowable earlier.
    tokyo_stamp = knowable_at_utc(
        day, venue_tz="Asia/Tokyo", close_local=time(15, 30), rule=DAILY_1
    )
    new_york_stamp = knowable_at_utc(
        day, venue_tz=NY, close_local=time(16, 0), rule=DAILY_1
    )
    assert tokyo_stamp == new_york_stamp  # same US publication day end
    assert tokyo_stamp > new_york.astimezone(UTC)


def test_knowable_at_is_monotone_and_after_close_over_a_full_year() -> None:
    days = np.arange("2025-01-01", "2026-01-01", dtype="datetime64[D]")
    for rule in (DAILY_1, WEEKLY_MONDAY, ReleaseRule(lag_business_days=3)):
        stamps = knowable_at_array(
            days, venue_tz="Asia/Tokyo", close_local=time(15, 30), rule=rule
        )
        assert np.all(np.diff(stamps.astype(np.int64)) >= 0)
        closes = np.asarray(
            [
                np.datetime64(
                    session_close_utc(
                        day.astype(date), "Asia/Tokyo", time(15, 30)
                    ).replace(tzinfo=None),
                    "ns",
                )
                for day in days
            ]
        )
        assert np.all(stamps > closes)


def test_release_rule_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        ReleaseRule(lag_business_days=-1)
    with pytest.raises(ValueError):
        ReleaseRule(lag_business_days=1, weekly_release_weekday=7)


def _ns(value: datetime) -> int:
    return int(np.datetime64(value.replace(tzinfo=None), "ns").astype(np.int64))


def test_asof_alignment_uses_only_values_known_by_the_decision_time() -> None:
    knowable = np.asarray(
        [_ns(_utc(2025, 1, 11, 5)), _ns(_utc(2025, 1, 14, 5))], dtype=np.int64
    )
    grid = np.asarray(
        [
            _ns(_utc(2025, 1, 11, 4, 59)),  # one minute before the first release
            _ns(_utc(2025, 1, 11, 5)),  # exactly at the release: allowed
            _ns(_utc(2025, 1, 12, 8)),  # Sunday crypto bar carries the Friday value
            _ns(_utc(2025, 1, 14, 4)),
            _ns(_utc(2025, 1, 14, 8)),
        ],
        dtype=np.int64,
    )
    index = align_asof(knowable, grid)
    assert index.tolist() == [-1, 0, 0, 0, 1]


def test_asof_alignment_never_changes_when_future_values_change() -> None:
    knowable = np.asarray(
        [_ns(_utc(2025, 1, 11, 5)) + i * 86_400_000_000_000 for i in range(10)],
        dtype=np.int64,
    )
    base = np.arange(10, dtype=np.float64)
    grid = np.asarray(
        [_ns(_utc(2025, 1, 14, 12)), _ns(_utc(2025, 1, 16, 12))], dtype=np.int64
    )
    index = align_asof(knowable, grid)
    changed = base.copy()
    changed[index.max() + 1 :] += 1_000.0
    assert np.array_equal(base[index], changed[index])


def test_asof_alignment_rejects_non_monotone_release_times() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        align_asof(np.asarray([5, 4], dtype=np.int64), np.asarray([6], dtype=np.int64))


def test_asof_alignment_allows_equal_release_times_and_takes_the_latest_row() -> None:
    knowable = np.asarray([10, 10, 10, 20], dtype=np.int64)
    assert align_asof(knowable, np.asarray([10, 15, 20], dtype=np.int64)).tolist() == [
        2,
        2,
        3,
    ]
