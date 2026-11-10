"""Catalog invariants and parsers for the key-less public sources."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pytest

from trade_rl.integrations.cross_asset.catalog import (
    CATALOG,
    SOURCES,
    SeriesKind,
    SeriesSpec,
    series_by_key,
)
from trade_rl.integrations.cross_asset.observations import build_obs_series
from trade_rl.integrations.cross_asset.parsers import (
    SourceBlockedError,
    SourceDataError,
    diff_observations,
    parse_fred_csv,
    parse_stooq_csv,
    parse_yahoo_chart,
)
from trade_rl.integrations.cross_asset.pit import (
    knowable_at_utc,
    session_close_utc,
)


def test_catalog_keys_are_unique_and_sources_are_documented() -> None:
    keys = [spec.key for spec in CATALOG]
    assert len(keys) == len(set(keys))
    assert {spec.source for spec in CATALOG} <= set(SOURCES)
    for source in SOURCES.values():
        assert source.terms_url.startswith("https://")
        assert source.terms_note
    assert series_by_key("DGS10").symbol == "DGS10"
    with pytest.raises(KeyError):
        series_by_key("nope")


def test_every_series_uses_https_a_valid_zone_and_a_conservative_lag() -> None:
    probe = date(2025, 6, 10)
    for spec in CATALOG:
        assert spec.url().startswith("https://")
        assert "key" not in spec.url().lower()
        assert spec.release.lag_business_days >= 1, spec.key
        assert spec.max_staleness_hours > 0.0
        close = session_close_utc(probe, spec.venue_tz, spec.close_local)
        stamp = knowable_at_utc(
            probe,
            venue_tz=spec.venue_tz,
            close_local=spec.close_local,
            rule=spec.release,
        )
        assert stamp > close
        assert (stamp - close).total_seconds() >= 12 * 3600, spec.key


def test_yahoo_requests_explicit_daily_bars_not_range_max() -> None:
    url = series_by_key("y_gold").url()
    assert "interval=1d" in url and "period1=" in url and "period2=" in url
    assert "range=max" not in url  # Yahoo downgrades range=max to monthly bars
    assert "GC%3DF" in url


def test_lags_reflect_observed_publication_delays() -> None:
    assert series_by_key("SP500").release.lag_business_days >= 3
    assert series_by_key("DTWEXBGS").release.weekly_release_weekday == 0
    assert series_by_key("DEXJPUS").release.weekly_release_weekday == 0
    assert series_by_key("DCOILWTICO").release.lag_business_days >= 5
    assert series_by_key("DGS10").kind is SeriesKind.LEVEL
    assert series_by_key("SP500").kind is SeriesKind.PRICE


FRED = b"observation_date,DGS10\n2025-01-01,\n2025-01-02,4.57\n2025-01-03,.\n2025-01-06,4.62\n"


def test_fred_csv_drops_missing_rows_but_reports_them() -> None:
    parsed = parse_fred_csv(FRED, "DGS10")
    assert parsed.dates.tolist() == [
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-06"),
    ]
    assert parsed.values.tolist() == [4.57, 4.62]
    assert parsed.rows_total == 4
    assert parsed.missing_rows == 2
    assert parsed.last_row_date == date(2025, 1, 6)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"<html>blocked</html>",
        b"date,DGS10\n2025-01-02,1.0\n",
        b"observation_date,OTHER\n2025-01-02,1.0\n",
        b"observation_date,DGS10\n2025-01-03,1.0\n2025-01-02,2.0\n",
        b"observation_date,DGS10\n2025-01-02,1.0\n2025-01-02,2.0\n",
        b"observation_date,DGS10\n2025-01-02,abc\n",
        b"observation_date,DGS10\n2025-01-02,inf\n",
        b"observation_date,DGS10\n01/02/2025,1.0\n",
    ],
)
def test_fred_csv_rejects_malformed_payloads(payload: bytes) -> None:
    with pytest.raises(SourceDataError):
        parse_fred_csv(payload, "DGS10")


STOOQ = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2025-01-02,1,2,0.5,1.5,100\n"
    b"2025-01-03,1.5,2,1,1.75,120\n"
)


def test_stooq_csv_reads_close_column() -> None:
    parsed = parse_stooq_csv(STOOQ)
    assert parsed.values.tolist() == [1.5, 1.75]
    assert parsed.rows_total == 2


def test_stooq_javascript_challenge_is_reported_as_blocked_not_solved() -> None:
    challenge = (
        b"<!DOCTYPE html><html><body><noscript>This site requires JavaScript to "
        b'verify your browser.</noscript><script>fetch("/__verify")</script>'
    )
    with pytest.raises(SourceBlockedError):
        parse_stooq_csv(challenge)
    with pytest.raises(SourceBlockedError):
        parse_fred_csv(challenge, "DGS10")
    with pytest.raises(SourceDataError):
        parse_stooq_csv(b"No data")


def _yahoo(timestamps: list[int], closes: list[float | None], tz: str) -> bytes:
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": tz, "symbol": "X"},
                    "timestamp": timestamps,
                    "indicators": {"quote": [{"close": closes}]},
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload).encode()


def test_yahoo_chart_uses_the_exchange_calendar_date() -> None:
    # 2025-01-14 00:30 UTC is still 2025-01-13 in New York but 09:30 on the 14th
    # in Tokyo: the bar date is the exchange-local session date.
    stamp = int(datetime(2025, 1, 14, 0, 30, tzinfo=UTC).timestamp())
    tokyo = parse_yahoo_chart(_yahoo([stamp], [1.0], "Asia/Tokyo"))
    assert tokyo.dates.tolist() == [np.datetime64("2025-01-14")]
    new_york = parse_yahoo_chart(_yahoo([stamp], [1.0], "America/New_York"))
    assert new_york.dates.tolist() == [np.datetime64("2025-01-13")]


def test_yahoo_chart_drops_null_closes_and_keeps_the_last_duplicate_date() -> None:
    first = int(datetime(2025, 1, 14, 14, 30, tzinfo=UTC).timestamp())
    second = int(datetime(2025, 1, 15, 14, 30, tzinfo=UTC).timestamp())
    third = int(datetime(2025, 1, 15, 20, 59, tzinfo=UTC).timestamp())
    parsed = parse_yahoo_chart(
        _yahoo([first, second, third], [None, 10.0, 11.0], "America/New_York")
    )
    assert parsed.dates.tolist() == [np.datetime64("2025-01-15")]
    assert parsed.values.tolist() == [11.0]
    assert parsed.rows_total == 3
    assert parsed.missing_rows == 1


def test_yahoo_chart_errors_are_data_errors() -> None:
    body = json.dumps({"chart": {"result": None, "error": {"code": "Not Found"}}})
    with pytest.raises(SourceDataError):
        parse_yahoo_chart(body.encode())
    with pytest.raises(SourceDataError):
        parse_yahoo_chart(b"not json")


def test_revision_diff_reports_changed_added_and_dropped_dates() -> None:
    old = parse_fred_csv(
        b"observation_date,X\n2025-01-02,1.0\n2025-01-03,2.0\n2025-01-06,3.0\n", "X"
    )
    new = parse_fred_csv(
        b"observation_date,X\n2025-01-02,1.0\n2025-01-03,2.5\n2025-01-07,4.0\n", "X"
    )
    summary = diff_observations(old, new)
    assert summary == {
        "added": 1,
        "changed": 1,
        "dropped": 1,
        "first_changed": "2025-01-03",
    }
    assert diff_observations(old, old)["changed"] == 0


def test_series_spec_payload_is_json_serialisable() -> None:
    spec: SeriesSpec = series_by_key("DEXJPUS")
    json.dumps(spec.canonical_payload(), sort_keys=True)


def test_rows_whose_session_was_still_open_at_fetch_time_are_dropped() -> None:
    spec = series_by_key("y_gold")  # closes 17:00 New York
    payload = b"observation_date,X\n2025-03-03,1\n2025-03-04,2\n2025-03-05,3\n"
    parsed = parse_fred_csv(payload, "X")
    # Fetched 2025-03-05 15:00 UTC (10:00 EST): the 5th is still in progress.
    fetched = datetime(2025, 3, 5, 15, 0, tzinfo=UTC)
    series = build_obs_series(spec, parsed, fetched_at=fetched)
    assert series.dates.tolist() == parsed.dates[:2].tolist()
    assert series.values.tolist() == [1.0, 2.0]
    # Without a fetch time (synthetic data) nothing is dropped.
    assert build_obs_series(spec, parsed).dates.size == 3
