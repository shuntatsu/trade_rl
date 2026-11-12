"""Strict parsers for FRED CSV, Stooq CSV and Yahoo chart JSON payloads."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import numpy as np

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MISSING = frozenset({"", "."})


class SourceDataError(ValueError):
    """The payload is not a valid series for this source."""


class SourceBlockedError(SourceDataError):
    """The source answered with a bot check instead of data. It is not bypassed."""


@dataclass(frozen=True, slots=True)
class ParsedSeries:
    """Valid observations of one series plus bookkeeping about dropped rows."""

    dates: np.ndarray  # datetime64[D], strictly increasing
    values: np.ndarray  # float64, finite
    rows_total: int
    missing_rows: int
    last_row_date: date | None


def _raise_if_markup(payload: bytes) -> None:
    head = payload.lstrip()[:512].lower()
    if head.startswith((b"<!doctype html", b"<html")):
        lowered = payload.lower()
        if b"__verify" in lowered or b"requires javascript" in lowered:
            raise SourceBlockedError(
                "source returned a JavaScript bot-detection challenge"
            )
        raise SourceDataError("source returned HTML instead of data")


def _parse_date(text: str) -> date:
    if not _ISO_DATE.match(text):
        raise SourceDataError(f"date is not YYYY-MM-DD: {text!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise SourceDataError(f"invalid calendar date: {text!r}") from error


def _parse_value(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise SourceDataError(f"value is not numeric: {text!r}") from error
    if not math.isfinite(value):
        raise SourceDataError(f"value is not finite: {text!r}")
    return value


def _finish(
    days: list[date],
    values: list[float],
    *,
    rows_total: int,
    last: date | None,
    missing_rows: int | None = None,
) -> ParsedSeries:
    for earlier, later in zip(days, days[1:], strict=False):
        if later <= earlier:
            raise SourceDataError("observation dates must be strictly increasing")
    return ParsedSeries(
        dates=np.asarray(
            [np.datetime64(day, "D") for day in days], dtype="datetime64[D]"
        ),
        values=np.asarray(values, dtype=np.float64),
        rows_total=rows_total,
        missing_rows=rows_total - len(days) if missing_rows is None else missing_rows,
        last_row_date=last,
    )


def _rows(payload: bytes) -> list[list[str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SourceDataError("payload is not UTF-8") from error
    return [row for row in csv.reader(io.StringIO(text)) if row]


def parse_fred_csv(payload: bytes, series_id: str) -> ParsedSeries:
    _raise_if_markup(payload)
    rows = _rows(payload)
    if not rows:
        raise SourceDataError("empty FRED payload")
    header = rows[0]
    if len(header) != 2 or header[0] not in {"observation_date", "DATE"}:
        raise SourceDataError("unexpected FRED header")
    if header[1] != series_id:
        raise SourceDataError(f"FRED payload is for {header[1]!r}, not {series_id!r}")
    days: list[date] = []
    values: list[float] = []
    last: date | None = None
    for row in rows[1:]:
        if len(row) != 2:
            raise SourceDataError("FRED row must have two columns")
        day = _parse_date(row[0])
        if last is not None and day <= last:
            raise SourceDataError("observation dates must be strictly increasing")
        last = day
        if row[1].strip() in _MISSING:
            continue
        days.append(day)
        values.append(_parse_value(row[1]))
    return _finish(days, values, rows_total=len(rows) - 1, last=last)


def parse_stooq_csv(payload: bytes, series_id: str = "") -> ParsedSeries:
    del series_id
    _raise_if_markup(payload)
    rows = _rows(payload)
    if not rows or rows[0][0] != "Date" or "Close" not in rows[0]:
        raise SourceDataError("unexpected Stooq payload")
    close_index = rows[0].index("Close")
    days: list[date] = []
    values: list[float] = []
    last: date | None = None
    for row in rows[1:]:
        if len(row) <= close_index:
            raise SourceDataError("Stooq row is shorter than its header")
        day = _parse_date(row[0])
        if last is not None and day <= last:
            raise SourceDataError("observation dates must be strictly increasing")
        last = day
        if row[close_index].strip() in _MISSING:
            continue
        days.append(day)
        values.append(_parse_value(row[close_index]))
    return _finish(days, values, rows_total=len(rows) - 1, last=last)


def parse_yahoo_chart(payload: bytes, series_id: str = "") -> ParsedSeries:
    del series_id
    _raise_if_markup(payload)
    try:
        document = json.loads(payload)
        chart = document["chart"]
        if chart.get("error") is not None or not chart.get("result"):
            raise SourceDataError(f"Yahoo chart error: {chart.get('error')!r}")
        result = chart["result"][0]
        timezone = ZoneInfo(result["meta"]["exchangeTimezoneName"])
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except SourceDataError:
        raise
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise SourceDataError(f"unexpected Yahoo chart payload: {error}") from error
    if len(stamps) != len(closes):
        raise SourceDataError("Yahoo timestamps and closes differ in length")
    by_day: dict[date, float] = {}
    null_closes = 0
    last: date | None = None
    previous_stamp: int | None = None
    for stamp, close in zip(stamps, closes, strict=True):
        if not isinstance(stamp, int) or isinstance(stamp, bool):
            raise SourceDataError("Yahoo timestamp must be an integer")
        if previous_stamp is not None and stamp <= previous_stamp:
            raise SourceDataError("Yahoo timestamps must be strictly increasing")
        previous_stamp = stamp
        # A daily bar is stamped with its session OPEN in exchange-local time, so
        # the exchange-local calendar date is the bar's own session date (never an
        # earlier one). Verified on a live sample: see the committed timestamp
        # fixture and its test.
        day = datetime.fromtimestamp(stamp, UTC).astimezone(timezone).date()
        last = day
        if close is None:
            null_closes += 1
            continue
        if isinstance(close, bool) or not isinstance(close, (int, float)):
            raise SourceDataError("Yahoo close must be numeric")
        if not math.isfinite(float(close)):
            raise SourceDataError("Yahoo close must be finite")
        by_day[day] = float(close)  # a later bar on the same date replaces it
    days = sorted(by_day)
    return _finish(
        days,
        [by_day[day] for day in days],
        rows_total=len(stamps),
        last=last,
        missing_rows=null_closes,
    )


Parser = Callable[[bytes, str], ParsedSeries]

PARSERS: dict[str, tuple[str, Parser]] = {
    "fred": ("fred_csv_v1", parse_fred_csv),
    "stooq": ("stooq_csv_v1", parse_stooq_csv),
    "yahoo": ("yahoo_chart_v1", parse_yahoo_chart),
}


def diff_observations(
    old: ParsedSeries, new: ParsedSeries, *, rel_tol: float = 1e-9
) -> dict[str, object]:
    """Summarise how a refetched series differs from the previous vintage."""

    old_map = dict(zip(old.dates.tolist(), old.values.tolist(), strict=True))
    new_map = dict(zip(new.dates.tolist(), new.values.tolist(), strict=True))
    changed = sorted(
        day
        for day in old_map.keys() & new_map.keys()
        if not math.isclose(old_map[day], new_map[day], rel_tol=rel_tol, abs_tol=1e-12)
    )
    return {
        "added": len(new_map.keys() - old_map.keys()),
        "changed": len(changed),
        "dropped": len(old_map.keys() - new_map.keys()),
        "first_changed": changed[0].isoformat() if changed else None,
    }


__all__ = [
    "PARSERS",
    "ParsedSeries",
    "SourceBlockedError",
    "SourceDataError",
    "diff_observations",
    "parse_fred_csv",
    "parse_stooq_csv",
    "parse_yahoo_chart",
]
