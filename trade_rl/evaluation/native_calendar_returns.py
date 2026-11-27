"""Canonical return payload for complete calendar-month spot replay."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_execution_clock import (
    calendar_month_clock_payload,
    frequency_payload,
)
from trade_rl.evaluation.series import CalendarMonthReturnSeries

NATIVE_CALENDAR_MONTH_RETURN_SCHEMA = "native_calendar_month_return_series_v1"


def calendar_month_returns_for_replay(
    dataset: MarketDataset, values: tuple[float, ...], *, start_index: int
) -> CalendarMonthReturnSeries:
    if (
        type(dataset) is not MarketDataset
        or not dataset.native_calendar_month
        or not dataset.identity_verified
    ):
        raise ValueError("expected a verified calendar month Dataset")
    if type(start_index) is not int or not 0 <= start_index < dataset.n_bars:
        raise ValueError("invalid calendar month return start index")
    stop_index = start_index + len(values)
    if not values or stop_index >= dataset.n_bars:
        raise ValueError("calendar month returns exceed the Dataset interval")
    return CalendarMonthReturnSeries.from_closes(
        values, dataset.timestamps[start_index + 1 : stop_index + 1]
    )


def native_calendar_month_return_payload(
    dataset: MarketDataset, returns: CalendarMonthReturnSeries, *, start_index: int
) -> dict[str, object]:
    if type(
        returns
    ) is not CalendarMonthReturnSeries or returns != calendar_month_returns_for_replay(
        dataset, returns.values, start_index=start_index
    ):
        raise ValueError("calendar month returns do not match the Dataset interval")
    assert dataset.native_clock is not None
    assert type(returns.periods_per_year) is Fraction
    return {
        "schema_version": NATIVE_CALENDAR_MONTH_RETURN_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "start_index": start_index,
        "stop_index": start_index + len(returns.values),
        "kind": returns.kind.value,
        "clock": calendar_month_clock_payload(dataset.native_clock),
        "closes_ns": list(returns.closes_ns),
        "periods_per_year": frequency_payload(returns.periods_per_year),
        "elapsed_years": frequency_payload(returns.exact_elapsed_years),
        "values": list(returns.values),
    }


def load_native_calendar_month_return_payload(
    dataset: MarketDataset, payload: Mapping[str, object]
) -> CalendarMonthReturnSeries:
    if (
        type(payload) is not dict
        or payload.get("schema_version") != NATIVE_CALENDAR_MONTH_RETURN_SCHEMA
    ):
        raise ValueError("expected a calendar month return payload")
    values, start = payload.get("values"), payload.get("start_index")
    if (
        type(values) is not list
        or any(type(x) is not float for x in values)
        or type(start) is not int
    ):
        raise ValueError("invalid calendar month return payload")
    returns = calendar_month_returns_for_replay(
        dataset, tuple(values), start_index=start
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(
        native_calendar_month_return_payload(dataset, returns, start_index=start)
    ):
        raise ValueError(
            "calendar month return payload does not match the Dataset interval"
        )
    return returns
