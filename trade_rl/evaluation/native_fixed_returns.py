"""Canonical native fixed-grid return payload, separate from legacy run artifacts.

The content binds the exact dataset and return interval. It is not evidence
that a strategy executed those returns or that market data is authentic.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_execution_clock import (
    execution_clock_payload,
    frequency_payload,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

NATIVE_FIXED_RETURN_SCHEMA = "native_fixed_return_series_v1"


def native_fixed_return_payload(
    dataset: MarketDataset, returns: ReturnSeries, *, start_index: int
) -> dict[str, object]:
    """Bind each close-to-close base-bar return to a complete native interval."""
    if (
        type(dataset) is not MarketDataset
        or dataset.native_clock is None
        or dataset.native_calendar_month
        or not dataset.identity_verified
        or not isinstance(dataset.periods_per_year, Fraction)
    ):
        raise ValueError("expected a verified native fixed Dataset")
    if (
        type(returns) is not ReturnSeries
        or returns.kind is not ReturnKind.BASE_BAR
        or type(returns.periods_per_year) is not Fraction
        or returns.periods_per_year != dataset.periods_per_year
        or returns.elapsed_years is not None
    ):
        raise ValueError("return series does not match the native fixed frequency")
    if type(start_index) is not int or not 0 <= start_index < dataset.n_bars:
        raise ValueError("native return start index is invalid")
    stop_index = start_index + len(returns.values)
    if stop_index >= dataset.n_bars:
        raise ValueError("native return series exceeds the Dataset interval")
    elapsed_years = len(returns.values) / dataset.periods_per_year
    return {
        "schema_version": NATIVE_FIXED_RETURN_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "start_index": start_index,
        "stop_index": stop_index,
        "kind": returns.kind.value,
        "clock": execution_clock_payload(dataset.native_clock),
        "periods_per_year": frequency_payload(dataset.periods_per_year),
        "elapsed_years": frequency_payload(elapsed_years),
        "values": list(returns.values),
    }


def load_native_fixed_return_payload(
    dataset: MarketDataset, payload: Mapping[str, object]
) -> ReturnSeries:
    """Validate exact schema and Dataset binding before returning the series."""
    if (
        type(payload) is not dict
        or payload.get("schema_version") != NATIVE_FIXED_RETURN_SCHEMA
    ):
        raise ValueError("expected a native fixed return payload")
    if type(dataset) is not MarketDataset:
        raise ValueError("expected a verified native fixed Dataset")
    raw = payload.get("values")
    start = payload.get("start_index")
    frequency = dataset.periods_per_year
    if (
        type(raw) is not list
        or any(type(item) is not float for item in raw)
        or type(start) is not int
        or not isinstance(frequency, Fraction)
    ):
        raise ValueError("invalid native fixed return payload")
    returns = ReturnSeries(tuple(raw), ReturnKind.BASE_BAR, frequency)
    expected = native_fixed_return_payload(dataset, returns, start_index=start)
    if canonical_json_bytes(payload) != canonical_json_bytes(expected):
        raise ValueError(
            "native fixed return payload does not match its Dataset interval"
        )
    return returns


__all__ = [
    "NATIVE_FIXED_RETURN_SCHEMA",
    "load_native_fixed_return_payload",
    "native_fixed_return_payload",
]
