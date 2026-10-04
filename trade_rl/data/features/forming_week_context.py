"""Causal hourly Bollinger observations of the current UTC calendar week."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import (
    PORTABLE_FEATURE_NUMERICS_SCHEMA,
    MarketCalendarKind,
)
from trade_rl.data.features.numerics import portable_mean, portable_std
from trade_rl.data.market import MarketDataset

FORMING_WEEK_NAMES = (
    "forming_week_bb_hour_high_position",
    "forming_week_bb_hour_low_position",
)
_HOUR_NS = 3_600_000_000_000
_WEEK_NS = 168 * _HOUR_NS
_MONDAY_NS = int(np.datetime64("1970-01-05", "ns").astype(np.int64))


def with_forming_week_context(dataset: MarketDataset) -> MarketDataset:
    """Append BB20/2 positions using nineteen prior weeks and today's hour.

    Each week contains the hourly close endpoints after Monday00 through next
    Monday00, inclusive. At each completed hour, nineteen complete valid prior
    weekly closes plus the current close define the population BB. Positions
    use this hour's high and low, never the forming week's cumulative extremes.
    Monday00 remains in the old week; Monday01 starts the next week.

    A forming prefix must start at Monday01. A late, missing or inactive hour
    invalidates the rest of that week without backfill. Invalid complete weeks
    remain unusable while in the nineteen-week prior window. Differences from
    the first prior close are scaled by the positive sample maximum before
    portable mean/std, preserving adjacent-price differences near that close.
    Exact zero width is valid neutral context; no variance tolerance is used.
    """
    times = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or np.any(np.diff(times) != _HOUR_NS)
        or np.any(times % _HOUR_NS != 0)
    ):
        raise ValueError(
            "forming weekly context requires complete continuous UTC hours"
        )
    if set(FORMING_WEEK_NAMES) & set(dataset.feature_names):
        raise ValueError("forming weekly context already exists or is incomplete")

    shape = (dataset.n_bars, dataset.n_symbols, len(FORMING_WEEK_NAMES))
    values = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    usable = (
        dataset.resolved_array("information_available")
        & dataset.resolved_array("symbol_active")
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None])
    )
    ends = np.flatnonzero((times - _MONDAY_NS) % _WEEK_NS == 0)
    ends = ends[ends >= 167]
    closes = dataset.close[ends]
    complete_valid = np.asarray(
        [usable[end - 167 : end + 1].all(axis=0) for end in ends]
    )

    for week in range(18, len(ends)):
        start = int(ends[week]) + 1
        stop = min(start + 168, dataset.n_bars)
        for symbol in range(dataset.n_symbols):
            if not complete_valid[week - 18 : week + 1, symbol].all():
                continue
            prefix_valid = np.logical_and.accumulate(usable[start:stop, symbol])
            sample = np.empty(20, dtype=np.float64)
            sample[:19] = closes[week - 18 : week + 1, symbol]
            for offset in np.flatnonzero(prefix_valid):
                row = start + int(offset)
                sample[-1] = dataset.close[row, symbol]
                reference = float(sample[0])
                scale = float(sample.max())
                deviations = (sample - reference) / scale
                mean_delta = portable_mean(deviations)
                width = 2 * portable_std(deviations)
                if width > 0.0:
                    values[row, symbol] = (
                        (
                            (float(dataset.high[row, symbol]) - reference) / scale
                            - mean_delta
                        )
                        / width,
                        (
                            (float(dataset.low[row, symbol]) - reference) / scale
                            - mean_delta
                        )
                        / width,
                    )
                mask[row, symbol] = True

    age = np.where(mask, 0.0, 168.0)
    definition = {
        "transform": "forming_week_context_v1",
        "source_dataset": {
            "dataset_id": dataset.dataset_id,
            "identity_payload": json.loads(dataset.identity_payload_json)
            if dataset.identity_payload_json is not None
            else None,
        },
        "week_boundary": "Monday_00:00_UTC_close_belongs_to_ending_week",
        "constituent_close_count": 168,
        "forming_prefix_start": "Monday_01:00_UTC",
        "late_constituents": "invalidate_week_no_backfill",
        "required_complete_prior_weeks": 19,
        "bb_sample": "nineteen_prior_weekly_closes_plus_current_hour_close",
        "bb_periods": 20,
        "bb_population_std_multiplier": 2,
        "observed_wicks": "current_hour_high_low_not_cumulative_week_extremes",
        "numerics_schema": PORTABLE_FEATURE_NUMERICS_SCHEMA,
        "price_scaling": "subtract_first_prior_close_then_divide_by_positive_sample_maximum",
        "zero_variance_band_positions": [0, 0],
        "absolute_variance_epsilon": None,
        "feature_names": list(FORMING_WEEK_NAMES),
        "staleness_denominator_hours": 168,
    }
    augmented = replace(
        dataset,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, values), axis=2),
        feature_names=dataset.feature_names + FORMING_WEEK_NAMES,
        feature_available=np.concatenate((dataset.feature_available, mask), axis=2),
        feature_staleness=np.concatenate(
            (
                dataset.resolved_array("feature_staleness"),
                (age / 168).astype(np.float32),
            ),
            axis=2,
        ),
        feature_staleness_hours=np.concatenate(
            (dataset.resolved_array("feature_staleness_hours"), age), axis=2
        ),
        feature_missing_reason=np.concatenate(
            (dataset.resolved_array("feature_missing_reason"), (~mask).astype(np.int8)),
            axis=2,
        ),
        feature_config_digest=content_digest(definition),
    )
    return augmented.with_content_identity(definition)
