"""Completed UTC calendar-week Bollinger and plotted Ichimoku context."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.features.numerics import portable_mean, portable_std
from trade_rl.data.market import MarketDataset

WEEKLY_NAMES = (
    "weekly_bb_position",
    "weekly_bb_high_position",
    "weekly_bb_low_position",
    "weekly_tenkan_distance",
    "weekly_kijun_distance",
    "weekly_cloud_upper_distance",
    "weekly_cloud_lower_distance",
)
_HOUR_NS = 3_600_000_000_000
_WEEK_NS = 168 * _HOUR_NS
_MONDAY_NS = int(np.datetime64("1970-01-05", "ns").astype(np.int64))


def with_weekly_context(dataset: MarketDataset) -> MarketDataset:
    """Append fixed BB20/2 and Ichimoku9/26/52 with a 26-week cloud shift.

    Input rows are close endpoints of complete, continuous UTC hours. A week
    contains the 168 closes after Monday 00:00 through next Monday 00:00.
    Any unavailable, late-at-own-close, or inactive constituent invalidates its
    week. No retrospective late-data repair is performed. All 78 required weeks
    must be usable; partial boundary weeks are excluded. Values update only at
    a completed week and never use the currently forming week's prices.
    """
    times = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or np.any(np.diff(times) != _HOUR_NS)
        or np.any(times % _HOUR_NS != 0)
    ):
        raise ValueError("weekly context requires complete continuous UTC hours")
    if set(WEEKLY_NAMES) & set(dataset.feature_names):
        raise ValueError("weekly context already exists or is incomplete")

    ends = np.flatnonzero((times - _MONDAY_NS) % _WEEK_NS == 0)
    ends = ends[ends >= 167]
    shape = (dataset.n_bars, dataset.n_symbols, len(WEEKLY_NAMES))
    values = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    age = np.full(shape, 168.0, dtype=np.float64)
    usable = (
        dataset.resolved_array("information_available")
        & dataset.resolved_array("symbol_active")
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None])
    )
    highs = np.asarray([dataset.high[end - 167 : end + 1].max(axis=0) for end in ends])
    lows = np.asarray([dataset.low[end - 167 : end + 1].min(axis=0) for end in ends])
    closes = dataset.close[ends]
    valid = np.asarray([usable[end - 167 : end + 1].all(axis=0) for end in ends])

    for week in range(77, len(ends)):
        row = int(ends[week])
        stop = min(row + 168, dataset.n_bars)
        for symbol in range(dataset.n_symbols):
            if not valid[week - 77 : week + 1, symbol].all():
                continue

            def midpoint(end: int, length: int) -> float:
                start = end - length + 1
                return float(
                    (
                        highs[start : end + 1, symbol].max()
                        + lows[start : end + 1, symbol].min()
                    )
                    / 2
                )

            mean = portable_mean(closes[week - 19 : week + 1, symbol])
            width = 2 * portable_std(closes[week - 19 : week + 1, symbol])
            close = float(closes[week, symbol])
            bands = (
                tuple(
                    (float(price) - mean) / width
                    for price in (close, highs[week, symbol], lows[week, symbol])
                )
                if width > 1e-12
                else (0.0, 0.0, 0.0)
            )
            shifted = week - 26
            span_a = (midpoint(shifted, 9) + midpoint(shifted, 26)) / 2
            span_b = midpoint(shifted, 52)
            context = (
                *bands,
                (close - midpoint(week, 9)) / close,
                (close - midpoint(week, 26)) / close,
                close / max(span_a, span_b) - 1,
                close / min(span_a, span_b) - 1,
            )
            values[row:stop, symbol] = context
            current_usable = usable[row:stop, symbol, None]
            mask[row:stop, symbol] = current_usable
            age[row:stop, symbol] = np.where(
                current_usable, np.arange(stop - row)[:, None], 168.0
            )
    values[~mask] = 0.0
    definition = {
        "schema": "completed_weekly_context_v1",
        "transform": "completed_weekly_context_v1",
        "source_dataset": {
            "dataset_id": dataset.dataset_id,
            "identity_payload": json.loads(dataset.identity_payload_json)
            if dataset.identity_payload_json is not None
            else None,
        },
        "week_boundary": "Monday_00:00_UTC",
        "constituent_close_count": 168,
        "late_constituents": "invalidate_no_backfill",
        "bb_periods": 20,
        "bb_population_std_multiplier": 2,
        "ichimoku_periods": [9, 26, 52],
        "plotted_cloud_lag_weeks": 26,
        "required_complete_weeks": 78,
        "feature_names": list(WEEKLY_NAMES),
        "zero_variance_band_positions": 0,
        "staleness_denominator_hours": 168,
    }
    augmented = replace(
        dataset,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, values), axis=2),
        feature_names=dataset.feature_names + WEEKLY_NAMES,
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
