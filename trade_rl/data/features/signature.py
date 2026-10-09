"""Opt-in causal rolling piecewise-linear path signatures.

This is a feature representation, not a trading/alpha claim. Only the close
observations at completed, on-time bars enter the path. High/low intrabar
ordering, exchange queue position, and raw future funding are never inferred.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset

_SCHEMA = "rolling_path_signature_v1"
_MAX_DEPTH = 3
_MAX_WINDOW_BARS = 512


def _segment_signature(increments: np.ndarray, depth: int) -> np.ndarray:
    """Truncated Chen product of straight-segment iterated integrals.

    For a displacement dx, level k is dx**tensor(k)/k!. Compose segments
    in chronological order. Index ordering is the natural tensor word order.
    """
    dimension = int(increments.shape[1])
    levels = [np.array([1.0], dtype=np.float64)]
    levels.extend(
        np.zeros(dimension**level, dtype=np.float64) for level in range(1, depth + 1)
    )
    for delta in increments:
        segment = [np.array([1.0], dtype=np.float64), delta]
        for level in range(2, depth + 1):
            segment.append(np.kron(segment[-1], delta) / float(level))
        # Descending order leaves lower levels at their previous-path values.
        for level in range(depth, 0, -1):
            for segment_level in range(1, level + 1):
                levels[level] += np.kron(
                    levels[level - segment_level], segment[segment_level]
                )
    return np.concatenate(levels[1:])


def with_path_signatures(
    dataset: MarketDataset,
    *,
    window_bars: int = 24,
    depth: int = 2,
    include_volume: bool = False,
) -> MarketDataset:
    """Append a finite rolling Signature computed at each completed bar close.

    Path channels are normalized bar index (0..1), log(close), and optionally
    log(volume); all values are translated to their first-window observation.
    The log-volume channel requires strictly positive native volume in every
    row of the window. Time is *bar index*, not elapsed clock time, including
    for session datasets. This is piecewise-linear interpolation of observed
    bar-close points, never an invented path through OHLC high/low.

    The window includes the decision bar t and w-1 preceding completed bars.
    Its value is usable at t only if every constituent row was already
    available by its own close. Historical gaps invalidate the entire window;
    missing outputs are zero with an explicit unavailable mask. All features
    are opt-in: no canonical dataset, model, or frozen study changes by default.
    """
    if (
        isinstance(window_bars, bool)
        or not isinstance(window_bars, int)
        or not 3 <= window_bars <= min(dataset.n_bars, _MAX_WINDOW_BARS)
    ):
        raise ValueError("window_bars must be between 3 and min(n_bars, 512)")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or not 1 <= depth <= _MAX_DEPTH
    ):
        raise ValueError("depth must be an integer from 1 through 3")
    if not isinstance(include_volume, bool):
        raise ValueError("include_volume must be boolean")

    channel_labels = ("t", "p", "v") if include_volume else ("t", "p")
    dimension = len(channel_labels)
    prefix = f"path_sig_v1_w{window_bars}_d{depth}_{''.join(channel_labels)}"
    names = tuple(
        f"{prefix}_{''.join(word)}"
        for level in range(1, depth + 1)
        for word in product(channel_labels, repeat=level)
    )
    if set(names) & set(dataset.feature_names):
        raise ValueError("path signature feature names already exist")

    values = np.zeros(
        (dataset.n_bars, dataset.n_symbols, len(names)), dtype=np.float32
    )
    available = np.zeros_like(values, dtype=np.bool_)
    source_usable = (
        dataset.resolved_array("information_available")
        & dataset.resolved_array("symbol_active")
        & dataset.tradable
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None])
        & np.isfinite(dataset.close)
        & (dataset.close > 0.0)
    )
    if include_volume:
        source_usable &= np.isfinite(dataset.volume) & (dataset.volume > 0.0)

    step = 1.0 / float(window_bars - 1)
    for symbol_index in range(dataset.n_symbols):
        good = source_usable[:, symbol_index]
        bad_prefix = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(~good, dtype=np.int64))
        )
        log_price = np.zeros(dataset.n_bars, dtype=np.float64)
        log_volume = np.zeros(dataset.n_bars, dtype=np.float64)
        for row in np.flatnonzero(good):
            log_price[row] = math.log(float(dataset.close[row, symbol_index]))
            if include_volume:
                log_volume[row] = math.log(float(dataset.volume[row, symbol_index]))

        for end in range(window_bars - 1, dataset.n_bars):
            start = end - window_bars + 1
            if bad_prefix[end + 1] != bad_prefix[start]:
                continue
            increments = np.empty((window_bars - 1, dimension), dtype=np.float64)
            increments[:, 0] = step
            increments[:, 1] = np.diff(log_price[start : end + 1])
            if include_volume:
                increments[:, 2] = np.diff(log_volume[start : end + 1])
            signature = _segment_signature(increments, depth)
            if not np.isfinite(signature).all():
                continue
            cast_signature = signature.astype(np.float32)
            if not np.isfinite(cast_signature).all():
                continue
            values[end, symbol_index] = cast_signature
            available[end, symbol_index] = True

    definition: dict[str, object] = {
        "schema": _SCHEMA,
        "source_dataset_id": dataset.dataset_id,
        "window_bars": window_bars,
        "depth": depth,
        "channels": channel_labels,
        "path": "piecewise_linear_completed_bar_close",
        "time": "normalized_bar_index",
        "log_channels": "unscaled_natural_log_increments",
        "availability": "every_constituent_row_available_by_own_close",
    }
    missing = ~available
    augmented = replace(
        dataset,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, values), axis=2),
        feature_names=dataset.feature_names + names,
        feature_available=np.concatenate(
            (dataset.feature_available, available), axis=2
        ),
        feature_staleness=np.concatenate(
            (dataset.resolved_array("feature_staleness"), missing.astype(np.float32)),
            axis=2,
        ),
        feature_staleness_hours=np.concatenate(
            (
                dataset.resolved_array("feature_staleness_hours"),
                np.where(missing, window_bars * dataset.bar_hours, 0.0),
            ),
            axis=2,
        ),
        feature_missing_reason=np.concatenate(
            (
                dataset.resolved_array("feature_missing_reason"),
                missing.astype(np.int16),
            ),
            axis=2,
        ),
        feature_config_digest=content_digest(definition),
    )
    return augmented.with_content_identity(definition)


__all__ = ["with_path_signatures"]
