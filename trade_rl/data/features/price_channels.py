"""Causal completed-candle price channels for directional trading."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset

CHANNEL_NAMES = (
    "channel_entry_upper",
    "channel_entry_lower",
    "channel_exit_upper",
    "channel_exit_lower",
)


def with_price_channels(
    dataset: MarketDataset, *, entry_bars: int = 480, exit_bars: int = 240
) -> MarketDataset:
    """Append distance from prior candle extrema; never include today's high.

    Unavailable source rows invalidate the whole required window. Existing
    prices, economics, features and availability remain unchanged.
    """
    if (
        isinstance(entry_bars, bool)
        or not isinstance(entry_bars, int)
        or isinstance(exit_bars, bool)
        or not isinstance(exit_bars, int)
        or not 0 < exit_bars < entry_bars < dataset.n_bars
    ):
        raise ValueError("channel lengths must satisfy 0 < exit < entry < n_bars")
    if set(CHANNEL_NAMES) & set(dataset.feature_names):
        raise ValueError("price channels already exist")
    values = np.zeros((dataset.n_bars, dataset.n_symbols, 4), dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.bool_)
    usable = (
        dataset.resolved_array("information_available")
        & dataset.resolved_array("symbol_active")
        & dataset.tradable
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None])
    )
    for offset, window in ((0, entry_bars), (2, exit_bars)):
        upper = np.lib.stride_tricks.sliding_window_view(
            dataset.high[:-1], window, axis=0
        ).max(axis=-1)
        lower = np.lib.stride_tricks.sliding_window_view(
            dataset.low[:-1], window, axis=0
        ).min(axis=-1)
        valid = (
            np.lib.stride_tricks.sliding_window_view(usable[:-1], window, axis=0).all(
                axis=-1
            )
            & usable[window:]
        )
        for column, boundary in ((offset, upper), (offset + 1, lower)):
            values[window:, :, column] = np.where(
                valid, dataset.close[window:] / boundary - 1.0, 0.0
            )
            mask[window:, :, column] = valid
    definition = {
        "schema": "causal_price_channels_v1",
        "source_dataset_id": dataset.dataset_id,
        "entry_bars": entry_bars,
        "exit_bars": exit_bars,
        "current_bar_in_extrema": False,
    }
    augmented = replace(
        dataset,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, values), axis=2),
        feature_names=dataset.feature_names + CHANNEL_NAMES,
        feature_available=np.concatenate((dataset.feature_available, mask), axis=2),
        feature_staleness=np.concatenate(
            (dataset.resolved_array("feature_staleness"), (~mask).astype(np.float32)),
            axis=2,
        ),
        feature_staleness_hours=np.concatenate(
            (
                dataset.resolved_array("feature_staleness_hours"),
                np.where(mask, 0.0, float(entry_bars)),
            ),
            axis=2,
        ),
        feature_missing_reason=np.concatenate(
            (dataset.resolved_array("feature_missing_reason"), (~mask).astype(np.int8)),
            axis=2,
        ),
        feature_config_digest=content_digest(definition),
    )
    return augmented.with_content_identity(definition)
