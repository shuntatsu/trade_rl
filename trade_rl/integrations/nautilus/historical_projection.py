"""Projection of canonical market rows into deterministic historical source bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketCalendarKind, MarketDataset


@dataclass(frozen=True, slots=True)
class HistoricalSourceBar:
    """One single-symbol continuous-market bar at integer nanosecond boundaries."""

    open_ns: int
    close_ns: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    mark_price: float
    index_price: float


def project_historical_source_bar(
    market: MarketDataset,
    *,
    processing_index: int,
) -> HistoricalSourceBar:
    """Project one source row without introducing a synthetic timing convention.

    ``MarketDataset.timestamps`` are bar-close timestamps. For the continuous
    single-symbol contract used by the first Nautilus migration slice, the bar
    open is therefore exactly one fixed dataset cadence before the close.
    """

    if len(market.symbols) != 1:
        raise ValueError(
            "historical source-bar projection requires a single-symbol dataset"
        )
    if market.calendar_kind is not MarketCalendarKind.CONTINUOUS:
        raise ValueError(
            "historical source-bar projection requires continuous market data"
        )
    if not isinstance(processing_index, int) or isinstance(processing_index, bool):
        raise TypeError("processing_index must be an integer")
    if processing_index < 0 or processing_index >= len(market.timestamps):
        raise IndexError("processing_index is outside the market dataset")

    timestamps_ns = market.timestamps.astype("datetime64[ns]").astype(np.int64)
    cadence_ns = int(timestamps_ns[1] - timestamps_ns[0])
    close_ns = int(timestamps_ns[processing_index])

    return HistoricalSourceBar(
        open_ns=close_ns - cadence_ns,
        close_ns=close_ns,
        open_price=float(market.open[processing_index, 0]),
        high_price=float(market.high[processing_index, 0]),
        low_price=float(market.low[processing_index, 0]),
        close_price=float(market.close[processing_index, 0]),
        mark_price=float(market.mark_price[processing_index, 0]),
        index_price=float(market.index_price[processing_index, 0]),
    )
