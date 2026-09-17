"""Identity-bound spot/perpetual market inputs for structural carry research."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance.dataset import _parse_kline_rows
from trade_rl.integrations.binance.types import _aware_utc
from trade_rl.integrations.binance.vision import _epoch_ms, _normalize_epoch_ms


def parse_carry_spot_rows(
    rows: Sequence[Sequence[object]],
    *,
    start_time: datetime,
    end_time: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
) -> RawMarketSeries:
    """Use stale marks only for whole bins inside independently declared halts.

    Every partially or fully halted bin is nontradable and has zero capacity.
    Published prices are preserved; inserted marks use only the preceding close.
    The caller must bind the halt calendar and its external evidence before any
    economic replay. Missing rows never imply a halt automatically.
    """
    hour = 3_600_000
    start = _epoch_ms(_aware_utc(start_time, field="start_time"))
    end = _epoch_ms(_aware_utc(end_time, field="end_time"))
    if start % hour or end % hour or end - start < 2 * hour:
        raise ValueError("carry spot range must contain aligned hourly bars")
    halts = [
        (
            _epoch_ms(_aware_utc(lo, field="halt_start")),
            _epoch_ms(_aware_utc(hi, field="halt_end")),
        )
        for lo, hi in halt_intervals
    ]
    if any(lo >= hi for lo, hi in halts) or any(
        lo < previous_hi for (_, previous_hi), (lo, _) in zip(halts, halts[1:])
    ):
        raise ValueError("halt intervals must be ordered, positive and nonoverlapping")
    by_time: dict[int, Sequence[object]] = {}
    previous = None
    for row in rows:
        if len(row) < 8:
            raise ValueError("carry spot row must contain at least eight fields")
        opened = _normalize_epoch_ms(row[0])
        if not start <= opened < end:
            continue
        if opened % hour or (previous is not None and opened <= previous):
            raise ValueError("carry spot rows must be unique, ordered and hourly")
        by_time[opened] = row
        previous = opened
    completed: list[Sequence[object]] = []
    tradable = []
    for opened in range(start, end, hour):
        closed = opened + hour
        halted = any(lo < closed and hi > opened for lo, hi in halts)
        source_row = by_time.get(opened)
        if source_row is None:
            if not completed or not any(
                lo <= opened and closed <= hi for lo, hi in halts
            ):
                raise ValueError("unexplained carry spot gap or missing preceding mark")
            mark = completed[-1][4]
            source_row = [opened, mark, mark, mark, mark, 0, closed - 1, 0]
        completed.append(source_row)
        tradable.append(not halted)
    timestamps, opened_prices, high, low, close, volume = _parse_kline_rows(
        completed, interval_ms=hour, start_ms=start, end_ms=end
    )
    if np.any(volume < 0):
        raise ValueError("published carry spot volume must be non-negative")
    mask = np.asarray(tradable, dtype=np.bool_)
    return RawMarketSeries(
        timestamps=timestamps,
        open=opened_prices,
        high=high,
        low=low,
        close=close,
        volume=np.where(mask, volume, 0.0),
        funding_rate=np.zeros(len(timestamps)),
        funding_available=np.zeros(len(timestamps), dtype=np.bool_),
        funding_event_count=np.zeros(len(timestamps), dtype=np.int32),
        tradable=mask,
    )


def assemble_carry_dataset(
    pairs: Mapping[str, tuple[RawMarketSeries, RawMarketSeries]],
    *,
    cost_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
) -> MarketDataset:
    """Assemble the fixed BTC/ETH hourly development contract, without forecasts.

    Raw volume is Binance quote turnover, not base units. Published funding
    events remain on their original completed-bar clock. First/last rows are
    source boundaries; interior funding gaps above nine hourly bins fail closed
    (eight-hour events may straddle bins due to timestamp precision).
    """
    if not pairs or not set(pairs).issubset({"BTCUSDT", "ETHUSDT"}):
        raise ValueError("carry source requires BTCUSDT and/or ETHUSDT pairs")
    for value in (cost_multiplier, capacity_multiplier):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("carry cost/capacity multipliers must be positive")
    if capacity_multiplier > 1.0:
        raise ValueError("carry capacity multiplier may only reduce participation")
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("carry source requires exactly two legs per pair")
    symbols = tuple(sorted(pairs))
    legs = [leg for symbol in symbols for leg in pairs[symbol]]
    timestamps = legs[0].timestamps
    if len(timestamps) < 3 or np.any(np.diff(timestamps) != np.timedelta64(1, "h")):
        raise ValueError("carry source clock must be complete and hourly")
    for index, leg in enumerate(legs):
        if not np.array_equal(leg.timestamps, timestamps):
            raise ValueError("carry source clock differs between legs")
        if not np.array_equal(np.asarray(leg.available_at), timestamps):
            raise ValueError("carry source must be available at its declared close")
        counts = np.asarray(leg.funding_event_count)
        if index % 2 == 0:
            if np.any(counts) or np.any(leg.funding_rate):
                raise ValueError("spot source must not contain funding")
        else:
            events = np.flatnonzero(counts)
            if (
                not len(events)
                or np.any(counts > 1)
                or events[0] > 8
                or len(timestamps) - 2 - events[-1] > 8
                or np.any(np.diff(events) > 9)
                or np.any(leg.funding_rate[counts == 0] != 0)
            ):
                raise ValueError("perpetual source has incomplete funding coverage")

    shape = (len(timestamps), len(legs))

    def stacked(name: str) -> np.ndarray:
        return np.column_stack([getattr(leg, name) for leg in legs])

    fee = np.tile([0.001, 0.0005], len(symbols)) * cost_multiplier
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=tuple(
            f"{symbol}:{market}" for symbol in symbols for market in ("spot", "perp")
        ),
        timestamps=timestamps,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((len(timestamps), 1), dtype=np.float32),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("carry_no_forecast",),
        global_feature_names=("carry_no_forecast",),
        open=stacked("open"),
        high=stacked("high"),
        low=stacked("low"),
        close=stacked("close"),
        volume=stacked("volume"),
        volume_units=tuple(VolumeUnit.QUOTE_NOTIONAL for _ in legs),
        funding_rate=stacked("funding_rate"),
        funding_event_count=stacked("funding_event_count"),
        funding_due=stacked("funding_available"),
        tradable=stacked("tradable"),
        periods_per_year=8760,
        fee_rate=np.zeros(shape),
        maker_fee_rate=np.broadcast_to(fee, shape),
        taker_fee_rate=np.broadcast_to(fee, shape),
        spread_rate=np.full(shape, 0.0005 * cost_multiplier),
        lot_size=np.full(shape, 0.001),
        minimum_notional=np.full(shape, 10.0),
        max_participation_rate=np.full(shape, 0.01 * capacity_multiplier),
        borrow_available=np.broadcast_to(np.tile([False, True], len(symbols)), shape),
        mark_price=stacked("close"),
    )
    return dataset.with_content_identity(
        {
            "source_contract": "funding_carry_development_v1",
            "cost_multiplier": cost_multiplier,
            "capacity_multiplier": capacity_multiplier,
            "mark_price_limitation": "perpetual_bar_close_proxy",
            "funding_coverage": "published_events;maximum_internal_gap_9_hour_bins",
            "production_eligible": False,
        }
    )
