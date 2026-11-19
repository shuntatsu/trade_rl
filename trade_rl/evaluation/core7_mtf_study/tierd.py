"""Tier D (known-used) real-data assembly for the Core7 MTF study, without any economics.

This module turns checksum-verified hourly Binance spot/USD-M perpetual series into the
exact 4h, 11-pair, 22-leg alternating ``MarketDataset`` that ``run_study`` requires.
It owns no network, database, replay or return computation: callers (the
``tools.core7_mtf_tierd`` wiring) read the rows, this module only guards, aggregates
and assembles them, and refuses everything that is not complete.

Contract (fail closed everywhere):

* The embargo instant is ``protocol.DATA_END_EXCLUSIVE`` (``2026-08-01T00:00:00Z``).
  ``MarketDataset`` timestamps are bar *close* times, so the last admissible 4h bar
  opens at 16:00 and closes at 20:00 on 2026-07-31 (``LOAD_END``); the bar that would
  close exactly at the embargo instant is excluded because ``enforce_pre_fs1_boundary``
  refuses a timestamp equal to it. :class:`GuardedRowSource` refuses the request
  *before* the inner source is read and again checks every returned row, so the
  embargoed span is never even loaded; the assembler re-checks after aggregation.
* The roster is exactly ``protocol.ROSTER`` in protocol order (not sorted). A missing,
  extra or reordered symbol, a differing leg clock or an incomplete funding calendar
  is a refusal; nothing is dropped or defaulted.
* ``SPOT_HALTS`` stay unavailable rows: a 4h spot bar is tradable only if all four of its
  hours are, halted hours carry zero volume and only the preceding published close as
  a non-tradable valuation mark (never a tradable price), and the assembler verifies
  that every 4h spot bar touching a declared halt is non-tradable.
* Cost columns of the dataset are zero on purpose: ``ExecutionCostConfig`` costs are
  additive to dataset costs, so the frozen scenario costs must be the only costs.
  Exchange lot size and minimum notional come from an exchange-rule snapshot that is
  applied statically and is not point-in-time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.core7_mtf_study import protocol

HOUR_MS: Final = 3_600_000
BAR_HOURS: Final = 4
BAR_MS: Final = BAR_HOURS * HOUR_MS
BASE_INTERVAL: Final = "1h"
TIERD_CONTRACT: Final = "core7_mtf_tierd_v1"

DATA_END_EXCLUSIVE: Final = datetime.fromisoformat(protocol.DATA_END_EXCLUSIVE).replace(
    tzinfo=UTC
)
DATA_END_EXCLUSIVE_MS: Final = int(DATA_END_EXCLUSIVE.timestamp() * 1_000)
#: Exclusive end of the hourly load range: the close of the last admissible 4h bar.
LOAD_END: Final = DATA_END_EXCLUSIVE - timedelta(hours=BAR_HOURS)
LOAD_END_MS: Final = int(LOAD_END.timestamp() * 1_000)
#: First bar of the Tier D replay window (known-used development data).
REPLAY_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
#: Earliest warm-up instant the frozen protocol discloses (Pre-FS1 window start).
EARLIEST_LOAD_START: Final = datetime(2021, 1, 1, tzinfo=UTC)

_MAX_FUNDING_GAP_BARS: Final = 3
_MAX_FUNDING_EDGE_BARS: Final = 3


class TierDEmbargoError(ValueError):
    """Data at or after ``2026-08-01T00:00:00Z`` was requested or encountered."""


class TierDDataError(ValueError):
    """The assembled input is incomplete, mis-ordered or internally inconsistent."""


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def validate_load_window(
    load_start: datetime, load_end: datetime = LOAD_END
) -> tuple[datetime, datetime]:
    """Validate the hourly load range; the embargo is refused before any read."""
    start = _utc(load_start, field="load_start")
    end = _utc(load_end, field="load_end")
    if end > LOAD_END:
        raise TierDEmbargoError(
            f"load_end {end.isoformat()} is later than {LOAD_END.isoformat()}; the "
            f"bar closing at or after {DATA_END_EXCLUSIVE.isoformat()} is refused"
        )
    if not EARLIEST_LOAD_START <= start <= REPLAY_START:
        raise ValueError(
            "load_start must lie within [2021-01-01, 2022-01-01] "
            "(protocol warm-up allowance up to the Tier D replay start)"
        )
    for name, value in (("load_start", start), ("load_end", end)):
        if _epoch_ms(value) % BAR_MS:
            raise ValueError(f"{name} must be aligned to the {BAR_HOURS}h grid")
    if (_epoch_ms(end) - _epoch_ms(start)) < 8 * BAR_MS:
        raise ValueError("the load range must contain at least eight 4h bars")
    return start, end


class _RowSource(Protocol):
    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> Sequence[Any]: ...

    def read_funding(
        self, market: Any, symbol: str, start_ms: int, end_ms: int
    ) -> Sequence[Any]: ...


class GuardedRowSource:
    """Refuse the embargo at load time: before the read and on every returned row.

    The guard is independent of the inner source's own embargo (the PostgreSQL store
    and the Vision archive source have their own); a source that returns a row outside
    the requested range, or at or after ``2026-08-01T00:00:00Z``, is refused even if its
    own checks pass. Only hourly klines are served (the base interval of the 4h build).
    """

    def __init__(self, inner: _RowSource) -> None:
        self._inner = inner
        self.rows_read = 0
        self.max_open_time_ms: int | None = None

    def _check_request(self, start_ms: int, end_ms: int, *, what: str) -> None:
        if end_ms > DATA_END_EXCLUSIVE_MS:
            raise TierDEmbargoError(
                f"{what} request ends at {end_ms} ms, past the embargo "
                f"{DATA_END_EXCLUSIVE.isoformat()}; refused before reading"
            )
        if end_ms > LOAD_END_MS:
            raise TierDEmbargoError(
                f"{what} request ends at {end_ms} ms, past the last admissible 4h "
                f"bar close {LOAD_END.isoformat()}; refused before reading"
            )
        if start_ms >= end_ms:
            raise ValueError(f"{what} request range is empty")

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        if interval != BASE_INTERVAL:
            raise ValueError(f"only {BASE_INTERVAL} klines are admitted")
        self._check_request(start_ms, end_ms, what="kline")
        rows = list(self._inner.read_klines(market, symbol, interval, start_ms, end_ms))
        for row in rows:
            opened = int(row.open_time_ms)
            if opened >= DATA_END_EXCLUSIVE_MS:
                raise TierDEmbargoError(
                    f"{symbol} kline opening at {opened} ms is at or after the embargo"
                )
            if not start_ms <= opened < end_ms:
                raise TierDEmbargoError(
                    f"{symbol} kline opening at {opened} ms lies outside the "
                    f"requested range [{start_ms}, {end_ms})"
                )
            self.max_open_time_ms = (
                opened
                if self.max_open_time_ms is None
                else max(self.max_open_time_ms, opened)
            )
        self.rows_read += len(rows)
        return rows

    def read_funding(
        self, market: Any, symbol: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        self._check_request(start_ms, end_ms, what="funding")
        rows = list(self._inner.read_funding(market, symbol, start_ms, end_ms))
        for row in rows:
            stamp = int(row.funding_time_ms)
            if stamp >= DATA_END_EXCLUSIVE_MS:
                raise TierDEmbargoError(
                    f"{symbol} funding row at {stamp} ms is at or after the embargo"
                )
            # inclusive-end sources may return a row exactly at ``end_ms``; the
            # exclusive-end filter of the pair loader drops it
            if not start_ms <= stamp <= end_ms:
                raise TierDEmbargoError(
                    f"{symbol} funding row at {stamp} ms lies outside the "
                    f"requested range [{start_ms}, {end_ms}]"
                )
        self.rows_read += len(rows)
        return rows


@dataclass(frozen=True, slots=True)
class LegRules:
    """Static exchange rules of one leg (a non point-in-time snapshot)."""

    lot_size: float
    min_notional: float

    def __post_init__(self) -> None:
        for name, value in (
            ("lot_size", self.lot_size),
            ("min_notional", self.min_notional),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class _Bars4h:
    close_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding: np.ndarray
    funding_count: np.ndarray
    tradable: np.ndarray


def _ms(series: RawMarketSeries) -> np.ndarray:
    return np.asarray(series.timestamps).astype("datetime64[ms]").astype(np.int64)


def _aggregate_4h(series: RawMarketSeries, *, leg: str) -> _Bars4h:
    stamps = _ms(series)
    n_hours = int(stamps.size)
    if n_hours == 0 or n_hours % BAR_HOURS:
        raise TierDDataError(f"{leg}: hourly rows must fill whole {BAR_HOURS}h bars")
    if np.any(np.diff(stamps) != HOUR_MS):
        raise TierDDataError(f"{leg}: hourly clock must be complete and regular")
    if (int(stamps[0]) - HOUR_MS) % BAR_MS:
        raise TierDDataError(f"{leg}: the first bar must open on the {BAR_HOURS}h grid")
    if int(stamps[-1]) >= DATA_END_EXCLUSIVE_MS:
        raise TierDEmbargoError(
            f"{leg}: bar closing at {int(stamps[-1])} ms is at or after the embargo"
        )
    counts = series.funding_event_count
    if counts is None:
        raise TierDDataError(f"{leg}: funding event counts are required")

    def grid(values: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(-1, BAR_HOURS)

    open_ = grid(series.open)
    high = grid(series.high)
    low = grid(series.low)
    close = grid(series.close)
    volume = grid(series.volume)
    funding = grid(series.funding_rate)
    count = grid(counts, np.int64)
    tradable = grid(series.tradable, np.bool_)
    stamps4 = stamps.reshape(-1, BAR_HOURS)
    return _Bars4h(
        close_ns=(stamps4[:, -1] * 1_000_000).astype("datetime64[ns]"),
        open=open_[:, 0].copy(),
        high=high.max(axis=1),
        low=low.min(axis=1),
        close=close[:, -1].copy(),
        volume=volume.sum(axis=1),
        funding=funding.sum(axis=1),
        funding_count=count.sum(axis=1).astype(np.int32),
        tradable=tradable.all(axis=1),
    )


def _check_funding_calendar(bars: _Bars4h, *, leg: str) -> None:
    events = np.flatnonzero(bars.funding_count)
    if not len(events):
        raise TierDDataError(f"{leg}: no funding events")
    if events[0] > _MAX_FUNDING_EDGE_BARS:
        raise TierDDataError(
            f"{leg}: the first funding event is too far from the start"
        )
    if len(bars.funding_count) - 1 - events[-1] > _MAX_FUNDING_EDGE_BARS:
        raise TierDDataError(f"{leg}: the last funding event is too far from the end")
    if np.any(np.diff(events) > _MAX_FUNDING_GAP_BARS):
        raise TierDDataError(
            f"{leg}: an interior funding gap exceeds the calendar bound"
        )
    if np.any(bars.funding[bars.funding_count == 0] != 0.0):
        raise TierDDataError(f"{leg}: funding rate present without a funding event")


def _check_halts_masked(
    close_ns: np.ndarray,
    spot_tradable: np.ndarray,
    halts: Sequence[tuple[datetime, datetime]],
    *,
    symbol: str,
) -> None:
    close_ms = close_ns.astype("datetime64[ms]").astype(np.int64)
    for lo, hi in halts:
        lo_ms, hi_ms = (
            _epoch_ms(_utc(lo, field="halt")),
            _epoch_ms(_utc(hi, field="halt")),
        )
        touched = (close_ms - BAR_MS < hi_ms) & (close_ms > lo_ms)
        if np.any(spot_tradable[touched]):
            raise TierDDataError(
                f"{symbol}:spot has a tradable bar inside the declared halt "
                f"[{lo.isoformat()}, {hi.isoformat()})"
            )


def assemble_tierd_dataset(
    pairs: Mapping[str, tuple[RawMarketSeries, RawMarketSeries]],
    *,
    rules: Mapping[str, tuple[LegRules, LegRules]],
    spot_halt_intervals: Sequence[tuple[datetime, datetime]] = (),
) -> MarketDataset:
    """Assemble the 4h, 11-pair, 22-leg alternating Dataset in ``protocol.ROSTER`` order.

    ``pairs`` maps each symbol to its hourly ``(spot, perpetual)`` series, as produced by
    the strict carry pair loader (spot halts as non-tradable zero-volume hours, no
    other missing bar). ``rules`` maps each symbol to its ``(spot, perpetual)`` rules.
    """
    roster = protocol.ROSTER
    if set(pairs) != set(roster) or len(pairs) != len(roster):
        missing = sorted(set(roster) - set(pairs))
        extra = sorted(set(pairs) - set(roster))
        raise TierDDataError(
            f"pairs must be exactly the {len(roster)}-symbol roster "
            f"(missing={missing}, extra={extra})"
        )
    if set(rules) != set(roster):
        raise TierDDataError("rules must cover exactly the roster symbols")

    columns: list[_Bars4h] = []
    legs: list[str] = []
    lot: list[float] = []
    minimum: list[float] = []
    for symbol in roster:
        spot_series, perp_series = pairs[symbol]
        spot = _aggregate_4h(spot_series, leg=f"{symbol}:spot")
        perp = _aggregate_4h(perp_series, leg=f"{symbol}:perp")
        if np.any(spot.funding_count) or np.any(spot.funding):
            raise TierDDataError(f"{symbol}:spot must not carry funding")
        _check_funding_calendar(perp, leg=f"{symbol}:perp")
        if not np.all(perp.tradable):
            raise TierDDataError(
                f"{symbol}:perp has no halt calendar and must be tradable"
            )
        _check_halts_masked(
            spot.close_ns, spot.tradable, spot_halt_intervals, symbol=symbol
        )
        spot_rules, perp_rules = rules[symbol]
        columns.extend((spot, perp))
        legs.extend((f"{symbol}:spot", f"{symbol}:perp"))
        lot.extend((spot_rules.lot_size, perp_rules.lot_size))
        minimum.extend((spot_rules.min_notional, perp_rules.min_notional))

    clock = columns[0].close_ns
    if len(clock) < 8:
        raise TierDDataError("the dataset needs at least eight 4h bars")
    for leg, bars in zip(legs, columns, strict=True):
        if not np.array_equal(bars.close_ns, clock):
            raise TierDDataError(f"{leg}: clock differs from the first leg")
    if (
        int(clock[-1].astype("datetime64[ms]").astype(np.int64))
        >= DATA_END_EXCLUSIVE_MS
    ):
        raise TierDEmbargoError(
            "a bar timestamp reaches the 2026-08-01T00:00:00Z embargo"
        )

    n_bars, n_legs = len(clock), len(legs)
    shape = (n_bars, n_legs)

    def stacked(name: str) -> np.ndarray:
        return np.column_stack([getattr(bars, name) for bars in columns])

    funding_count = stacked("funding_count").astype(np.int32)
    close = stacked("close")
    zero = np.zeros(shape)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=tuple(legs),
        timestamps=clock,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("core7_no_features",),
        global_feature_names=("core7_no_features",),
        open=stacked("open"),
        high=stacked("high"),
        low=stacked("low"),
        close=close,
        volume=stacked("volume"),
        volume_units=tuple(VolumeUnit.QUOTE_NOTIONAL for _ in legs),
        funding_rate=stacked("funding"),
        funding_event_count=funding_count,
        funding_due=funding_count > 0,
        tradable=stacked("tradable"),
        periods_per_year=2190,
        fee_rate=zero,
        maker_fee_rate=zero,
        taker_fee_rate=zero,
        spread_rate=zero,
        lot_size=np.broadcast_to(np.asarray(lot, dtype=np.float64), shape),
        minimum_notional=np.broadcast_to(np.asarray(minimum, dtype=np.float64), shape),
        # spot legs are not borrowable (no spot shorts; the engine enforces it, the
        # policy already clips spot to >= 0); perpetuals short natively
        borrow_available=np.broadcast_to(
            np.tile(np.asarray([False, True]), len(roster)), shape
        ),
        contract_multipliers=np.ones(n_legs, dtype=np.float64),
        mark_price=close,
    )
    return dataset.with_content_identity(
        {
            "source_contract": TIERD_CONTRACT,
            "roster": list(roster),
            "bar_hours": BAR_HOURS,
            "aggregation": (
                "hourly_to_4h_open_first_close_last_high_max_low_min_volume_sum;"
                "funding_summed_over_the_four_hours;tradable_only_if_all_hours_tradable"
            ),
            "timestamps": "bar_close",
            "data_end_exclusive": protocol.DATA_END_EXCLUSIVE,
            "spot_halts": [
                [_utc(lo, field="halt").isoformat(), _utc(hi, field="halt").isoformat()]
                for lo, hi in spot_halt_intervals
            ],
            "halted_bar_marks": "preceding_published_close_non_tradable_zero_volume",
            "dataset_costs": "zero_scenario_costs_additive_in_ExecutionCostConfig",
            "rules_assumption": "static_exchange_rule_snapshot_not_point_in_time",
            "borrow_assumption": "spot_not_borrowable_perp_native_short_zero_borrow_rate",
            "mark_price_limitation": "bar_close_proxy",
            "production_eligible": False,
        }
    )


def without_identity_payload(dataset: MarketDataset) -> MarketDataset:
    """The same arrays and ``dataset_id`` without the self-verifying identity payload.

    ``run_study``'s causality probe perturbs a ``dataclasses.replace`` copy under a new
    id; a copy that still carries the source identity payload is refused by
    ``MarketDataset`` (the id no longer matches payload and arrays). Until the study
    runner drops the payload from its probe copy itself, the driver hands it a dataset
    whose content id is unchanged but whose payload is not re-verified.
    """
    return replace(dataset, identity_payload_json=None)


def replay_indices(dataset: MarketDataset) -> tuple[int, int]:
    """``(start_index, stop_index)`` of the Tier D replay.

    ``start_index`` is the first row whose bar opens at or after ``REPLAY_START``; all
    earlier rows are warm-up history only. ``stop_index`` is the last row.
    """
    close_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    first_close = _epoch_ms(REPLAY_START + timedelta(hours=BAR_HOURS)) * 1_000_000
    start = int(np.searchsorted(close_ns, first_close, side="left"))
    if start >= dataset.n_bars - 1:
        raise TierDDataError("the dataset does not reach the Tier D replay window")
    return start, dataset.n_bars - 1


def dataset_summary(dataset: MarketDataset) -> dict[str, object]:
    """Shape, range and availability counts only; computes no return or signal."""
    close_ns = dataset.timestamps.astype("datetime64[ns]")
    tradable = np.asarray(dataset.tradable, dtype=bool)
    events = np.asarray(dataset.resolved_array("funding_event_count"))
    start_index, stop_index = replay_indices(dataset)
    per_leg: dict[str, dict[str, int]] = {}
    for column, leg in enumerate(dataset.symbols):
        per_leg[leg] = {
            "bars": int(dataset.n_bars),
            "tradable": int(tradable[:, column].sum()),
            "unavailable": int((~tradable[:, column]).sum()),
            "funding_events": int(events[:, column].sum()),
        }
    return {
        "dataset_id": dataset.dataset_id,
        "shape": [int(dataset.n_bars), int(dataset.n_symbols)],
        "bar_hours": BAR_HOURS,
        "first_bar_close": str(close_ns[0]) + "Z",
        "last_bar_close": str(close_ns[-1]) + "Z",
        "embargo_instant": DATA_END_EXCLUSIVE.isoformat(),
        "start_index": start_index,
        "stop_index": stop_index,
        "legs": list(dataset.symbols),
        "per_leg": per_leg,
    }


__all__ = [
    "BAR_HOURS",
    "DATA_END_EXCLUSIVE",
    "DATA_END_EXCLUSIVE_MS",
    "EARLIEST_LOAD_START",
    "LOAD_END",
    "LOAD_END_MS",
    "REPLAY_START",
    "TIERD_CONTRACT",
    "GuardedRowSource",
    "LegRules",
    "TierDDataError",
    "TierDEmbargoError",
    "assemble_tierd_dataset",
    "dataset_summary",
    "replay_indices",
    "validate_load_window",
    "without_identity_payload",
]
