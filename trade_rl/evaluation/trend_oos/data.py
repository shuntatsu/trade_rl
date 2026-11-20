"""Stage-window 4h dataset assembly for ``trend_long_spot_oos_v1``, without any economics.

This module turns checksum-verified hourly Binance *spot* series into the 4h, 11-pair,
22-leg alternating ``MarketDataset`` the unchanged lowturn policy trades on, for a stage
window (BACKCAST or FORWARD) instead of the single Tier D window of
``core7_mtf_study.tierd``. It owns no network, database, replay or return computation:
the ``tools.trend_oos`` wiring reads the rows; this module only guards, aggregates and
assembles them and refuses everything that is not complete.

Contract (fail closed everywhere):

* **Leg layout.** The 22-leg layout and the roster are exactly lowturn's. Neither arm
  reads a perpetual, so no perpetual is loaded: every perp leg is unavailable (inactive,
  not tradable, information unavailable) in every row. A spot market that did not exist
  yet is unavailable in the same way before its first complete 4h bar; a pair with no
  spot data in the window is a valid all-unavailable leg. Unavailable rows carry one
  constant *sentinel* price (``UNAVAILABLE_PRICE``) only because ``MarketDataset`` needs
  finite positive OHLC; no price is ever invented, back-filled or forward-filled, and the
  signals (:mod:`trade_rl.evaluation.trend_oos.signals`) treat every such row as missing.
* **Embargo.** ``tierd`` keeps refusing 2026-08-01 and later; nothing here changes it.
  The FORWARD stage has its own read limit (:func:`read_limit`): it exists only for the
  exact owner authorization URL, never exceeds the stage's last row (the last decision
  bar plus settlement) nor the latest fully closed bar at fetch time, and
  :class:`StageRowSource` refuses any request or returned row beyond it before the data
  reaches the assembler.
* **Halts** are ``protocol.stage_halt_intervals()`` (the registered ``SPOT_HALTS`` plus the
  declared exchange-wide outages): a 4h spot bar is tradable only if all four
  hours are, halted hours carry zero volume and only the preceding published close.
* **Exchange rules** (lot size, minimum notional) are one static snapshot applied to
  every row; they are not point-in-time (for BACKCAST this approximates 2018-2020 by
  the retrieval-time rules) and the dataset identity says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.core7_mtf_study import tierd
from trade_rl.evaluation.trend_oos import protocol

HOUR_MS: Final = tierd.HOUR_MS
BAR_MS: Final = tierd.BAR_MS
BAR: Final = timedelta(hours=protocol.BAR_HOURS)
BASE_INTERVAL: Final = tierd.BASE_INTERVAL
STAGE_DATA_CONTRACT: Final = "trend_oos_stage_v1"
#: Constant placeholder of an unavailable row; never a market price.
UNAVAILABLE_PRICE: Final = 1.0
#: The default (Tier D / core7) embargo instant; reading at or after it needs the
#: explicit FORWARD authorization path below.
DEFAULT_EMBARGO: Final = tierd.DATA_END_EXCLUSIVE


class StageEmbargoError(ValueError):
    """A read beyond a stage window, the latest closed bar or the authorization path."""


class StageDataError(ValueError):
    """The assembled input is incomplete, mis-ordered or inconsistent with its window."""


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _parse_instant(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def _on_grid(value: datetime) -> bool:
    return _ms(value) % BAR_MS == 0


@dataclass(frozen=True, slots=True)
class StageWindow:
    """One stage's clock; all instants are bar-close timestamps of dataset rows."""

    stage: str
    load_start: datetime
    first_decision: datetime
    last_decision_bar: datetime
    settles: datetime
    warmup_bars: int = protocol.WARMUP_BARS

    def __post_init__(self) -> None:
        if self.stage not in protocol.STAGES:
            raise ValueError(f"unknown stage: {self.stage!r}")
        for name in ("load_start", "first_decision", "last_decision_bar", "settles"):
            value = _utc(getattr(self, name), field=name)
            if not _on_grid(value):
                raise ValueError(f"{name} must lie on the {protocol.BAR_HOURS}h grid")
        if self.settles != self.last_decision_bar + BAR:
            raise ValueError("settles must be one bar after the last decision bar")
        if not self.load_start < self.first_decision < self.last_decision_bar:
            raise ValueError("window instants must be strictly increasing")
        if self.first_decision_index < self.warmup_bars:
            raise ValueError("the first decision precedes the warm-up bars")

    @property
    def first_decision_index(self) -> int:
        """Row index of the first decision (row 0 closes one bar after load start)."""
        return self.row_index(self.first_decision)

    @property
    def last_row_close(self) -> datetime:
        """Close of the last row: the settlement bar's close, two bars after the last
        decision bar (the engine decides through ``stop - 2``)."""
        return self.last_decision_bar + 2 * BAR

    @property
    def last_row_index(self) -> int:
        return self.row_index(self.last_row_close)

    def row_index(self, close: datetime) -> int:
        delta = _utc(close, field="close") - self.load_start - BAR
        if delta < timedelta(0) or delta % BAR != timedelta(0):
            raise ValueError("instant is not the close of a row of this clock")
        return int(delta // BAR)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "load_start": self.load_start.isoformat(),
            "first_decision": self.first_decision.isoformat(),
            "last_decision_bar": self.last_decision_bar.isoformat(),
            "settles": self.settles.isoformat(),
            "last_row_close": self.last_row_close.isoformat(),
            "first_decision_index": self.first_decision_index,
            "last_row_index": self.last_row_index,
            "warmup_bars": self.warmup_bars,
        }


def frozen_window(stage: str) -> StageWindow:
    """The window of ``stage`` exactly as the frozen protocol table gives it."""
    if stage not in protocol.STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    window = protocol.STAGE_WINDOWS[stage]
    return StageWindow(
        stage=stage,
        load_start=_parse_instant(window["load_start"]),
        first_decision=_parse_instant(window["first_decision"]),
        last_decision_bar=_parse_instant(window["last_decision_bar"]),
        settles=_parse_instant(window["settles"]),
    )


def latest_closed_bar(now: datetime) -> datetime:
    """Close of the latest fully closed 4h bar at ``now`` (a bar closes at its stamp)."""
    moment = _utc(now, field="now")
    return datetime.fromtimestamp(_ms(moment) // BAR_MS * BAR_MS / 1_000, UTC)


def read_limit(
    window: StageWindow,
    *,
    now: datetime,
    authorization_url: str | None = None,
    through: datetime | None = None,
) -> datetime:
    """The last row close that may be read: the exclusive limit of the hourly opens.

    * BACKCAST ends 2021-01-01 and needs no authorization, but it is refused if the
      stage is not yet over.
    * FORWARD data reads at or after the default embargo exist only on the explicit path
      bound to ``protocol.DATA_AUTHORIZATION_URL``; any other value is refused.
    * The limit never exceeds the stage's last row or the latest fully closed bar.
      ``through`` (a smaller 4h-grid instant) may only lower it.
    """
    limit = min(window.last_row_close, latest_closed_bar(now))
    if limit <= window.first_decision + 2 * BAR:
        raise StageEmbargoError(
            f"no closed bar beyond the first decision {window.first_decision.isoformat()}"
        )
    if window.stage == protocol.STAGE_FORWARD:
        if authorization_url != protocol.DATA_AUTHORIZATION_URL:
            raise StageEmbargoError(
                "FORWARD reads need the owner authorization "
                f"{protocol.DATA_AUTHORIZATION_URL}"
            )
    elif limit != window.last_row_close:
        raise StageEmbargoError(f"the {window.stage} stage window is not over yet")
    if through is not None:
        stamp = _utc(through, field="through")
        if not _on_grid(stamp) or not window.first_decision + 2 * BAR < stamp <= limit:
            raise StageEmbargoError(
                "through must be a 4h-grid instant after the first decision and not "
                f"after {limit.isoformat()}"
            )
        limit = stamp
    if limit > DEFAULT_EMBARGO and window.stage != protocol.STAGE_FORWARD:
        raise StageEmbargoError(
            "only the FORWARD stage may read past the default embargo"
        )
    return limit


class _RowSource(Protocol):
    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> Sequence[Any]: ...


class StageRowSource:
    """Refuse reads outside the stage: before the read and on every returned row.

    Only hourly *spot* klines of roster symbols from the load start to ``limit_close``
    are served; no perpetual kline and no funding row is ever requested. ``limit_close``
    is the exclusive bound of the hourly opens (the close of the last admissible 4h bar).
    """

    def __init__(
        self, inner: _RowSource, *, window: StageWindow, limit_close: datetime
    ) -> None:
        if limit_close > window.last_row_close:
            raise StageEmbargoError("limit is past the stage's last row")
        self._inner = inner
        self._start_ms = _ms(window.load_start)
        self._end_ms = _ms(_utc(limit_close, field="limit_close"))
        self.rows_read = 0
        self.max_open_time_ms: int | None = None

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        if interval != BASE_INTERVAL:
            raise ValueError(f"only {BASE_INTERVAL} klines are admitted")
        if getattr(market, "value", market) != "spot":
            raise StageDataError("only spot klines are read; perpetuals are not loaded")
        if symbol not in protocol.ROSTER:
            raise StageDataError(f"{symbol} is not in the roster")
        if end_ms > self._end_ms:
            raise StageEmbargoError(
                f"kline request ends at {end_ms} ms, past the read limit "
                f"{self._end_ms} ms; refused before reading"
            )
        if start_ms < self._start_ms:
            raise StageEmbargoError("kline request starts before the stage load start")
        if start_ms >= end_ms:
            raise ValueError("kline request range is empty")
        rows = list(self._inner.read_klines(market, symbol, interval, start_ms, end_ms))
        for row in rows:
            opened = int(row.open_time_ms)
            if not start_ms <= opened < end_ms:
                raise StageEmbargoError(
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

    def probe_published_day(self, symbol: str, day_start_ms: int) -> bool | None:
        """Whether a daily archive publishes the first hour of that day (None: cannot say).

        Only a day inside the stage and below the read limit may be probed.
        """
        probe = getattr(self._inner, "probe_published_day", None)
        if probe is None:
            return None
        if day_start_ms < self._start_ms or day_start_ms + 24 * HOUR_MS > self._end_ms:
            raise StageEmbargoError("a probed day lies outside the stage read range")
        return bool(probe(symbol, day_start_ms))

    def read_funding(self, *_: object, **__: object) -> list[Any]:
        raise StageDataError("funding is never read: no perpetual leg is loaded")


@dataclass(frozen=True, slots=True)
class _Bars4h:
    close_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    tradable: np.ndarray


def _aggregate_spot_4h(series: RawMarketSeries, *, leg: str) -> _Bars4h:
    """Hourly (close-stamped) spot rows to whole 4h bars; the series must be complete."""
    stamps = np.asarray(series.timestamps).astype("datetime64[ms]").astype(np.int64)
    n_hours = int(stamps.size)
    if n_hours == 0 or n_hours % protocol.BAR_HOURS:
        raise StageDataError(f"{leg}: hourly rows must fill whole 4h bars")
    if np.any(np.diff(stamps) != HOUR_MS):
        raise StageDataError(f"{leg}: hourly clock must be complete and regular")
    if (int(stamps[0]) - HOUR_MS) % BAR_MS:
        raise StageDataError(f"{leg}: the first bar must open on the 4h grid")
    if np.any(np.asarray(series.funding_event_count)) or np.any(series.funding_rate):
        raise StageDataError(f"{leg}: spot must not carry funding")

    def grid(values: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(-1, protocol.BAR_HOURS)

    stamps4 = stamps.reshape(-1, protocol.BAR_HOURS)
    return _Bars4h(
        close_ns=(stamps4[:, -1] * 1_000_000).astype("datetime64[ns]"),
        open=grid(series.open)[:, 0].copy(),
        high=grid(series.high).max(axis=1),
        low=grid(series.low).min(axis=1),
        close=grid(series.close)[:, -1].copy(),
        volume=grid(series.volume).sum(axis=1),
        tradable=grid(series.tradable, np.bool_).all(axis=1),
    )


def stage_clock(window: StageWindow, limit_close: datetime) -> np.ndarray:
    """Row close timestamps from the first row to ``limit_close`` inclusive."""
    limit = _utc(limit_close, field="limit_close")
    if not _on_grid(limit) or limit > window.last_row_close:
        raise StageEmbargoError("limit must be a 4h-grid instant within the stage")
    if limit <= window.first_decision + 2 * BAR:
        raise StageDataError("the window holds no replayable decision")
    n_rows = window.row_index(limit) + 1
    start = np.datetime64(
        (window.load_start + BAR).replace(tzinfo=None).isoformat(), "ns"
    )
    return (start + np.arange(n_rows) * np.timedelta64(protocol.BAR_HOURS, "h")).astype(
        "datetime64[ns]"
    )


def assemble_stage_dataset(
    window: StageWindow,
    spot: Mapping[str, RawMarketSeries | None],
    *,
    rules: Mapping[str, tuple[tierd.LegRules, tierd.LegRules]],
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    data_authorization_url: str | None = None,
) -> MarketDataset:
    """Assemble the 4h, 11-pair, 22-leg alternating dataset of one stage window.

    ``spot`` maps each roster symbol to its complete hourly spot series from its first
    4h-aligned listed hour to ``limit_close`` (see the module docstring), or ``None``
    when the symbol has no spot data in the window. ``rules`` maps each symbol to its
    ``(spot, perp)`` rules; the perpetual rules only fill inert columns.
    """
    roster = protocol.ROSTER
    if halt_intervals is None:
        halt_intervals = protocol.stage_halt_intervals()
    if set(spot) != set(roster) or len(spot) != len(roster):
        raise StageDataError(
            "spot must cover exactly the roster "
            f"(missing={sorted(set(roster) - set(spot))}, "
            f"extra={sorted(set(spot) - set(roster))})"
        )
    if set(rules) != set(roster):
        raise StageDataError("rules must cover exactly the roster symbols")
    clock = stage_clock(window, limit_close)
    limit = _utc(limit_close, field="limit_close")
    needs_authorization = limit > DEFAULT_EMBARGO
    if needs_authorization and (
        window.stage != protocol.STAGE_FORWARD
        or data_authorization_url != protocol.DATA_AUTHORIZATION_URL
    ):
        raise StageEmbargoError(
            "rows at or after the default embargo need the FORWARD stage and the "
            f"owner authorization {protocol.DATA_AUTHORIZATION_URL}"
        )
    n_rows, n_legs = len(clock), 2 * len(roster)
    shape = (n_rows, n_legs)
    clock_ns = clock.astype("datetime64[ns]").astype(np.int64)

    open_ = np.full(shape, UNAVAILABLE_PRICE)
    high = np.full(shape, UNAVAILABLE_PRICE)
    low = np.full(shape, UNAVAILABLE_PRICE)
    close = np.full(shape, UNAVAILABLE_PRICE)
    volume = np.zeros(shape)
    tradable = np.zeros(shape, dtype=np.bool_)
    active = np.zeros(shape, dtype=np.bool_)
    lot = np.empty(n_legs)
    minimum = np.empty(n_legs)
    listing: dict[str, dict[str, object]] = {}
    for pair, symbol in enumerate(roster):
        spot_rules, perp_rules = rules[symbol]
        lot[2 * pair], lot[2 * pair + 1] = spot_rules.lot_size, perp_rules.lot_size
        minimum[2 * pair] = spot_rules.min_notional
        minimum[2 * pair + 1] = perp_rules.min_notional
        series = spot[symbol]
        if series is None:
            listing[symbol] = {"first_listed_row": None, "unavailable_rows": n_rows}
            continue
        bars = _aggregate_spot_4h(series, leg=f"{symbol}:spot")
        first_ns = int(bars.close_ns[0].astype("datetime64[ns]").astype(np.int64))
        first_row = int(np.searchsorted(clock_ns, first_ns))
        stop = first_row + len(bars.close_ns)
        if first_row >= n_rows or clock_ns[first_row] != first_ns:
            raise StageDataError(f"{symbol}:spot starts off the stage clock")
        if stop != n_rows or not np.array_equal(
            clock_ns[first_row:],
            bars.close_ns.astype("datetime64[ns]").astype(np.int64),
        ):
            raise StageDataError(
                f"{symbol}:spot must run without a gap to the read limit "
                f"{limit.isoformat()}"
            )
        tierd._check_halts_masked(
            bars.close_ns, bars.tradable, halt_intervals, symbol=symbol
        )
        column = 2 * pair
        window_rows = slice(first_row, stop)
        open_[window_rows, column] = bars.open
        high[window_rows, column] = bars.high
        low[window_rows, column] = bars.low
        close[window_rows, column] = bars.close
        volume[window_rows, column] = bars.volume
        tradable[window_rows, column] = bars.tradable
        active[window_rows, column] = True
        listing[symbol] = {"first_listed_row": first_row, "unavailable_rows": first_row}

    if listing[protocol.ROSTER[0]]["first_listed_row"] != 0:
        raise StageDataError(
            f"{protocol.ROSTER[0]} must be listed from the stage load start: an "
            "all-unavailable dataset means the source is unreachable or misconfigured"
        )
    zero = np.zeros(shape)
    funding_count = np.zeros(shape, dtype=np.int32)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=protocol.LEG_ROSTER,
        timestamps=clock,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((n_rows, 1), dtype=np.float32),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("trend_oos_no_features",),
        global_feature_names=("trend_oos_no_features",),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        volume_units=tuple(VolumeUnit.QUOTE_NOTIONAL for _ in range(n_legs)),
        funding_rate=zero,
        funding_event_count=funding_count,
        funding_due=funding_count > 0,
        tradable=tradable,
        asset_active=active,
        periods_per_year=2190,
        fee_rate=zero,
        maker_fee_rate=zero,
        taker_fee_rate=zero,
        spread_rate=zero,
        lot_size=np.broadcast_to(lot, shape),
        minimum_notional=np.broadcast_to(minimum, shape),
        borrow_available=np.broadcast_to(
            np.tile(np.asarray([False, True]), len(roster)), shape
        ),
        contract_multipliers=np.ones(n_legs, dtype=np.float64),
        mark_price=close,
    )
    identity: dict[str, object] = {
        "source_contract": STAGE_DATA_CONTRACT,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": window.stage,
        "window": window.to_dict(),
        "read_limit_close": limit.isoformat(),
        "roster": list(roster),
        "bar_hours": protocol.BAR_HOURS,
        "aggregation": (
            "hourly_to_4h_open_first_close_last_high_max_low_min_volume_sum;"
            "tradable_only_if_all_hours_tradable"
        ),
        "timestamps": "bar_close",
        "spot_halts": [
            [_utc(lo, field="halt").isoformat(), _utc(hi, field="halt").isoformat()]
            for lo, hi in halt_intervals
        ],
        "halted_bar_marks": "preceding_published_close_non_tradable_zero_volume",
        "perp_legs": "not_loaded_inactive_in_every_row",
        "unavailable_rows": (
            f"asset_inactive_not_tradable_constant_sentinel_price_{UNAVAILABLE_PRICE}"
        ),
        "listing": listing,
        "dataset_costs": "zero_scenario_costs_additive_in_ExecutionCostConfig",
        "rules_assumption": "static_exchange_rule_snapshot_not_point_in_time",
        "borrow_assumption": "spot_not_borrowable_perp_native_short_zero_borrow_rate",
        "mark_price_limitation": "bar_close_proxy",
        "production_eligible": False,
    }
    if needs_authorization:
        identity["data_authorization_url"] = data_authorization_url
    return dataset.with_content_identity(identity)


def validate_stage_dataset(dataset: MarketDataset, window: StageWindow) -> int:
    """Check a dataset against its stage window; return the index of its last row.

    The clock must be the stage clock from the first row, regular, and end at or before
    the stage's last row; the 22-leg roster layout is exact; every perpetual leg is
    unavailable in every row; and a dataset that reaches the default embargo must carry
    the owner authorization URL in its recorded identity. Computes nothing economic.
    """
    if tuple(dataset.symbols) != protocol.LEG_ROSTER:
        raise StageDataError("dataset legs are not the 22-leg roster layout")
    ts = dataset.timestamps.astype("datetime64[ns]")
    if ts.size < 2:
        raise StageDataError("dataset is too short")
    last = _stamp_to_datetime(ts[-1])
    if last > window.last_row_close:
        raise StageEmbargoError("dataset reaches past the stage's last row")
    if not np.array_equal(ts, stage_clock(window, last)):
        raise StageDataError("dataset clock is not the stage clock from its first row")
    perp = np.arange(len(protocol.ROSTER)) * 2 + 1
    active = dataset.resolved_array("asset_active")
    if active[:, perp].any() or np.asarray(dataset.tradable)[:, perp].any():
        raise StageDataError("a perpetual leg is available; none may be")
    if dataset.resolved_array("information_available")[:, perp].any():
        raise StageDataError("perpetual information is available; none may be")
    if last > DEFAULT_EMBARGO:
        payload = dataset.identity_payload_json or ""
        if (
            window.stage != protocol.STAGE_FORWARD
            or protocol.DATA_AUTHORIZATION_URL not in payload
        ):
            raise StageEmbargoError(
                "rows at or after the default embargo carry no FORWARD authorization"
            )
    return int(ts.size - 1)


def _stamp_to_datetime(stamp: np.datetime64) -> datetime:
    ns = int(stamp.astype("datetime64[ns]").astype(np.int64))
    return datetime.fromtimestamp(ns / 1_000_000_000, UTC)


def stage_summary(dataset: MarketDataset, window: StageWindow) -> dict[str, object]:
    """Shape, range and availability counts only; computes no return or signal."""
    last_index = validate_stage_dataset(dataset, window)
    tradable = np.asarray(dataset.tradable, dtype=bool)
    active = dataset.resolved_array("asset_active")
    per_leg: dict[str, dict[str, int]] = {}
    for column, leg in enumerate(dataset.symbols):
        per_leg[leg] = {
            "rows": int(dataset.n_bars),
            "listed": int(active[:, column].sum()),
            "tradable": int(tradable[:, column].sum()),
            "unavailable": int((~active[:, column]).sum()),
        }
    return {
        "dataset_id": dataset.dataset_id,
        "shape": [int(dataset.n_bars), int(dataset.n_symbols)],
        "window": window.to_dict(),
        "last_index": last_index,
        "first_bar_close": str(dataset.timestamps[0]) + "Z",
        "last_bar_close": str(dataset.timestamps[-1]) + "Z",
        "per_leg": per_leg,
    }


__all__ = [
    "DEFAULT_EMBARGO",
    "STAGE_DATA_CONTRACT",
    "UNAVAILABLE_PRICE",
    "StageDataError",
    "StageEmbargoError",
    "StageRowSource",
    "StageWindow",
    "assemble_stage_dataset",
    "frozen_window",
    "latest_closed_bar",
    "read_limit",
    "stage_clock",
    "stage_summary",
    "validate_stage_dataset",
]
