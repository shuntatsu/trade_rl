"""Stage-window 4h dataset assembly of ``xs_lowvol_funding_v1``, without any economics.

This module turns checksum-verified hourly Binance USD-M perpetual series (kline rows and
funding events, already parsed by the tool) into the 4h, perp-only ``MarketDataset`` the
weekly book trades on, for the DEV, DISJOINT and FORWARD windows. It owns no network,
database, replay or return computation: ``tools.xs_lowvol_funding`` reads the rows; this
module only guards, aggregates and assembles them and refuses everything that is not
complete.

Contract (fail closed everywhere):

* **Legs** are every symbol that is ever a universe member in the stage (the 11 Tier D core
  pairs for DEV), in sorted order, one perp leg each. No spot market is read.
* **Bar states.** A 4h bar whose four hours are all published is active and tradable. A bar
  whose missing hours all lie inside a declared exchange-wide halt (and whose preceding bar
  is active) is active but not tradable: zero volume, the preceding close, no funding.
  Every other bar (before the first listing, after the archive ends, inside an unexplained
  gap, the partial first bar) is inactive and not tradable with one constant sentinel price,
  never an invented price. The first inactive bar after an active one carries that last
  close as its open: it is the bar at which the engine settles a position the forced-close
  rule could not close.
* **Embargo.** DEV and DISJOINT stop strictly before 2026-08-01; only FORWARD may read past
  it, and only on the exact owner authorization URL, never past the stage's last row nor the
  latest closed bar. DISJOINT and FORWARD datasets carry the authorization in their identity.
* **Exchange rules** are one static snapshot, not point-in-time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import tierd
from trade_rl.evaluation.trend_oos.data import (
    DEFAULT_EMBARGO,
    UNAVAILABLE_PRICE,
    StageDataError,
    StageEmbargoError,
    latest_closed_bar,
)
from trade_rl.evaluation.xs_lowvol_funding import protocol
from trade_rl.evaluation.xs_lowvol_funding.universe import (
    MonthUniverse,
    UniverseSeries,
)

HOUR_MS: Final = tierd.HOUR_MS
BAR_MS: Final = tierd.BAR_MS
BAR: Final = timedelta(hours=protocol.BAR_HOURS)
BASE_INTERVAL: Final = tierd.BASE_INTERVAL
STAGE_DATA_CONTRACT: Final = "xs_lowvol_funding_stage_v1"
_PERP_MARKETS: Final = ("usds_m",)


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
    """One stage's clock; all instants are bar-close timestamps of dataset rows.

    ``settles`` is the close of the last dataset row. The weekly decisions run on Mondays
    from ``first_decision`` to ``last_decision_bar``; the engine decides through the row
    two before the last, flattens on the row before it and marks on the last row.
    """

    stage: str
    load_start: datetime
    first_decision: datetime
    last_decision_bar: datetime
    settles: datetime

    def __post_init__(self) -> None:
        if self.stage not in protocol.STAGES:
            raise ValueError(f"unknown stage: {self.stage!r}")
        for name in ("load_start", "first_decision", "last_decision_bar", "settles"):
            value = _utc(getattr(self, name), field=name)
            if not _on_grid(value):
                raise ValueError(f"{name} must lie on the {protocol.BAR_HOURS}h grid")
        for name in ("first_decision", "last_decision_bar"):
            moment = getattr(self, name)
            if moment.weekday() != 0 or moment.hour != 0:
                raise ValueError(f"{name} must be a Monday 00:00 UTC bar close")
        if (self.last_decision_bar - self.first_decision) % timedelta(days=7):
            raise ValueError("the decisions must be a whole number of weeks apart")
        if not (
            self.load_start
            < self.first_decision
            <= self.last_decision_bar
            < self.settles
        ):
            raise ValueError("window instants must be strictly increasing")
        if self.first_decision_index < protocol.COV_BARS:
            raise ValueError("the first decision precedes the covariance warm-up")

    @property
    def first_decision_index(self) -> int:
        return self.row_index(self.first_decision)

    @property
    def last_row_close(self) -> datetime:
        return self.settles

    @property
    def last_row_index(self) -> int:
        return self.row_index(self.settles)

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


def read_limit(
    window: StageWindow,
    *,
    now: datetime,
    authorization_url: str | None = None,
    through: datetime | None = None,
) -> datetime:
    """The last row close that may be read: the exclusive limit of the hourly opens.

    DEV needs no authorization; DISJOINT and FORWARD reads exist only on the path bound to
    ``protocol.DATA_AUTHORIZATION_URL``. DEV and DISJOINT are refused while their window is
    not over. The limit never exceeds the stage's last row or the latest fully closed bar;
    ``through`` (a smaller 4h-grid instant) may only lower it.
    """
    limit = min(window.last_row_close, latest_closed_bar(now))
    if limit <= window.first_decision + 2 * BAR:
        raise StageEmbargoError(
            f"no closed bar beyond the first decision {window.first_decision.isoformat()}"
        )
    if window.stage in protocol.AUTHORIZED_STAGES:
        if authorization_url != protocol.DATA_AUTHORIZATION_URL:
            raise StageEmbargoError(
                f"{window.stage} reads need the owner authorization "
                f"{protocol.DATA_AUTHORIZATION_URL}"
            )
    if window.stage != protocol.STAGE_FORWARD and limit != window.last_row_close:
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


def _market_name(market: Any) -> str:
    return str(getattr(market, "value", market)).lower()


class StageRowSource:
    """Refuse reads outside the stage: before the read and on every returned row.

    Hourly USD-M klines and funding of the stage's declared symbols from the load start to
    ``limit_close`` are served; ``limit_close`` is the exclusive bound of hourly opens (the
    close of the last admissible 4h bar) and funding rows may not be later.
    """

    def __init__(
        self,
        inner: Any,
        *,
        window: StageWindow,
        limit_close: datetime,
        symbols: Sequence[str],
    ) -> None:
        if limit_close > window.last_row_close:
            raise StageEmbargoError("limit is past the stage's last row")
        self._inner = inner
        self._symbols = frozenset(symbols)
        self._start_ms = _ms(window.load_start)
        self._end_ms = _ms(_utc(limit_close, field="limit_close"))
        self.rows_read = 0
        self.max_open_time_ms: int | None = None

    def _check(self, market: Any, symbol: str, start_ms: int, end_ms: int) -> None:
        name = _market_name(market)
        if name not in _PERP_MARKETS:
            raise StageDataError(f"market {name!r} is not admitted (perps only)")
        if symbol not in self._symbols:
            raise StageDataError(f"{symbol} is not a symbol of this stage")
        if end_ms > self._end_ms:
            raise StageEmbargoError(
                f"request ends at {end_ms} ms, past the read limit {self._end_ms} ms; "
                "refused before reading"
            )
        if start_ms < self._start_ms:
            raise StageEmbargoError("request starts before the stage load start")
        if start_ms >= end_ms:
            raise ValueError("request range is empty")

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        if interval != BASE_INTERVAL:
            raise ValueError(f"only {BASE_INTERVAL} klines are admitted")
        self._check(market, symbol, start_ms, end_ms)
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

    def read_funding(
        self, market: Any, symbol: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        self._check(market, symbol, start_ms, end_ms)
        rows = list(self._inner.read_funding(market, symbol, start_ms, end_ms))
        for row in rows:
            stamp = int(row.funding_time_ms)
            # an inclusive-end source may return a row at exactly ``end_ms``
            if not start_ms <= stamp <= end_ms:
                raise StageEmbargoError(
                    f"{symbol} funding row at {stamp} ms lies outside the requested "
                    f"range [{start_ms}, {end_ms}]"
                )
        self.rows_read += len(rows)
        return rows

    def probe_published_day(
        self, market: Any, symbol: str, day_start_ms: int
    ) -> bool | None:
        """Whether a daily archive publishes the first hour of that day (None: cannot say)."""
        probe = getattr(self._inner, "probe_published_day", None)
        if probe is None:
            return None
        self._check(market, symbol, day_start_ms, day_start_ms + HOUR_MS)
        if day_start_ms + 24 * HOUR_MS > self._end_ms:
            raise StageEmbargoError("a probed day lies outside the stage read range")
        return bool(probe(market, symbol, day_start_ms))


@dataclass(frozen=True, slots=True)
class PerpHourly:
    """One perp's hourly series on the stage's hourly grid (hour ``i`` opens at start + i).

    ``present`` marks a published row; ``halted`` marks a missing hour that lies inside a
    declared exchange-wide halt and between the first and the last published hour. The
    price and volume arrays are NaN where nothing is published. ``funding_ms`` and
    ``funding_rate`` are the settled funding events (strictly increasing times).
    """

    present: np.ndarray
    halted: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    funding_ms: np.ndarray
    funding_rate: np.ndarray

    def __post_init__(self) -> None:
        present = np.asarray(self.present, dtype=np.bool_)
        halted = np.asarray(self.halted, dtype=np.bool_)
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in ("open", "high", "low", "close", "quote_volume")
        }
        n = present.size
        if present.ndim != 1 or halted.shape != (n,):
            raise StageDataError("present and halted must be aligned 1-d arrays")
        if any(arr.shape != (n,) for arr in arrays.values()):
            raise StageDataError("hourly arrays must align with the hourly grid")
        if np.any(present & halted):
            raise StageDataError("an hour cannot be both published and halted")
        for name, arr in arrays.items():
            values = arr[present]
            floor_ok = values >= 0.0 if name == "quote_volume" else values > 0.0
            if not np.isfinite(values).all() or not floor_ok.all():
                raise StageDataError(
                    f"published {name} values must be finite and valid"
                )
        if halted.any():
            first = int(np.argmax(present)) if present.any() else n
            last = n - 1 - int(np.argmax(present[::-1])) if present.any() else -1
            inside = np.flatnonzero(halted)
            if inside.size and (inside[0] <= first or inside[-1] >= last):
                raise StageDataError(
                    "a halted hour must lie between the first and the last published hour"
                )
        funding_ms = np.asarray(self.funding_ms, dtype=np.int64)
        funding_rate = np.asarray(self.funding_rate, dtype=np.float64)
        if funding_ms.shape != funding_rate.shape or funding_ms.ndim != 1:
            raise StageDataError("funding times and rates must align")
        if np.any(np.diff(funding_ms) <= 0) or not np.isfinite(funding_rate).all():
            raise StageDataError("funding events must be increasing and finite")
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "halted", halted)
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "funding_ms", funding_ms)
        object.__setattr__(self, "funding_rate", funding_rate)

    @property
    def n_hours(self) -> int:
        return int(self.present.size)

    def universe_series(self) -> UniverseSeries:
        """What the universe rules read: completeness, hourly quote volume, funding times."""
        return UniverseSeries(
            complete=self.present | self.halted,
            quote_volume=np.where(self.present, self.quote_volume, 0.0),
            funding_ms=self.funding_ms,
        )


@dataclass(frozen=True, slots=True)
class _Bars4h:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding: np.ndarray
    funding_count: np.ndarray
    active: np.ndarray
    tradable: np.ndarray


def _aggregate_4h(leg: PerpHourly, *, clock_ms: np.ndarray, grid_ms: int) -> _Bars4h:
    """Hourly rows to 4h bars (see the module docstring for the bar states)."""
    n_hours = leg.n_hours
    if n_hours == 0 or n_hours % protocol.BAR_HOURS:
        raise StageDataError("hourly rows must fill whole 4h bars")
    n_rows = n_hours // protocol.BAR_HOURS
    if clock_ms.size != n_rows:
        raise StageDataError("hourly grid and 4h clock disagree")

    def grid(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64).reshape(-1, protocol.BAR_HOURS)

    pres = leg.present.reshape(-1, protocol.BAR_HOURS)
    hal = leg.halted.reshape(-1, protocol.BAR_HOURS)
    full = pres.all(axis=1)
    halted_bar = (pres | hal).all(axis=1) & hal.any(axis=1)
    active = full.copy()
    for row in np.flatnonzero(halted_bar):
        if row > 0 and active[row - 1]:
            active[row] = True
    open_ = grid(leg.open)[:, 0]
    high = np.where(pres, grid(leg.high), -np.inf).max(axis=1)
    low = np.where(pres, grid(leg.low), np.inf).min(axis=1)
    close = grid(leg.close)[:, -1]
    volume = np.where(pres, grid(leg.quote_volume), 0.0).sum(axis=1)
    tradable = full.copy()
    close = np.where(full, close, np.nan)
    open_ = np.where(full, open_, np.nan)
    high = np.where(full, high, np.nan)
    low = np.where(full, low, np.nan)
    volume = np.where(full, volume, 0.0)
    # a halted bar carries the preceding close and no volume
    carried = halted_bar & active
    for row in np.flatnonzero(carried):
        previous = float(close[row - 1])
        open_[row] = high[row] = low[row] = close[row] = previous
        volume[row] = 0.0
    # funding events: stamped on the first 4h close at or after the event time
    funding = np.zeros(n_rows, dtype=np.float64)
    count = np.zeros(n_rows, dtype=np.int32)
    if leg.funding_ms.size:
        index = np.searchsorted(clock_ms, leg.funding_ms, side="left")
        keep = (index < n_rows) & (
            leg.funding_ms > clock_ms[np.minimum(index, n_rows - 1)] - grid_ms
        )
        for row, rate in zip(index[keep], leg.funding_rate[keep], strict=True):
            funding[row] += rate
            count[row] += 1
    funding = np.where(active & tradable, funding, 0.0)
    count = np.where(active & tradable, count, 0).astype(np.int32)
    return _Bars4h(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding=funding,
        funding_count=count,
        active=active,
        tradable=tradable,
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


@dataclass(frozen=True, slots=True)
class StageData:
    """The assembled stage: dataset, universe membership by row and the universe by month."""

    dataset: MarketDataset
    #: ``(rows, legs)``: True where the leg belongs to the universe of the row's month
    member: np.ndarray
    universe: Mapping[datetime, MonthUniverse]
    static_universe: bool

    def member_sha256(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(self.member, dtype=np.bool_).tobytes()
        ).hexdigest()


def assemble_stage_dataset(
    window: StageWindow,
    legs: Mapping[str, PerpHourly],
    *,
    member: np.ndarray,
    universe: Mapping[datetime, MonthUniverse],
    static_universe: bool,
    rules: Mapping[str, tierd.LegRules],
    limit_close: datetime,
    data_authorization_url: str | None = None,
    listing: Mapping[str, object] | None = None,
) -> StageData:
    """Assemble the 4h, perp-only dataset of a stage from complete hourly series.

    ``legs`` maps each leg symbol (sorted into the dataset order) to its hourly series on
    the grid ``[load_start, limit_close)``. ``member`` is the ``(rows, legs)`` universe
    mask in the same sorted order, ``rules`` the per-symbol lot size and minimum notional.
    """
    names = tuple(sorted(legs))
    if not names or len(set(names)) != len(names):
        raise StageDataError("a stage needs at least one unique perp leg")
    if set(rules) != set(names):
        raise StageDataError("rules must cover exactly the legs")
    clock = stage_clock(window, limit_close)
    limit = _utc(limit_close, field="limit_close")
    needs_authorization = window.stage in protocol.AUTHORIZED_STAGES
    if (
        needs_authorization
        and data_authorization_url != protocol.DATA_AUTHORIZATION_URL
    ):
        raise StageEmbargoError(
            f"{window.stage} reads need the owner authorization "
            f"{protocol.DATA_AUTHORIZATION_URL}"
        )
    if limit > DEFAULT_EMBARGO and window.stage != protocol.STAGE_FORWARD:
        raise StageEmbargoError(
            "rows at or after the default embargo need the FORWARD stage"
        )
    n_rows, n_legs = len(clock), len(names)
    shape = (n_rows, n_legs)
    mask = np.asarray(member, dtype=np.bool_)
    if mask.shape != shape:
        raise StageDataError("the membership mask must be (rows, legs)")
    clock_ms = clock.astype("datetime64[ms]").astype(np.int64)
    open_ = np.full(shape, UNAVAILABLE_PRICE)
    high = np.full(shape, UNAVAILABLE_PRICE)
    low = np.full(shape, UNAVAILABLE_PRICE)
    close = np.full(shape, UNAVAILABLE_PRICE)
    volume = np.zeros(shape)
    funding = np.zeros(shape)
    funding_count = np.zeros(shape, dtype=np.int32)
    tradable = np.zeros(shape, dtype=np.bool_)
    active = np.zeros(shape, dtype=np.bool_)
    lot = np.empty(n_legs)
    minimum = np.empty(n_legs)
    coverage: dict[str, dict[str, object]] = {}
    for column, symbol in enumerate(names):
        leg = legs[symbol]
        if leg.n_hours != n_rows * protocol.BAR_HOURS:
            raise StageDataError(
                f"{symbol}: the hourly grid must run to the read limit {limit.isoformat()}"
            )
        bars = _aggregate_4h(leg, clock_ms=clock_ms, grid_ms=BAR_MS)
        rows = bars.active
        open_[rows, column] = bars.open[rows]
        high[rows, column] = bars.high[rows]
        low[rows, column] = bars.low[rows]
        close[rows, column] = bars.close[rows]
        volume[:, column] = np.where(rows, bars.volume, 0.0)
        funding[:, column] = bars.funding
        funding_count[:, column] = bars.funding_count
        tradable[:, column] = bars.tradable & rows
        active[:, column] = rows
        # the first inactive bar after an active one is where an unclosed position is
        # settled: it opens at the last published close (no invented price)
        ended = np.flatnonzero(rows[:-1] & ~rows[1:])
        for last in ended:
            for field_array in (open_, high, low, close):
                field_array[last + 1, column] = close[last, column]
        rule = rules[symbol]
        lot[column], minimum[column] = rule.lot_size, rule.min_notional
        first = int(np.argmax(rows)) if rows.any() else None
        coverage[symbol] = {
            "active_rows": int(rows.sum()),
            "first_active_row": first,
            "ended_rows": [int(i) for i in ended],
        }
    zero = np.zeros(shape)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=names,
        timestamps=clock,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((n_rows, 1), dtype=np.float32),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("xs_lowvol_funding_no_features",),
        global_feature_names=("xs_lowvol_funding_no_features",),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        volume_units=tuple(VolumeUnit.QUOTE_NOTIONAL for _ in range(n_legs)),
        funding_rate=funding,
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
        borrow_available=np.ones(shape, dtype=np.bool_),
        contract_multipliers=np.ones(n_legs, dtype=np.float64),
        mark_price=close,
    )
    identity: dict[str, object] = {
        "source_contract": STAGE_DATA_CONTRACT,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": window.stage,
        "window": window.to_dict(),
        "read_limit_close": limit.isoformat(),
        "legs": list(names),
        "bar_hours": protocol.BAR_HOURS,
        "aggregation": (
            "hourly_to_4h_open_first_close_last_high_max_low_min_volume_sum;"
            "funding_summed_over_the_four_hours;tradable_only_if_all_hours_tradable"
        ),
        "timestamps": "bar_close",
        "perp_series": "strict_declared_halts_only",
        "unavailable_rows": (
            f"asset_inactive_not_tradable_constant_sentinel_price_{UNAVAILABLE_PRICE}"
        ),
        "first_inactive_bar_open": "the last published close (settlement price)",
        "coverage": coverage,
        "static_universe": static_universe,
        "universe": {
            start.isoformat(): month.to_dict()
            for start, month in sorted(universe.items())
        },
        "membership_sha256": hashlib.sha256(
            np.ascontiguousarray(mask).tobytes()
        ).hexdigest(),
        "dataset_costs": "zero_scenario_costs_additive_in_ExecutionCostConfig",
        "rules_assumption": "static_exchange_rule_snapshot_not_point_in_time",
        "borrow_assumption": "perp_native_short_zero_borrow_rate",
        "mark_price_limitation": "bar_close_proxy",
        "production_eligible": False,
    }
    if listing is not None:
        identity["listing"] = dict(listing)
    if needs_authorization:
        identity["data_authorization_url"] = data_authorization_url
    return StageData(
        dataset=dataset.with_content_identity(identity),
        member=mask,
        universe=dict(universe),
        static_universe=static_universe,
    )


def _stamp_to_datetime(stamp: np.datetime64) -> datetime:
    ns = int(stamp.astype("datetime64[ns]").astype(np.int64))
    return datetime.fromtimestamp(ns / 1_000_000_000, UTC)


def validate_stage_dataset(stage: StageData, window: StageWindow) -> int:
    """Check an assembled stage against its window; return the index of its last row.

    The clock must be the stage clock from the first row, regular, and end at or before the
    stage's last row; the legs are sorted and unique; an inactive leg is never tradable and
    carries no volume; the membership mask matches the dataset; and a dataset that reaches
    the default embargo must carry the owner authorization URL in its recorded identity.
    Computes nothing economic.
    """
    dataset = stage.dataset
    legs = tuple(dataset.symbols)
    if not legs or legs != tuple(sorted(set(legs))):
        raise StageDataError("dataset legs must be unique and sorted")
    ts = dataset.timestamps.astype("datetime64[ns]")
    if ts.size < 2:
        raise StageDataError("dataset is too short")
    last = _stamp_to_datetime(ts[-1])
    if last > window.last_row_close:
        raise StageEmbargoError("dataset reaches past the stage's last row")
    if not np.array_equal(ts, stage_clock(window, last)):
        raise StageDataError("dataset clock is not the stage clock from its first row")
    if stage.member.shape != (dataset.n_bars, dataset.n_symbols):
        raise StageDataError("the membership mask does not match the dataset")
    volume = np.asarray(dataset.volume)
    active = dataset.resolved_array("asset_active")
    if np.asarray(dataset.tradable)[~active].any() or volume[~active].any():
        raise StageDataError("an inactive leg is tradable or carries volume")
    payload = dataset.identity_payload_json or ""
    if (
        window.stage in protocol.AUTHORIZED_STAGES
        and protocol.DATA_AUTHORIZATION_URL not in payload
    ):
        raise StageEmbargoError(
            f"a {window.stage} dataset carries no owner authorization in its identity"
        )
    if last > DEFAULT_EMBARGO and window.stage != protocol.STAGE_FORWARD:
        raise StageEmbargoError("only the FORWARD stage may reach the default embargo")
    return int(ts.size - 1)


def stage_summary(stage: StageData, window: StageWindow) -> dict[str, object]:
    """Shape, range and availability counts only; computes no return or signal."""
    dataset = stage.dataset
    last_index = validate_stage_dataset(stage, window)
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
        "membership_sha256": stage.member_sha256(),
        "months": {
            start.isoformat(): month.to_dict()
            for start, month in sorted(stage.universe.items())
        },
        "per_leg": per_leg,
    }


__all__ = [
    "BAR",
    "BAR_MS",
    "BASE_INTERVAL",
    "DEFAULT_EMBARGO",
    "HOUR_MS",
    "STAGE_DATA_CONTRACT",
    "PerpHourly",
    "StageData",
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
