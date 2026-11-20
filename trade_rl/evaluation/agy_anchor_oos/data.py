"""Stage-window 4h dataset assembly for ``agy_anchor_realistic_v1``, without any economics.

This module turns checksum-verified hourly Binance *spot and USD-M perpetual* series (and
the taker-buy quote volume of the same 1h klines) into the 4h, 11-pair, 22-leg alternating
``MarketDataset`` the anchor and its controls trade on, for the DEV, BACKCAST_2020 and
FORWARD windows. It owns no network, database, replay or return computation: the
``tools.agy_anchor_oos`` wiring reads the rows; this module only guards, aggregates and
assembles them and refuses everything that is not complete.

Carried outside the dataset: the Tier D assembler (bound by a research packet) drops
taker-buy volume, so it is aggregated here with the same 4h rule and returned beside the
dataset as a ``(T, legs)`` array, bound into the dataset identity by its SHA-256.

Contract (fail closed everywhere):

* **Leg layout** is lowturn's/Tier D's. A market that did not exist yet (spot or perp,
  separately) is unavailable before its first complete 4h bar: inactive, not tradable,
  information unavailable, one constant sentinel price (no price is ever invented, back
  filled or forward filled).
* **Pair activation.** A pair is available to the anchor signal only from the first row at
  which both its spot and its perp are listed plus ``PAIR_WARMUP_BARS`` (or from row 0 for a
  pair listed at the first row). :func:`build_anchor_inputs` applies this and rescales the
  prices of a late pair by one common factor so the sentinel-to-price step is no return.
* **Embargo.** Only FORWARD may read past the default (Tier D) embargo, and only on the exact
  owner authorization URL, never past the stage's last row nor the latest closed bar.
* **Halts** are ``protocol.stage_halt_intervals()`` for spot; perpetual series are strict.
* **Exchange rules** are one static snapshot, not point-in-time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.agy_anchor_oos import protocol
from trade_rl.evaluation.core7_mtf_study import tierd
from trade_rl.evaluation.trend_oos.data import (
    DEFAULT_EMBARGO,
    UNAVAILABLE_PRICE,
    StageDataError,
    StageEmbargoError,
    latest_closed_bar,
)
from trade_rl.strategies.agy_anchor.inputs import AnchorInputs, from_market_dataset

HOUR_MS: Final = tierd.HOUR_MS
BAR_MS: Final = tierd.BAR_MS
BAR: Final = timedelta(hours=protocol.BAR_HOURS)
BASE_INTERVAL: Final = tierd.BASE_INTERVAL
STAGE_DATA_CONTRACT: Final = "agy_anchor_oos_stage_v1"
_SPOT_MARKETS: Final = ("spot",)
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
    """One stage's clock; all instants are bar-close timestamps of dataset rows."""

    stage: str
    load_start: datetime
    first_decision: datetime
    last_decision_bar: datetime
    settles: datetime
    warmup_bars: int = protocol.STAGE_WARMUP_BARS

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
        return self.row_index(self.first_decision)

    @property
    def last_row_close(self) -> datetime:
        """Close of the last row: two bars after the last decision bar."""
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


def read_limit(
    window: StageWindow,
    *,
    now: datetime,
    authorization_url: str | None = None,
    through: datetime | None = None,
) -> datetime:
    """The last row close that may be read: the exclusive limit of the hourly opens.

    DEV and BACKCAST_2020 need no authorization but are refused while their window is not
    over. FORWARD reads exist only on the path bound to ``protocol.DATA_AUTHORIZATION_URL``.
    The limit never exceeds the stage's last row or the latest fully closed bar; ``through``
    (a smaller 4h-grid instant) may only lower it.
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

    def read_funding(
        self, market: Any, symbol: str, start_ms: int, end_ms: int
    ) -> Sequence[Any]: ...


def _market_name(market: Any) -> str:
    return str(getattr(market, "value", market)).lower()


class StageRowSource:
    """Refuse reads outside the stage: before the read and on every returned row.

    Hourly klines of both markets and perpetual funding of roster symbols from the load
    start to ``limit_close`` are served. ``limit_close`` is the exclusive bound of hourly
    opens (the close of the last admissible 4h bar); funding rows may not be later.
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

    def _check(self, market: Any, symbol: str, start_ms: int, end_ms: int) -> str:
        name = _market_name(market)
        if name not in _SPOT_MARKETS and name not in _PERP_MARKETS:
            raise StageDataError(f"market {name!r} is not admitted")
        if symbol not in protocol.ROSTER:
            raise StageDataError(f"{symbol} is not in the roster")
        if end_ms > self._end_ms:
            raise StageEmbargoError(
                f"request ends at {end_ms} ms, past the read limit {self._end_ms} ms; "
                "refused before reading"
            )
        if start_ms < self._start_ms:
            raise StageEmbargoError("request starts before the stage load start")
        if start_ms >= end_ms:
            raise ValueError("request range is empty")
        return name

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
        if self._check(market, symbol, start_ms, end_ms) in _SPOT_MARKETS:
            raise StageDataError("spot carries no funding")
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
class HourlyLeg:
    """A complete hourly series of one leg and its hourly taker-buy quote volume."""

    series: RawMarketSeries
    taker_buy_quote: np.ndarray

    def __post_init__(self) -> None:
        taker = np.asarray(self.taker_buy_quote, dtype=np.float64)
        if taker.shape != (len(self.series.timestamps),):
            raise StageDataError("taker-buy volume must align with the hourly series")
        if not np.isfinite(taker).all() or np.any(taker < 0.0):
            raise StageDataError("taker-buy volume must be finite and non-negative")
        object.__setattr__(self, "taker_buy_quote", taker)


@dataclass(frozen=True, slots=True)
class _Bars4h:
    close_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    taker: np.ndarray
    funding: np.ndarray
    funding_count: np.ndarray
    tradable: np.ndarray


def _aggregate_4h(leg: HourlyLeg, *, name: str, perp: bool) -> _Bars4h:
    """Hourly (close-stamped) rows to whole 4h bars; the series must be complete."""
    series = leg.series
    stamps = np.asarray(series.timestamps).astype("datetime64[ms]").astype(np.int64)
    n_hours = int(stamps.size)
    if n_hours == 0 or n_hours % protocol.BAR_HOURS:
        raise StageDataError(f"{name}: hourly rows must fill whole 4h bars")
    if np.any(np.diff(stamps) != HOUR_MS):
        raise StageDataError(f"{name}: hourly clock must be complete and regular")
    if (int(stamps[0]) - HOUR_MS) % BAR_MS:
        raise StageDataError(f"{name}: the first bar must open on the 4h grid")
    counts = series.funding_event_count
    if perp:
        if counts is None:
            raise StageDataError(f"{name}: funding event counts are required")
    elif np.any(np.asarray(counts)) or np.any(series.funding_rate):
        raise StageDataError(f"{name}: spot must not carry funding")

    def grid(values: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(-1, protocol.BAR_HOURS)

    tradable_h = grid(series.tradable, np.bool_)
    taker_h = np.where(tradable_h, grid(leg.taker_buy_quote), 0.0)
    stamps4 = stamps.reshape(-1, protocol.BAR_HOURS)
    return _Bars4h(
        close_ns=(stamps4[:, -1] * 1_000_000).astype("datetime64[ns]"),
        open=grid(series.open)[:, 0].copy(),
        high=grid(series.high).max(axis=1),
        low=grid(series.low).min(axis=1),
        close=grid(series.close)[:, -1].copy(),
        volume=grid(series.volume).sum(axis=1),
        taker=taker_h.sum(axis=1),
        funding=(
            grid(series.funding_rate).sum(axis=1)
            if perp
            else np.zeros(stamps4.shape[0])
        ),
        funding_count=(
            grid(counts, np.int64).sum(axis=1).astype(np.int32)
            if perp and counts is not None
            else np.zeros(stamps4.shape[0], dtype=np.int32)
        ),
        tradable=tradable_h.all(axis=1),
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
    """The assembled stage: dataset, taker-buy array and each pair's activation row."""

    dataset: MarketDataset
    taker_buy_quote_volume: np.ndarray
    #: first row at which both markets of the pair are listed; ``None`` if never both
    activation_rows: tuple[int | None, ...]

    def with_dataset(self, dataset: MarketDataset, taker: np.ndarray) -> StageData:
        return replace(self, dataset=dataset, taker_buy_quote_volume=taker)


def assemble_stage_dataset(
    window: StageWindow,
    spot: Mapping[str, HourlyLeg | None],
    perp: Mapping[str, HourlyLeg | None],
    *,
    rules: Mapping[str, tuple[tierd.LegRules, tierd.LegRules]],
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    data_authorization_url: str | None = None,
) -> StageData:
    """Assemble the 4h, 11-pair, 22-leg alternating dataset and taker-buy array of a stage.

    ``spot`` / ``perp`` map each roster symbol to its complete hourly series from its first
    4h-aligned listed hour to ``limit_close``, or ``None`` if the market has no data in the
    window. ``rules`` maps each symbol to its ``(spot, perp)`` rules.
    """
    roster = protocol.ROSTER
    if halt_intervals is None:
        halt_intervals = protocol.stage_halt_intervals()
    for label, mapping in (("spot", spot), ("perp", perp)):
        if set(mapping) != set(roster) or len(mapping) != len(roster):
            raise StageDataError(
                f"{label} must cover exactly the roster "
                f"(missing={sorted(set(roster) - set(mapping))}, "
                f"extra={sorted(set(mapping) - set(roster))})"
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
    taker = np.zeros(shape)
    funding = np.zeros(shape)
    funding_count = np.zeros(shape, dtype=np.int32)
    tradable = np.zeros(shape, dtype=np.bool_)
    active = np.zeros(shape, dtype=np.bool_)
    lot = np.empty(n_legs)
    minimum = np.empty(n_legs)
    listing: dict[str, dict[str, object]] = {}
    first_rows: dict[tuple[str, int], int] = {}
    for pair, symbol in enumerate(roster):
        spot_rules, perp_rules = rules[symbol]
        lot[2 * pair], lot[2 * pair + 1] = spot_rules.lot_size, perp_rules.lot_size
        minimum[2 * pair] = spot_rules.min_notional
        minimum[2 * pair + 1] = perp_rules.min_notional
        entry: dict[str, object] = {}
        for offset, (market, leg) in enumerate(
            (("spot", spot[symbol]), ("perp", perp[symbol]))
        ):
            if leg is None:
                entry[market] = {"first_listed_row": None, "unavailable_rows": n_rows}
                continue
            is_perp = offset == 1
            bars = _aggregate_4h(leg, name=f"{symbol}:{market}", perp=is_perp)
            first_ns = int(bars.close_ns[0].astype("datetime64[ns]").astype(np.int64))
            first_row = int(np.searchsorted(clock_ns, first_ns))
            stop = first_row + len(bars.close_ns)
            if first_row >= n_rows or clock_ns[first_row] != first_ns:
                raise StageDataError(f"{symbol}:{market} starts off the stage clock")
            if stop != n_rows or not np.array_equal(
                clock_ns[first_row:],
                bars.close_ns.astype("datetime64[ns]").astype(np.int64),
            ):
                raise StageDataError(
                    f"{symbol}:{market} must run without a gap to the read limit "
                    f"{limit.isoformat()}"
                )
            if is_perp:
                tierd._check_funding_calendar(
                    tierd._Bars4h(
                        close_ns=bars.close_ns,
                        open=bars.open,
                        high=bars.high,
                        low=bars.low,
                        close=bars.close,
                        volume=bars.volume,
                        funding=bars.funding,
                        funding_count=bars.funding_count,
                        tradable=bars.tradable,
                    ),
                    leg=f"{symbol}:perp",
                )
                if not np.all(bars.tradable):
                    raise StageDataError(
                        f"{symbol}:perp has no halt calendar and must be tradable"
                    )
            else:
                tierd._check_halts_masked(
                    bars.close_ns, bars.tradable, halt_intervals, symbol=symbol
                )
            column = 2 * pair + offset
            rows = slice(first_row, stop)
            open_[rows, column] = bars.open
            high[rows, column] = bars.high
            low[rows, column] = bars.low
            close[rows, column] = bars.close
            volume[rows, column] = bars.volume
            taker[rows, column] = bars.taker
            funding[rows, column] = bars.funding
            funding_count[rows, column] = bars.funding_count
            tradable[rows, column] = bars.tradable
            active[rows, column] = True
            first_rows[(symbol, offset)] = first_row
            entry[market] = {
                "first_listed_row": first_row,
                "unavailable_rows": first_row,
            }
        listing[symbol] = entry

    anchor = protocol.ROSTER[0]
    if first_rows.get((anchor, 0)) != 0 or first_rows.get((anchor, 1)) != 0:
        raise StageDataError(
            f"{anchor} spot and perp must be listed from the stage load start: an "
            "all-unavailable dataset means the source is unreachable or misconfigured"
        )
    activation: list[int | None] = []
    for symbol in roster:
        both = (first_rows.get((symbol, 0)), first_rows.get((symbol, 1)))
        activation.append(None if None in both else max(both))  # type: ignore[type-var]
    zero = np.zeros(shape)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=protocol.LEG_ROSTER,
        timestamps=clock,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((n_rows, 1), dtype=np.float32),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("agy_anchor_oos_no_features",),
        global_feature_names=("agy_anchor_oos_no_features",),
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
            "funding_summed_over_the_four_hours;tradable_only_if_all_hours_tradable"
        ),
        "timestamps": "bar_close",
        "spot_halts": [
            [_utc(lo, field="halt").isoformat(), _utc(hi, field="halt").isoformat()]
            for lo, hi in halt_intervals
        ],
        "halted_bar_marks": "preceding_published_close_non_tradable_zero_volume",
        "perp_series": "strict_no_halt_calendar",
        "unavailable_rows": (
            f"asset_inactive_not_tradable_constant_sentinel_price_{UNAVAILABLE_PRICE}"
        ),
        "listing": listing,
        "activation_rows": list(activation),
        "taker_buy_quote_volume_sha256": hashlib.sha256(
            np.ascontiguousarray(taker).tobytes()
        ).hexdigest(),
        "dataset_costs": "zero_scenario_costs_additive_in_ExecutionCostConfig",
        "rules_assumption": "static_exchange_rule_snapshot_not_point_in_time",
        "borrow_assumption": "spot_not_borrowable_perp_native_short_zero_borrow_rate",
        "mark_price_limitation": "bar_close_proxy",
        "production_eligible": False,
    }
    if needs_authorization:
        identity["data_authorization_url"] = data_authorization_url
    return StageData(
        dataset=dataset.with_content_identity(identity),
        taker_buy_quote_volume=taker,
        activation_rows=tuple(activation),
    )


def validate_stage_dataset(stage: StageData, window: StageWindow) -> int:
    """Check an assembled stage against its window; return the index of its last row.

    The clock must be the stage clock from the first row, regular, and end at or before the
    stage's last row; the 22-leg layout is exact; the taker-buy array matches the dataset and
    never exceeds a leg's quote volume; an inactive leg is never tradable and carries no
    volume; and a dataset that reaches the default embargo must carry the owner
    authorization URL in its recorded identity. Computes nothing economic.
    """
    dataset = stage.dataset
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
    taker = np.asarray(stage.taker_buy_quote_volume)
    if taker.shape != (dataset.n_bars, dataset.n_symbols):
        raise StageDataError("taker-buy array does not match the dataset")
    volume = np.asarray(dataset.volume)
    if np.any(taker < 0.0) or np.any(taker > volume * (1.0 + 1e-9) + 1e-9):
        raise StageDataError("taker-buy volume exceeds a leg's quote volume")
    active = dataset.resolved_array("asset_active")
    if np.asarray(dataset.tradable)[~active].any() or volume[~active].any():
        raise StageDataError("an inactive leg is tradable or carries volume")
    if len(stage.activation_rows) != len(protocol.ROSTER):
        raise StageDataError("activation rows must cover the roster")
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


def available_from_rows(
    activation_rows: Sequence[int | None], *, n_rows: int
) -> tuple[int, ...]:
    """First row at which each pair is available to the anchor signal (Amendment 1).

    Row 0 for a pair listed at the first row (the stage warm-up covers it); otherwise the
    activation row plus ``PAIR_WARMUP_BARS``; ``n_rows`` (never) if a market is absent.
    """
    out: list[int] = []
    for row in activation_rows:
        if row is None:
            out.append(n_rows)
        elif row == 0:
            out.append(0)
        else:
            out.append(min(row + protocol.PAIR_WARMUP_BARS, n_rows))
    return tuple(out)


def build_anchor_inputs(stage: StageData) -> AnchorInputs:
    """The anchor's klines + funding + taker bundle for the stage (causal, Amendment 1).

    Validity is the leg's tradability and the pair's availability. Before a pair's
    activation both of its closes are the constant 1.0 and afterwards both are divided by
    the pair's spot close at the activation row, so the step from the sentinel to the first
    price is not a return and the basis is unchanged. A pair listed at the first row is not
    touched. Everything reads only the row itself and the activation row, which is not in
    the future of any row that uses it.
    """
    dataset = stage.dataset
    base = from_market_dataset(
        dataset, taker_buy_quote_volume=stage.taker_buy_quote_volume
    )
    n_rows = dataset.n_bars
    n_pairs = len(protocol.ROSTER)
    avail = available_from_rows(stage.activation_rows, n_rows=n_rows)
    rows = np.arange(n_rows)[:, None]
    gate = rows >= np.asarray(avail)[None, :]
    spot_close = np.array(base.spot_close, copy=True)
    perp_close = np.array(base.perp_close, copy=True)
    for pair in range(n_pairs):
        row = stage.activation_rows[pair]
        if row is None:
            spot_close[:, pair] = 1.0
            perp_close[:, pair] = 1.0
        elif row > 0:
            factor = spot_close[row, pair]
            spot_close[:row, pair] = 1.0
            perp_close[:row, pair] = 1.0
            spot_close[row:, pair] = spot_close[row:, pair] / factor
            perp_close[row:, pair] = perp_close[row:, pair] / factor
    spot_valid = np.asarray(base.spot_valid) & gate
    perp_valid = np.asarray(base.perp_valid) & gate
    return replace(
        base,
        spot_close=spot_close,
        perp_close=perp_close,
        spot_valid=spot_valid,
        perp_valid=perp_valid,
    )


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
        "activation_rows": list(stage.activation_rows),
        "per_leg": per_leg,
    }


__all__ = [
    "BAR",
    "BAR_MS",
    "BASE_INTERVAL",
    "HOUR_MS",
    "STAGE_DATA_CONTRACT",
    "HourlyLeg",
    "StageData",
    "StageRowSource",
    "StageWindow",
    "assemble_stage_dataset",
    "available_from_rows",
    "build_anchor_inputs",
    "frozen_window",
    "read_limit",
    "stage_clock",
    "stage_summary",
    "validate_stage_dataset",
]
