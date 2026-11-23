"""Driver for the out-of-sample test ``xs_lowvol_funding_v1``.

``python -m tools.xs_lowvol_funding plan --stage DEV|DISJOINT|FORWARD``
``python -m tools.xs_lowvol_funding coverage --stage DISJOINT ...``
``python -m tools.xs_lowvol_funding halts-from-coverage --coverage coverage.json``
``python -m tools.xs_lowvol_funding notices --output delisting_notices.json --allow-network``
``python -m tools.xs_lowvol_funding run --stage DEV``
``python -m tools.xs_lowvol_funding run --stage DISJOINT --dev-result DEV/dev-result.json``
``python -m tools.xs_lowvol_funding run --stage FORWARD --interim | --final``

The wiring mirrors :mod:`tools.agy_anchor_oos` (whose embargo policy and daily-fallback
source, and :mod:`tools.trend_oos`' fetchers, it imports) with USD-M perpetual 1h klines
and funding only: no spot data is read.

``plan`` prints the frozen protocol, its digest and the windows; it loads no data.

``notices`` (Amendment 5) records the public Binance futures delisting / migration / swap
notices (symbol, release time, article code) from the public CMS API into the file whose
SHA-256 is pinned as ``protocol.DELISTING_NOTICES_SHA256``. It is result-blind (no market
data), write-once, and uses the network only with ``--allow-network``. ``run`` verifies that
pin before any other access and closes a delisted name only from those notices.

``coverage`` is the result-blind data step before any run: it acquires the stage's archives
through the same source path and writes ``coverage.json`` (listing start, missing hour
ranges after the daily fallback, fills, placeholders, checksums, funding counts, the
perpetual hours missing for every symbol listed then). It builds no universe, signal, replay
or return and never refuses on a gap.

``run`` authenticates the G4 relay (owner author, this protocol's id, digest and the exact
code commit quoted, unedited between two fetches), requires a clean checkout and a new
output directory, builds the stage dataset, records a data receipt and writes every cell
before the result. The run order is enforced: DISJOINT needs a DEV result without an
invalid cell, FORWARD needs a DISJOINT decision labelled ``XS_SUPPORTED`` of the same
digest. ``--final`` is refused before 2027-05-08T00:00Z; ``--interim`` writes a dated
``INTERIM_NOT_A_DECISION`` report. DISJOINT and FORWARD reads are bound to the owner scope
relay of Issue #810 (comment 7040247247). The driver authorizes nothing beyond data reads
and the evaluation: no orders, no API keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from tools import agy_anchor_oos as agy_tool
from tools import core7_g4_comment as g4
from tools import core7_mtf_tierd as tierd_driver
from tools import trend_oos as base
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.core7_mtf_study.tierd import LegRules
from trade_rl.evaluation.trend_oos.data import StageEmbargoError
from trade_rl.evaluation.xs_lowvol_funding import (
    data,
    notices,
    protocol,
    runner,
    universe,
)
from trade_rl.integrations.binance import vision_funding_url
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryScreenHalt,
)
from trade_rl.integrations.market_store.carry_source import (
    CarryRowSource,
    _check_order,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.plan import (
    IngestionPlan,
    _month_start,
    _next_month,
    plan_ingestion,
)
from trade_rl.integrations.market_store.rules import (
    ExchangeInfoError,
    parse_symbol_rules,
)

EXIT_OK = base.EXIT_OK
EXIT_REFUSED = base.EXIT_REFUSED
EXIT_HALTED = base.EXIT_HALTED
EXIT_INVALID = base.EXIT_INVALID
EXIT_UNAUTHENTICATED = base.EXIT_UNAUTHENTICATED
COVERAGE_NAME = "coverage.json"
COVERAGE_SCHEMA = "xs_lowvol_funding_coverage_v1"
DATA_RECEIPT_NAME = "xs-lowvol-funding-data-receipt.json"
RUN_RECEIPT_NAME = "xs-lowvol-funding-run-receipt.json"
DATA_RECEIPT_SCHEMA = "xs_lowvol_funding_data_receipt_v1"
RUN_RECEIPT_SCHEMA = "xs_lowvol_funding_run_receipt_v1"
RULES_NOTE = (
    "exchange rules are one static retrieval-time snapshot applied to every row of the "
    "stage; they are not point-in-time; a symbol absent from it uses the fallback rules"
)
_MAX_REPORTED_GAP_RUNS = 25
_MARKET = StoreMarket.USDS_M

GitState = base.GitState
RowSourceFactory = base.RowSourceFactory
Clock = base.Clock
PlaceholderLog = list[tuple[str, int, int, str]]

_iso = base._iso
_month_chunks = base._month_chunks
_ceil_day = base._ceil_day
_floor_day = base._floor_day
_sha256 = base._sha256
_parse_instant = base._parse_instant
_hour_ranges = base._hour_ranges
_halt_error = base._halt_error
_read_errors = base._read_errors
_ceil_month = agy_tool._ceil_month
embargo_for = agy_tool.embargo_for
funding_boundary = agy_tool.funding_boundary
require_funding_published = agy_tool.require_funding_published
ForwardEmbargoPolicy = agy_tool.AnchorForwardEmbargoPolicy


# ----------------------------------------------------------------------- symbols


def read_symbol_file(path: Path) -> tuple[tuple[str, ...], str]:
    """The FORWARD listing: a JSON list (or newline text) of symbols and its SHA-256.

    Every symbol must pass the spec's exclusions (ASCII USDT perps minus
    stablecoin and gold bases); an excluded or duplicated symbol refuses the file.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("perp_symbols")
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise ValueError("the FORWARD symbol file must be a list of symbols")
    symbols = tuple(parsed)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("the FORWARD symbol file holds duplicates or no symbol")
    bad = sorted(s for s in symbols if not protocol.forward_symbol_allowed(s))
    if bad:
        raise ValueError(f"the FORWARD listing holds excluded symbols: {bad[:10]}")
    return tuple(sorted(symbols)), _sha256(raw)


def stage_symbols(stage: str, forward_symbols: Path | None) -> tuple[str, ...]:
    """The symbols a stage reads: the DEV roster, the frozen candidates or the listing."""
    if stage == protocol.STAGE_DEV:
        return tuple(sorted(protocol.DEV_ROSTER))
    if stage == protocol.STAGE_DISJOINT:
        return tuple(sorted(protocol.load_candidates()))
    if protocol.FORWARD_SYMBOLS_SHA256 is None:
        raise ValueError(
            "FORWARD is refused: FORWARD_SYMBOLS_SHA256 is not pinned (the symbol file "
            "is pinned by a result-blind amendment before any FORWARD read)"
        )
    symbols, digest = read_symbol_file(
        protocol.forward_symbols_path() if forward_symbols is None else forward_symbols
    )
    if digest != protocol.FORWARD_SYMBOLS_SHA256:
        raise ValueError(
            f"the FORWARD symbol file sha256 {digest} differs from the pinned "
            f"{protocol.FORWARD_SYMBOLS_SHA256}"
        )
    return symbols


def load_perp_rules(
    perp_payload: bytes, symbols: Sequence[str]
) -> tuple[dict[str, LegRules], list[str]]:
    """Per-symbol perpetual rules from ``exchangeInfo``; fallback rules for absent symbols."""
    document = json.loads(perp_payload)
    listed = {
        str(item["symbol"])
        for item in document.get("symbols", [])
        if isinstance(item, dict) and item.get("contractType") == "PERPETUAL"
    }
    present = [s for s in symbols if s in listed]
    parsed = (
        {
            rule.symbol: rule
            for rule in parse_symbol_rules(
                perp_payload, market=_MARKET, symbols=present
            )
        }
        if present
        else {}
    )
    rules: dict[str, LegRules] = {}
    fallback: list[str] = []
    for symbol in symbols:
        rule = parsed.get(symbol)
        if rule is None:
            fallback.append(symbol)
            rules[symbol] = LegRules(
                protocol.FALLBACK_LOT_SIZE, protocol.FALLBACK_MIN_NOTIONAL
            )
            continue
        lot, notional = float(rule.lot_size), float(rule.min_notional)
        rules[symbol] = LegRules(
            lot if lot > 0.0 else protocol.FALLBACK_LOT_SIZE,
            notional if notional > 0.0 else protocol.FALLBACK_MIN_NOTIONAL,
        )
    return rules, fallback


# ----------------------------------------------------------------------- loading


def build_plan(
    window: data.StageWindow,
    limit_close: datetime,
    now: datetime,
    policy: EmbargoPolicy,
    symbols: Sequence[str],
) -> IngestionPlan:
    """Hourly USD-M klines and monthly funding of every symbol over the read range.

    The market-store planner plans a funding month only when it ends inside the read range;
    the month that holds the read limit is planned here as well (its rows after the limit
    are discarded by the reader). A month that is not published yet refuses.
    """
    plan_universe = Universe(
        symbols=tuple(symbols), interval=data.BASE_INTERVAL, markets=(_MARKET,)
    )
    plan = plan_ingestion(
        plan_universe,
        start=window.load_start,
        end=_ceil_day(limit_close),
        now=now,
        embargo=policy,
    )
    require_funding_published(limit_close, now)
    needed_end = _ceil_month(limit_close)
    have = {
        (spec.symbol, spec.period_start)
        for spec in plan.specs
        if spec.kind is ArchiveKind.FUNDING_RATE
    }
    extra: list[ArchiveSpec] = []
    for symbol in symbols:
        month = _month_start(window.load_start)
        while month < needed_end:
            following = _next_month(month)
            if (symbol, month) not in have:
                url = vision_funding_url(_MARKET.binance, symbol, month)
                extra.append(
                    ArchiveSpec(
                        kind=ArchiveKind.FUNDING_RATE,
                        market=_MARKET,
                        symbol=symbol,
                        interval=None,
                        period_kind=PeriodKind.MONTHLY,
                        period_start=month,
                        period_end=following,
                        url=url,
                        checksum_url=f"{url}.CHECKSUM",
                    )
                )
            month = following
    for spec in extra:
        policy.check_archive_end(spec.period_end, url=spec.url)
    return replace(plan, specs=(*plan.specs, *extra))


def _probe(reader: Any, symbol: str, day_ms: int) -> bool:
    probe = getattr(reader, "probe_published_day", None)
    return bool(probe(_MARKET, symbol, day_ms)) if callable(probe) else False


def _first_listed_open(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[int | None, int, list[tuple[int, int]]]:
    """``(first published hourly open, unpublished leading months, leading chunks)``.

    Only a *leading* run of unpublished monthly archives (HTTP 404) counts as "not listed
    yet"; an unpublished archive after the first row, an unexplained gap or any other
    failure is handled by the caller or raises.
    """
    leading: list[tuple[int, int]] = []
    unpublished = 0
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(_MARKET, symbol, data.BASE_INTERVAL, lo, hi)
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") == "archive_not_published":
                unpublished += 1
                leading.append((lo, hi))
                continue
            raise
        except CarryIntegrityHalt as error:
            if (error.evidence or {}).get("kind") != "malformed_archive":
                raise
            rows = reader.read_klines(_MARKET, symbol, data.BASE_INTERVAL, lo, hi)
        if rows:
            return int(rows[0].open_time_ms), unpublished, leading
        unpublished += 1
    return None, unpublished, leading


def _read_klines_tolerant(
    reader: CarryRowSource, symbol: str, first_open: int, end_ms: int
) -> tuple[list[Any], list[dict[str, object]]]:
    """Every published row from ``first_open``; an unpublished month is a missing range.

    Integrity faults (checksum, malformed, duplicate) and operational faults still raise.
    """
    rows: list[Any] = []
    unpublished: list[dict[str, object]] = []
    for lo, hi in _month_chunks(first_open, end_ms):
        try:
            rows.extend(reader.read_klines(_MARKET, symbol, data.BASE_INTERVAL, lo, hi))
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") != "archive_not_published":
                raise
            unpublished.append({"from": _iso(lo), "to": _iso(hi)})
    rows.sort(key=lambda row: int(row.open_time_ms))
    return rows, unpublished


def _months_with_rows(opens: Sequence[int]) -> set[tuple[int, int]]:
    """``(year, month)`` of every month that holds a kline bar."""
    months: set[tuple[int, int]] = set()
    for opened in opens:
        moment = datetime.fromtimestamp(opened / 1_000, UTC)
        months.add((moment.year, moment.month))
    return months


# ------------------------------------------------------------- REST funding


REST_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
REST_PAGE_LIMIT = 1_000
_REST_MAX_PAGES = 100
#: One raw REST page: ``(symbol, start_ms, end_ms_inclusive) -> JSON bytes``.
RestFetch = Callable[[str, int, int], bytes]


class RestFundingError(RuntimeError):
    """The REST funding fallback is unavailable or returned an invalid response."""


class RestFundingNotCached(RestFundingError):
    """No cache file and no network: the fallback is not available (fail closed)."""


def binance_rest_fetch(symbol: str, start_ms: int, end_ms: int) -> bytes:
    """One page of the public ``fundingRate`` endpoint (no key); a few bounded retries."""
    url = (
        f"{REST_FUNDING_URL}?symbol={symbol}&startTime={start_ms}"
        f"&endTime={end_ms}&limit={REST_PAGE_LIMIT}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "xs-lowvol-funding"})
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in (418, 429) and error.code < 500:
                break
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
        time.sleep(float(attempt))
    raise RestFundingError(f"{symbol}: the REST funding request failed: {last}")


class RestFundingSource:
    """Monthly funding events from the REST endpoint, cached raw under the archive dir.

    A month is served from its cache file when one exists (so a run is reproducible
    offline); otherwise it is fetched (only when a fetcher is configured, i.e.
    ``--allow-network``), paginated by ``startTime``, validated and cached. Without a cache
    file and fetcher it raises :class:`RestFundingNotCached` (fail closed). Each month's raw
    JSON SHA-256 and event count are recorded as evidence.
    """

    def __init__(self, cache_dir: Path, fetch: RestFetch | None) -> None:
        self._cache_dir = cache_dir
        self._fetch = fetch
        self._evidence: dict[tuple[str, str], dict[str, object]] = {}

    def evidence(self) -> list[dict[str, object]]:
        return [self._evidence[key] for key in sorted(self._evidence)]

    def _path(self, symbol: str, month: str, clipped_end: int | None) -> Path:
        suffix = "" if clipped_end is None else f"-until-{clipped_end}"
        return self._cache_dir / symbol / f"{symbol}-fundingRate-{month}{suffix}.json"

    @staticmethod
    def _validate(
        raw: object, symbol: str, lo: int, hi: int
    ) -> list[tuple[int, float]]:
        """Symbol, strictly increasing times inside ``[lo, hi)`` and finite rates."""
        if not isinstance(raw, list):
            raise RestFundingError(f"{symbol}: the REST funding response is not a list")
        events: list[tuple[int, float]] = []
        previous = lo - 1
        for item in raw:
            if not isinstance(item, dict) or item.get("symbol") != symbol:
                raise RestFundingError(
                    f"{symbol}: a REST funding event names another symbol"
                )
            stamp = item.get("fundingTime")
            rate_raw = item.get("fundingRate")
            # The USD-M REST contract defines fundingTime as an integer
            # millisecond timestamp and fundingRate as a numeric string. In
            # particular, int(float_timestamp) must never shift an event.
            if type(stamp) is not int or not isinstance(rate_raw, str):
                raise RestFundingError(f"{symbol}: a REST funding event is malformed")
            try:
                rate = float(rate_raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise RestFundingError(
                    f"{symbol}: a REST funding event is malformed"
                ) from error
            if not math.isfinite(rate):
                raise RestFundingError(f"{symbol}: a REST funding rate is not finite")
            if stamp <= previous:
                raise RestFundingError(
                    f"{symbol}: REST funding times are not strictly increasing or "
                    "lie before the month"
                )
            if stamp >= hi:
                raise RestFundingError(
                    f"{symbol}: a REST funding event lies after the month"
                )
            previous = stamp
            events.append((stamp, rate))
        return events

    def _download(self, symbol: str, lo: int, hi: int) -> list[Any]:
        raw: list[Any] = []
        cursor = lo
        for _ in range(_REST_MAX_PAGES):
            try:
                page = json.loads(self._fetch(symbol, cursor, hi - 1))  # type: ignore[misc]
            except ValueError as error:
                raise RestFundingError(
                    f"{symbol}: a REST funding page is not JSON"
                ) from error
            if not isinstance(page, list):
                raise RestFundingError(f"{symbol}: the REST funding page is not a list")
            raw.extend(page)
            if len(page) < REST_PAGE_LIMIT:
                return raw
            try:
                cursor = int(page[-1]["fundingTime"]) + 1
            except (KeyError, TypeError, ValueError) as error:
                raise RestFundingError(
                    f"{symbol}: a REST funding page is malformed"
                ) from error
        raise RestFundingError(f"{symbol}: the REST funding pagination did not end")

    def month(
        self, symbol: str, lo: int, month_end: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Events of ``[lo, min(month_end, end_ms))``; the month needs no archive."""
        hi = min(month_end, end_ms)
        moment = datetime.fromtimestamp(lo / 1_000, UTC)
        label = _month_label(moment)
        path = self._path(symbol, label, None if hi == month_end else hi)
        source = "cache"
        if path.is_file():
            payload = path.read_bytes()
            try:
                raw: object = json.loads(payload)
            except ValueError as error:
                raise RestFundingError(
                    f"{path}: the cached REST response is not JSON"
                ) from error
        elif self._fetch is None:
            raise RestFundingNotCached(
                f"{symbol}: the REST funding of {label} is not cached and the network "
                "is not allowed"
            )
        else:
            source = "network"
            raw = self._download(symbol, lo, hi)
        events = self._validate(raw, symbol, lo, hi)
        if source == "network":
            payload = canonical_json_bytes(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".json.part")
            partial.write_bytes(payload)
            os.replace(partial, path)
        self._evidence[(symbol, label)] = {
            "symbol": symbol,
            "month": label,
            "events": len(events),
            "sha256": _sha256(payload),
            "source": source,
            "file": path.name,
        }
        return events


def _traded_row(row: Any) -> bool:
    """A bar counts as traded when its base or quote volume is positive."""
    return bool(row.volume > 0 or row.quote_volume > 0)


def split_zombie_tail(rows: Sequence[Any]) -> tuple[list[Any], int]:
    """``(rows up to the last traded bar, dropped trailing zero-volume bars)``.

    Vision keeps publishing flat zero-volume bars after a perp is delisted; only that
    trailing run is "not listed". Interior zero-volume bars are kept. With no traded bar
    at all every row is dropped.
    """
    last = -1
    for index, row in enumerate(rows):
        if _traded_row(row):
            last = index
    return list(rows[: last + 1]), len(rows) - (last + 1)


def _zero_runs(bars: Sequence[tuple[int, bool]]) -> list[tuple[int, int]]:
    """``[start open, end)`` of every run of consecutive zero-volume bars of at least
    ``FUNDING_MAX_GAP_HOURS`` hours; ``bars`` are ``(open, traded)`` sorted by open."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for opened, traded in bars:
        if not traded and start is not None and opened == end:
            end = opened + data.HOUR_MS
            continue
        if start is not None:
            runs.append((start, end))
            start = None
        if not traded:
            start, end = opened, opened + data.HOUR_MS
    if start is not None:
        runs.append((start, end))
    floor = protocol.FUNDING_MAX_GAP_HOURS * data.HOUR_MS
    return [(a, b) for a, b in runs if b - a >= floor]


def relisting_runs(bars: Sequence[tuple[int, bool]]) -> list[tuple[int, int]]:
    """Interior zero-volume runs of at least 24 h, funding events inside or not.

    Such a run is a delisting followed by a relisting (Amendment 2): the caller turns its
    bars into missing hours and drops the funding events inside it. A shorter run is kept.
    """
    return _zero_runs(bars)


def drop_funding_in_runs(
    events: Sequence[tuple[int, float]], runs: Sequence[tuple[int, int]]
) -> tuple[list[tuple[int, float]], int]:
    """``(events outside every run, number dropped)``: strictly inside ``(start, end)``."""
    kept = [(t, r) for t, r in events if not any(a < t < b for a, b in runs)]
    return kept, len(events) - len(kept)


def _run_records(runs: Sequence[tuple[int, int]]) -> list[list[object]]:
    return [[_iso(a), _iso(b), (b - a) // data.HOUR_MS] for a, b in runs]


def _month_label(moment: datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}"


def _next_month_ms(lo: int) -> int:
    return ms_from_utc(
        _next_month(_month_start(datetime.fromtimestamp(lo / 1_000, UTC)))
    )


def _read_funding_tolerant(
    reader: CarryRowSource,
    symbol: str,
    first_open: int,
    end_ms: int,
    *,
    kline_months: set[tuple[int, int]],
    traded_months: set[tuple[int, int]] | None = None,
    traded_span: tuple[int, int] | None = None,
    rest: RestFundingSource | None = None,
    refuse: bool = True,
) -> tuple[list[tuple[int, float]], list[dict[str, object]]]:
    """``(settled funding events, unpublished funding months)`` from ``first_open``.

    A funding month may be unpublished without a fallback only when the (truncated) perp
    has no bar with volume in that month (before the listing, after the effective end, or
    only zero-volume bars that remain after the relisting-gap conversion). In
    a month with a traded bar (positive volume) the events come from the REST fallback
    (``rest``); without it, or when it fails, or when it returns no event while the merged
    series has a gap longer than ``FUNDING_MAX_GAP_HOURS`` there (measured inside
    ``traded_span``), the month is not admissible: with ``refuse`` the call raises, naming
    the symbol and the month (coverage records it instead). Every unpublished month is
    returned.
    """
    traded = kline_months if traded_months is None else traded_months
    events: dict[int, float] = {}
    unpublished: list[dict[str, object]] = []
    empty: list[tuple[int, int, dict[str, object]]] = []
    for lo, hi in _month_chunks(first_open, end_ms):
        try:
            rows = reader.read_funding(_MARKET, symbol, lo, hi)
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") != "archive_not_published":
                raise
            moment = datetime.fromtimestamp(lo / 1_000, UTC)
            key = (moment.year, moment.month)
            start = ms_from_utc(_month_start(moment))
            record: dict[str, object] = {
                "symbol": symbol,
                "month": _month_label(moment),
                "admissible": key not in traded,
            }
            problem = ""
            if key in traded and rest is not None:
                try:
                    fetched = rest.month(symbol, start, _next_month_ms(lo), end_ms)
                except RestFundingNotCached as failure:
                    problem = f" ({failure})"
                except RestFundingError as failure:
                    problem = f" (REST fallback: {failure})"
                    record["rest_error"] = str(failure)
                else:
                    record["admissible"] = True
                    record["rest_events"] = len(fetched)
                    events.update(fetched)
                    if not fetched:
                        empty.append((lo, hi, record))
            unpublished.append(record)
            if refuse and not record["admissible"]:
                raise data.StageDataError(
                    f"{symbol}: the funding archive of {record['month']} is unpublished "
                    f"but the perp has kline bars in that month; refused{problem}"
                ) from error
            continue
        for row in rows:
            stamp = int(row.funding_time_ms)
            if stamp < end_ms:
                events[stamp] = float(row.rate)
    stamps = sorted(events)
    if empty:
        span_lo, span_hi = (first_open, end_ms) if traded_span is None else traded_span
        limit = protocol.FUNDING_MAX_GAP_HOURS * data.HOUR_MS
        for lo, hi, record in empty:
            before = [t for t in stamps if t < lo]
            after = [t for t in stamps if t >= hi]
            start = max(before[-1], span_lo) if before else span_lo
            stop = min(after[0], span_hi) if after else span_hi
            if stop - start > limit:
                record["admissible"] = False
                record["rest_gap_hours"] = (stop - start) / data.HOUR_MS
                if refuse:
                    raise data.StageDataError(
                        f"{symbol}: the funding archive of {record['month']} is unpublished, "
                        "the REST fallback returned no event and the funding series has a "
                        f"gap of {(stop - start) / data.HOUR_MS:g} hours there; refused"
                    )
    return [(t, events[t]) for t in stamps], unpublished


def load_perp(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
    rest: RestFundingSource | None = None,
) -> tuple[data.PerpHourly | None, dict[str, object]]:
    """The hourly series of one perp on the stage grid, or ``None`` if never listed.

    The effective listing end is the close of the last bar with volume: the trailing
    zero-volume bars Vision keeps publishing after a delisting ("zombie bars") are dropped
    first (Amendment 2), so the universe, the forced-close schedule, the funding rule and
    the gap checks all see the leg end there.

    Unlike a fixed roster the series may end (delisting) or hold gaps: a missing hour inside
    a declared exchange-wide halt (and between the first and the last published hour) is
    ``halted``; every other missing hour is simply absent, and the dataset assembler turns
    it into an unavailable bar. Nothing is filled forward or invented.
    """
    start_ms, end_ms = ms_from_utc(window.load_start), ms_from_utc(limit_close)
    first_open, unpublished_leading, leading = _first_listed_open(
        detect, reader, symbol, start_ms, end_ms
    )
    for lo, hi in leading:
        day = hi - 24 * data.HOUR_MS
        if _probe(reader, symbol, day):
            raise data.StageDataError(
                f"{symbol}: the monthly archive of the leading month starting "
                f"{_iso(lo)} is unpublished but a daily archive exists ({_iso(day)}); "
                "the listing would silently move"
            )
    coverage: dict[str, object] = {
        "unpublished_leading_months": unpublished_leading,
        "first_listed_open": None,
        "rows": 0,
    }
    if first_open is None:
        return None, coverage
    coverage["first_listed_open"] = _iso(first_open)
    rows, unpublished = _read_klines_tolerant(reader, symbol, first_open, end_ms)
    _check_order(
        [int(row.open_time_ms) for row in rows],
        market=_MARKET,
        symbol=symbol,
        what="klines",
    )
    rows, zombie_bars = split_zombie_tail(rows)
    coverage["zombie_tail_bars"] = zombie_bars
    if not rows:
        coverage["effective_end"] = None
        return None, coverage
    effective_end = int(rows[-1].open_time_ms) + data.HOUR_MS
    coverage["effective_end"] = _iso(effective_end)
    traded_span = (
        int(next(r for r in rows if _traded_row(r)).open_time_ms),
        effective_end,
    )
    gap_runs = relisting_runs([(int(r.open_time_ms), _traded_row(r)) for r in rows])
    if gap_runs:
        rows = [
            r
            for r in rows
            if not any(a <= int(r.open_time_ms) < b for a, b in gap_runs)
        ]
    coverage["relisting_gap_runs"] = _run_records(gap_runs)
    n_hours = (end_ms - start_ms) // data.HOUR_MS
    present = np.zeros(n_hours, dtype=np.bool_)
    arrays = {
        name: np.full(n_hours, np.nan)
        for name in ("open", "high", "low", "close", "vol")
    }
    for row in rows:
        slot = (int(row.open_time_ms) - start_ms) // data.HOUR_MS
        if not 0 <= slot < n_hours:
            raise data.StageEmbargoError(f"{symbol}: a row lies outside the read range")
        present[slot] = True
        arrays["open"][slot] = float(row.open)
        arrays["high"][slot] = float(row.high)
        arrays["low"][slot] = float(row.low)
        arrays["close"][slot] = float(row.close)
        arrays["vol"][slot] = float(row.quote_volume)
    halted = np.zeros(n_hours, dtype=np.bool_)
    if present.any():
        first_slot = int(np.argmax(present))
        last_slot = n_hours - 1 - int(np.argmax(present[::-1]))
        for lo_t, hi_t in halt_intervals:
            lo_slot = max(
                (ms_from_utc(lo_t) - start_ms) // data.HOUR_MS, first_slot + 1
            )
            hi_slot = min((ms_from_utc(hi_t) - start_ms) // data.HOUR_MS, last_slot)
            if lo_slot < hi_slot:
                span = slice(lo_slot, hi_slot)
                halted[span] = ~present[span]
    events, funding_unpublished = _read_funding_tolerant(
        reader,
        symbol,
        first_open,
        end_ms,
        kline_months=_months_with_rows([int(row.open_time_ms) for row in rows]),
        traded_months=_months_with_rows(
            [int(row.open_time_ms) for row in rows if _traded_row(row)]
        ),
        traded_span=traded_span,
        rest=rest,
    )
    events, dropped = drop_funding_in_runs(events, gap_runs)
    coverage["dropped_funding_events_in_gaps"] = dropped
    leg = data.PerpHourly(
        present=present,
        halted=halted,
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        quote_volume=arrays["vol"],
        funding_ms=np.asarray([t for t, _ in events], dtype=np.int64),
        funding_rate=np.asarray([r for _, r in events], dtype=np.float64),
    )
    last_open = start_ms + (n_hours - 1 - int(np.argmax(present[::-1]))) * data.HOUR_MS
    coverage.update(
        {
            "rows": len(rows),
            "last_published_open": _iso(last_open),
            "unpublished_months": unpublished,
            "unpublished_funding_months": funding_unpublished,
            "halted_hours": int(halted.sum()),
            "funding_events": len(events),
        }
    )
    return leg, coverage


def load_stage_hourly(
    detect: CarryRowSource,
    reader: CarryRowSource,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    symbols: Sequence[str],
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    rest: RestFundingSource | None = None,
) -> tuple[dict[str, data.PerpHourly], dict[str, dict[str, object]]]:
    """Read every symbol's hourly series through the stage guard (never-listed ones omitted)."""
    if halt_intervals is None:
        halt_intervals = protocol.declared_halt_intervals()
    guarded_detect = data.StageRowSource(
        detect, window=window, limit_close=limit_close, symbols=symbols
    )
    guarded_reader = data.StageRowSource(
        reader, window=window, limit_close=limit_close, symbols=symbols
    )
    legs: dict[str, data.PerpHourly] = {}
    coverage: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        leg, entry = load_perp(
            guarded_detect,
            guarded_reader,
            symbol,
            window=window,
            limit_close=limit_close,
            halt_intervals=halt_intervals,
            rest=rest,
        )
        coverage[symbol] = entry
        if leg is not None:
            legs[symbol] = leg
    return legs, coverage


def build_stage(
    hourly: dict[str, data.PerpHourly],
    *,
    window: data.StageWindow,
    limit_close: datetime,
    rules: dict[str, LegRules],
    data_authorization_url: str | None,
    listing: dict[str, object] | None = None,
    notice_book: notices.NoticeBook | None = None,
) -> data.StageData:
    """The universe (static for DEV, monthly top 30 otherwise) and the stage dataset."""
    if window.stage == protocol.STAGE_DEV:
        missing = sorted(set(protocol.DEV_ROSTER) - set(hourly))
        if missing:
            raise data.StageDataError(f"DEV legs without data: {missing}")
        names = tuple(sorted(protocol.DEV_ROSTER))
        months = universe.months_of_stage(
            window.first_decision, window.last_decision_bar
        )
        month_universe = {
            month: universe.MonthUniverse(
                month_start=month, eligible=names, members=names, book=True
            )
            for month in months
        }
        member = np.ones(
            (data.stage_clock(window, limit_close).size, len(names)), dtype=np.bool_
        )
        static = True
    else:
        series = {symbol: leg.universe_series() for symbol, leg in hourly.items()}
        last_month_bar = min(window.last_decision_bar, limit_close - data.BAR)
        month_universe = universe.stage_universe(
            series,
            grid_start=window.load_start,
            first_decision=window.first_decision,
            last_decision_bar=last_month_bar,
        )
        names = universe.ever_members(month_universe)
        if not names:
            raise data.StageDataError("no symbol is ever a universe member")
        clock = data.stage_clock(window, limit_close)
        member = universe.membership_mask(month_universe, clock, names)
        static = False
    return data.assemble_stage_dataset(
        window,
        {name: hourly[name] for name in names},
        member=member,
        universe=month_universe,
        static_universe=static,
        rules={name: rules[name] for name in names},
        limit_close=limit_close,
        data_authorization_url=data_authorization_url,
        listing=listing,
        notices=notice_book,
    )


# ---------------------------------------------------------------- sources


def _sources(
    args: argparse.Namespace, plan: IngestionPlan, policy: EmbargoPolicy
) -> tuple[CarryRowSource, CarryRowSource]:
    """``(detect, reader)``: the primary source and, with the fallback, its wrapper."""
    factory = base._vision_factory(args.archive_dir, args.allow_network, policy)
    primary = factory(plan)
    if not args.daily_fallback:
        return primary, primary
    daily = base._vision_factory(args.archive_dir, args.allow_network, policy)
    return primary, agy_tool._AnchorFallbackSource(
        primary, daily, halts=protocol.declared_halt_intervals(), embargo=policy
    )


# ----------------------------------------------------------------- CLI plumbing


def _auth_url(stage: str) -> str | None:
    """The owner relay a stage's reads are bound to (DEV reads known-used data: none)."""
    return (
        protocol.DATA_AUTHORIZATION_URL if stage in protocol.AUTHORIZED_STAGES else None
    )


def plan_document(stage: str, now: datetime) -> dict[str, Any]:
    """Protocol, digest and windows; reads no data."""
    window = data.frozen_window(stage)
    protocol.verify_candidates_binding()
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "candidates_sha256": protocol.CANDIDATES_SHA256,
        "initial_margin_rate": protocol.INITIAL_MARGIN_RATE,
        "lowturn_protocol_digest": protocol.LOWTURN_PROTOCOL_DIGEST,
        "stage": stage,
        "window": window.to_dict(),
        "windows": {
            name: data.frozen_window(name).to_dict() for name in protocol.STAGES
        },
        "final_not_before": protocol.FINAL_NOT_BEFORE,
        "latest_closed_bar": data.latest_closed_bar(now).isoformat(),
        "data_authorization_url": protocol.DATA_AUTHORIZATION_URL,
        "data_authorization_stages": list(protocol.AUTHORIZED_STAGES),
        "harness_source_digest": protocol.harness_source_digest(),
        "default_embargo": data.DEFAULT_EMBARGO.isoformat(),
        "protocol": protocol.protocol_payload(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.xs_lowvol_funding",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="print the protocol, its digest and the stage windows"
    )
    plan.add_argument("--stage", choices=protocol.STAGES, required=True)
    cov = commands.add_parser(
        "coverage",
        help="result-blind data-coverage report; builds no universe, signal or return",
    )
    cov.add_argument("--stage", choices=protocol.STAGES, required=True)
    cov.add_argument("--through", type=_parse_instant, default=None)
    cov.add_argument("--forward-symbols", type=Path, default=None)
    cov.add_argument("--archive-dir", type=Path, required=True)
    cov.add_argument("--allow-network", action="store_true")
    cov.add_argument("--daily-fallback", action="store_true")
    cov.add_argument("--output-dir", type=Path, required=True)
    halts = commands.add_parser(
        "halts-from-coverage",
        help="derive the exchange-wide perpetual halt list (and its evidence SHA-256) "
        "from a DISJOINT coverage.json; reads no market data",
    )
    halts.add_argument("--coverage", type=Path, required=True)
    notice_cmd = commands.add_parser(
        "notices",
        help="record the public delisting-notice file from the Binance CMS API "
        "(result-blind; reads no market data; network needs --allow-network)",
    )
    notice_cmd.add_argument("--output", type=Path, required=True)
    notice_cmd.add_argument("--since", type=_parse_instant, default=NOTICE_SINCE)
    notice_cmd.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="JSON list of {symbol, article_code, kind, evidence}; each is verified "
        "against the real article fetched from the API",
    )
    notice_cmd.add_argument("--allow-network", action="store_true")
    run = commands.add_parser(
        "run", help="run one stage (needs the G4 relay and the owner authorization)"
    )
    run.add_argument("--stage", choices=protocol.STAGES, required=True)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--interim", action="store_true")
    run.add_argument("--through", type=_parse_instant, default=None)
    run.add_argument(
        "--dev-result",
        type=Path,
        default=None,
        help="DISJOINT: the dev-result.json of the DEV run (the run order)",
    )
    run.add_argument(
        "--disjoint-decision",
        type=Path,
        default=None,
        help="FORWARD: the DISJOINT decision.json of the same digest (XS_SUPPORTED)",
    )
    run.add_argument("--forward-symbols", type=Path, default=None)
    run.add_argument(
        "--delisting-notices",
        type=Path,
        default=None,
        help="the recorded notice file (default: the repository file; its SHA-256 "
        "must equal protocol.DELISTING_NOTICES_SHA256)",
    )
    run.add_argument(
        "--confirm-single-use",
        action="store_true",
        help="DISJOINT: confirm it runs exactly once for this protocol id",
    )
    run.add_argument("--archive-dir", type=Path, required=True)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--daily-fallback", action="store_true")
    run.add_argument("--exchange-info-perp", type=Path, required=True)
    run.add_argument("--authorization-url", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_prior(path: Path, *, stage: str, what: str) -> dict[str, Any]:
    """A prior stage's result of exactly this protocol digest; anything else refuses."""
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"the {what} is not JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("protocol_id") != protocol.PROTOCOL_ID
        or document.get("protocol_digest") != protocol.protocol_digest()
        or document.get("stage") != stage
        or document.get("mode") != runner.MODE_DECISION
    ):
        raise ValueError(f"the file is not a {what} of this exact protocol digest")
    document["__sha256"] = _sha256(raw)
    return document


def _load_dev_result(path: Path) -> dict[str, Any]:
    document = _load_prior(path, stage=protocol.STAGE_DEV, what="DEV result")
    invalid = document.get("invalid_cells")
    if not isinstance(invalid, list):
        raise ValueError("the DEV result carries no invalid_cells list")
    if invalid:
        raise ValueError(
            f"the DEV result has invalid cells {invalid}: the harness is not fit to run "
            "(DEV may stop the line); DISJOINT is refused"
        )
    return {"sha256": document["__sha256"], "status": document.get("status")}


def _load_disjoint_decision(path: Path) -> dict[str, Any]:
    document = _load_prior(
        path, stage=protocol.STAGE_DISJOINT, what="DISJOINT decision"
    )
    label = document.get("label")
    if label != protocol.LABEL_SUPPORTED:
        raise ValueError(
            f"DISJOINT was {label!r}: the line stops and no FORWARD run is made "
            "(only XS_SUPPORTED goes to FORWARD)"
        )
    return {"label": label, "sha256": document["__sha256"]}


def _check_mode(args: argparse.Namespace, now: datetime) -> str:
    """The run mode; refuses every stage/mode/order combination the protocol forbids."""
    if args.stage != protocol.STAGE_FORWARD:
        if args.final or args.interim or args.through is not None:
            raise ValueError(f"{args.stage} takes no --final/--interim/--through")
        if args.disjoint_decision is not None:
            raise ValueError("--disjoint-decision belongs to FORWARD")
        if args.forward_symbols is not None:
            raise ValueError("--forward-symbols belongs to FORWARD")
        if args.confirm_single_use and args.stage != protocol.STAGE_DISJOINT:
            raise ValueError("--confirm-single-use belongs to DISJOINT")
        if args.stage == protocol.STAGE_DEV:
            if args.dev_result is not None:
                raise ValueError("--dev-result belongs to DISJOINT")
        elif args.dev_result is None:
            raise ValueError("DISJOINT needs --dev-result (DEV runs first)")
        elif not args.confirm_single_use:
            raise ValueError(
                "DISJOINT runs exactly once for this protocol id; pass --confirm-single-use"
            )
        return runner.MODE_DECISION
    if args.dev_result is not None:
        raise ValueError("--dev-result belongs to DISJOINT")
    if args.confirm_single_use:
        raise ValueError("--confirm-single-use belongs to DISJOINT")
    if args.final == args.interim:
        raise ValueError("FORWARD needs exactly one of --final and --interim")
    if args.disjoint_decision is None:
        raise ValueError("FORWARD needs --disjoint-decision (the stop rule)")
    if args.interim:
        return runner.MODE_INTERIM
    if args.through is not None:
        raise ValueError("--through is for --interim only")
    gate = datetime.fromisoformat(protocol.FINAL_NOT_BEFORE.replace("Z", "+00:00"))
    if now < gate:
        raise ValueError(
            f"--final is refused before {protocol.FINAL_NOT_BEFORE} (now {now.isoformat()})"
        )
    return runner.MODE_DECISION


def _stage_limit(
    args: argparse.Namespace, window: data.StageWindow, now: datetime, mode: str
) -> datetime:
    limit = data.read_limit(
        window,
        now=now,
        authorization_url=_auth_url(args.stage),
        through=args.through,
    )
    if mode == runner.MODE_INTERIM and args.through is None:
        # klines need a closed day; funding exists only through the last published month
        limit = min(limit, _floor_day(now), funding_boundary(now))
    if limit <= window.first_decision + 2 * data.BAR:
        raise StageEmbargoError("no closed day beyond the first decision")
    if _ceil_day(limit) > _floor_day(now):
        raise StageEmbargoError(
            "the day archive holding the last read row is not closed yet"
        )
    require_funding_published(limit, now)
    return limit


# --------------------------------------------------------------------- coverage


def _coverage_sources(
    args: argparse.Namespace, plan: IngestionPlan, policy: EmbargoPolicy
) -> tuple[CarryRowSource, CarryRowSource, PlaceholderLog]:
    log: PlaceholderLog = []

    def factory(inner_plan: IngestionPlan) -> CarryRowSource:
        inner = (
            base._StageVisionFetcher(policy, max_attempts=1)
            if args.allow_network
            else None
        )
        return agy_tool._CoverageVisionSource(
            base.ArchiveDirFetcher(args.archive_dir, inner), inner_plan, log
        )

    primary = factory(plan)
    if not args.daily_fallback:
        return primary, primary, log
    fallback = agy_tool._AnchorFallbackSource(
        primary, factory, halts=protocol.declared_halt_intervals(), embargo=policy
    )
    return primary, fallback, log


def _funding_coverage(
    reader: CarryRowSource,
    symbol: str,
    first_open: int,
    end_ms: int,
    errors: list[dict[str, object]],
    kline_months: set[tuple[int, int]],
    traded_months: set[tuple[int, int]],
    traded_span: tuple[int, int] | None,
    rest: RestFundingSource | None,
    gap_runs: Sequence[tuple[int, int]] = (),
) -> dict[str, object]:
    try:
        events, unpublished = _read_funding_tolerant(
            reader,
            symbol,
            first_open,
            end_ms,
            kline_months=kline_months,
            traded_months=traded_months,
            traded_span=traded_span,
            rest=rest,
            refuse=False,
        )
    except CarryScreenHalt as error:
        errors.append({"phase": "funding", **_halt_error(error)})
        return {"events": None}
    events, dropped = drop_funding_in_runs(events, gap_runs)
    stamps = [t for t, _ in events]
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    return {
        "unpublished_funding_months": unpublished,
        "rest_funding_months": sum(1 for u in unpublished if "rest_events" in u),
        "events": len(stamps),
        "dropped_funding_events_in_gaps": dropped,
        "first_event": None if not stamps else _iso(stamps[0]),
        "last_event": None if not stamps else _iso(stamps[-1]),
        "max_gap_hours": None if not gaps else max(gaps) / data.HOUR_MS,
    }


def _leg_coverage(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    rest: RestFundingSource | None = None,
) -> tuple[dict[str, Any], int | None, int | None, set[int]]:
    """Coverage facts of one perp; never raises on a gap, a 404 or a bad archive.

    Returns ``(entry, first listed open, last published open, missing hours between them)``.
    Hours after the last traded row are the effective end (a delisting), not a gap; the
    trailing zero-volume "zombie" bars after it are dropped like the run does (Amendment 2).
    """
    start_ms, end_ms = ms_from_utc(window.load_start), ms_from_utc(limit_close)
    errors: list[dict[str, object]] = []
    first_open: int | None = None
    unpublished_leading = 0
    leading_with_dailies: list[str] = []
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(_MARKET, symbol, data.BASE_INTERVAL, lo, hi)
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") == "archive_not_published":
                unpublished_leading += 1
                if _probe(reader, symbol, hi - 24 * data.HOUR_MS):
                    leading_with_dailies.append(_iso(lo))
                continue
            errors.append(
                {"phase": "listing", "kind": "coverage", "message": str(error)}
            )
            break
        except CarryIntegrityHalt as error:
            if (error.evidence or {}).get("kind") == "malformed_archive":
                try:
                    rows = reader.read_klines(
                        _MARKET, symbol, data.BASE_INTERVAL, lo, hi
                    )
                except CarryScreenHalt as again:
                    errors.append({"phase": "listing", **_halt_error(again)})
                    break
            else:
                errors.append({"phase": "listing", **_halt_error(error)})
                break
        except CarryScreenHalt as error:
            errors.append({"phase": "listing", **_halt_error(error)})
            break
        if rows:
            first_open = int(rows[0].open_time_ms)
            break
        unpublished_leading += 1
    entry: dict[str, Any] = {
        "first_listed_open": None if first_open is None else _iso(first_open),
        "unpublished_leading_months": unpublished_leading,
        "leading_unpublished_months_with_daily_archives": leading_with_dailies,
        "errors": errors,
    }
    if first_open is None:
        return entry, None, None, set()
    present: set[int] = set()
    zero_volume: list[int] = []
    traded_opens: set[int] = set()
    unread: list[dict[str, object]] = []
    for lo, hi in _month_chunks(first_open, end_ms):
        span = {"from": _iso(lo), "to": _iso(hi)}
        try:
            rows = reader.read_klines(_MARKET, symbol, data.BASE_INTERVAL, lo, hi)
        except CarryCoverageError as error:
            kind = (error.evidence or {}).get("kind", "coverage")
            unread.append(
                {**span, "kind": "coverage", "detail": kind, "message": str(error)}
            )
            continue
        except CarryScreenHalt as error:
            unread.append({**span, **_halt_error(error)})
            continue
        for row in rows:
            opened = int(row.open_time_ms)
            present.add(opened)
            if row.volume == 0:
                zero_volume.append(opened)
            if _traded_row(row):
                traded_opens.add(opened)
    last_open = max(traded_opens) if traded_opens else None
    zombie_bars = sum(
        1 for opened in present if last_open is None or opened > last_open
    )
    present = {
        opened for opened in present if last_open is not None and opened <= last_open
    }
    gap_runs: list[tuple[int, int]] = []
    if last_open is not None:
        gap_runs = relisting_runs([(o, o in traded_opens) for o in sorted(present)])
        present = {o for o in present if not any(a <= o < b for a, b in gap_runs)}
    zero_volume = [opened for opened in zero_volume if opened in present]
    missing: list[int] = []
    if last_open is not None:
        missing = [
            h
            for h in range(first_open, last_open + data.HOUR_MS, data.HOUR_MS)
            if h not in present
        ]
    trailing = (
        0
        if last_open is None
        else (end_ms - (last_open + data.HOUR_MS)) // data.HOUR_MS
    )
    entry.update(
        {
            "expected_hours": (end_ms - first_open) // data.HOUR_MS,
            "present_hours": len(present),
            "last_published_open": None if last_open is None else _iso(last_open),
            "missing_hours": len(missing),
            "missing_ranges": _hour_ranges(missing),
            "trailing_hours_after_last_row": int(trailing),
            "zombie_tail_bars": zombie_bars,
            "relisting_gap_runs": _run_records(gap_runs),
            "effective_end": (
                None if last_open is None else _iso(last_open + data.HOUR_MS)
            ),
            "zero_volume_regular_rows": _hour_ranges(zero_volume),
            "unread_chunks": unread,
            "funding": _funding_coverage(
                reader,
                symbol,
                first_open,
                end_ms,
                errors,
                _months_with_rows(sorted(present)),
                _months_with_rows(sorted(traded_opens)),
                (
                    None
                    if last_open is None
                    else (min(traded_opens), last_open + data.HOUR_MS)
                ),
                rest,
                gap_runs,
            ),
        }
    )
    entry["dropped_funding_events_in_gaps"] = entry["funding"].get(
        "dropped_funding_events_in_gaps", 0
    )
    return entry, first_open, last_open, set(missing)


def _exchange_wide(
    firsts: dict[str, int], lasts: dict[str, int], missing: dict[str, set[int]]
) -> dict[str, object]:
    """Perp hours missing for every symbol listed then (at least two listed)."""
    hours: list[int] = []
    union: set[int] = set().union(*missing.values()) if missing else set()
    for hour in sorted(union):
        listed = [
            name for name, first in firsts.items() if first <= hour <= lasts[name]
        ]
        if len(listed) >= 2 and all(hour in missing[name] for name in listed):
            hours.append(hour)
    return {
        "rule": "missing perp hours shared by every symbol listed then (at least two "
        "listed; a symbol is listed from its first to its last published hour)",
        "hours": len(hours),
        "ranges": _hour_ranges(hours),
    }


def coverage_report(
    detect: CarryRowSource,
    reader: CarryRowSource,
    placeholders: PlaceholderLog,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    symbols: Sequence[str],
    rest: RestFundingSource | None = None,
) -> dict[str, Any]:
    """Per-symbol listing, gaps, fills, placeholders, funding and checksum results.

    Rows are read through the stage guard only to validate and count them; this builds no
    universe, dataset, signal, replay or return, and never refuses because of a gap.
    """
    guarded_detect = data.StageRowSource(
        detect, window=window, limit_close=limit_close, symbols=symbols
    )
    guarded_reader = data.StageRowSource(
        reader, window=window, limit_close=limit_close, symbols=symbols
    )
    entries: dict[str, Any] = {}
    firsts: dict[str, int] = {}
    lasts: dict[str, int] = {}
    missing: dict[str, set[int]] = {}
    for symbol in symbols:
        entry, first, last, gone = _leg_coverage(
            guarded_detect,
            guarded_reader,
            symbol,
            window=window,
            limit_close=limit_close,
            rest=rest,
        )
        entries[symbol] = {_MARKET.value: entry}
        if first is not None and last is not None:
            firsts[symbol], lasts[symbol], missing[symbol] = first, last, gone
    evidence_fn = getattr(reader, "evidence", None)
    evidence = list(evidence_fn()) if callable(evidence_fn) else []
    fills_fn = getattr(reader, "fills", None)
    fills = list(fills_fn()) if callable(fills_fn) else []
    unusable_fn = getattr(reader, "unusable_days", None)
    unusable = list(unusable_fn()) if callable(unusable_fn) else []
    placeholder_rows = sorted(set(placeholders))
    for symbol, markets in entries.items():
        entry = markets[_MARKET.value]
        entry["unusable_daily_archives"] = [
            u for u in unusable if u.get("symbol") == symbol
        ]
        mine = [e for e in evidence if e.get("symbol") == symbol]
        failures = [
            c
            for c in entry.get("unread_chunks", [])
            if c.get("detail") in base._INTEGRITY_DETAILS
        ] + [e for e in entry["errors"] if e.get("detail") in base._INTEGRITY_DETAILS]
        entry["checksums"] = {
            "archives_verified": len(mine),
            "failures": failures,
            "archives": [
                {
                    "url": e["url"],
                    "archive_sha256": e.get("archive_sha256"),
                    "checksum_file_sha256": e.get("checksum_file_sha256"),
                    "verified": True,
                }
                for e in mine
            ],
        }
        filled = [f for f in fills if f.get("symbol") == symbol]
        filled_ms = [
            int(datetime.fromisoformat(str(f["open_time"])).timestamp() * 1_000)
            for f in filled
        ]
        entry["filled_by_daily_fallback"] = {
            "hours": len(filled),
            "ranges": _hour_ranges(filled_ms),
            "archives": sorted(
                {
                    (str(f.get("archive_url")), str(f.get("archive_sha256")))
                    for f in filled
                }
            ),
        }
        entry["placeholder_rows"] = [
            {
                "open_time": _iso(open_ms),
                "close_time": _iso(close_ms),
                "archive_url": url,
            }
            for sym, open_ms, close_ms, url in placeholder_rows
            if sym == symbol
        ]
    return {
        "symbols": entries,
        "exchange_wide_perp_gaps": _exchange_wide(firsts, lasts, missing),
        "archive_evidence_count": len(evidence),
        "archive_evidence_digest": tierd_driver._digest(evidence) if evidence else None,
        "rest_funding_evidence": [] if rest is None else rest.evidence(),
        "daily_fallback_filled_hours": len(fills),
        "registered_halts": [
            [_iso(ms_from_utc(a)), _iso(ms_from_utc(b))]
            for a, b in protocol.declared_halt_intervals()
        ],
    }


def halts_from_coverage(path: Path) -> dict[str, Any]:
    """The exchange-wide perpetual ranges of a DISJOINT coverage.json as a halt list.

    Only the report's exchange-wide ranges qualify (a range counts only if every symbol
    listed then is missing it). It also lists, per symbol, the missing ranges that would
    remain (and make that symbol's bars unavailable). It refuses a report with read errors
    and computes nothing from prices.
    """
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != COVERAGE_SCHEMA:
        raise ValueError("not an xs_lowvol_funding coverage report")
    if document.get("stage") != protocol.STAGE_DISJOINT:
        raise ValueError("declared halts are derived from the DISJOINT coverage")
    legs = [
        (name, market, entry)
        for name, markets in document["symbols"].items()
        for market, entry in markets.items()
    ]
    bad = sorted(f"{n}:{m}" for n, m, e in legs if _read_errors(e))
    if bad:
        raise ValueError(f"the coverage report has read errors for {bad}")
    hidden = sorted(
        f"{n}:{m}"
        for n, m, e in legs
        if e.get("leading_unpublished_months_with_daily_archives")
    )
    if hidden:
        raise ValueError(
            f"leading months with daily archives (listing would move): {hidden}"
        )

    def span(item: dict[str, Any]) -> tuple[datetime, datetime]:
        lo = datetime.fromisoformat(item["first_missing_open"])
        return lo, lo + timedelta(hours=int(item["hours"]))

    shared = [span(r) for r in document["exchange_wide_perp_gaps"]["ranges"]]

    def inside(moment: datetime) -> bool:
        return any(lo <= moment < hi for lo, hi in shared)

    remaining: dict[str, list[dict[str, object]]] = {}
    for name, market, entry in legs:
        left: list[int] = []
        for item in entry.get("missing_ranges", []):
            lo, hi = span(item)
            moment = lo
            while moment < hi:
                if not inside(moment):
                    left.append(int(moment.timestamp() * 1_000))
                moment += timedelta(hours=1)
        if left:
            remaining[f"{name}:{market}"] = _hour_ranges(left)
    return {
        "coverage_json_sha256": _sha256(raw),
        "coverage_protocol_digest": document.get("protocol_digest"),
        "coverage_code_commit": document.get("code_commit"),
        "declared_halts": [
            [lo.strftime("%Y-%m-%dT%H:%M:%SZ"), hi.strftime("%Y-%m-%dT%H:%M:%SZ")]
            for lo, hi in shared
        ],
        "declared_hours": sum(
            int((hi - lo).total_seconds() // 3600) for lo, hi in shared
        ),
        "unread_chunks": {
            f"{n}:{m}": e["unread_chunks"] for n, m, e in legs if e.get("unread_chunks")
        },
        "gaps_that_remain_and_make_bars_unavailable": remaining,
    }


def _coverage_read_errors(entries: dict[str, Any]) -> list[str]:
    return sorted(
        f"{n}:{m}"
        for n, markets in entries.items()
        for m, e in markets.items()
        if _read_errors(e)
    )


def coverage_summary_fields(entries: dict[str, Any]) -> dict[str, Any]:
    """What would refuse a run, visible before it: inadmissible funding months, REST, zombies."""
    inadmissible: list[str] = []
    rest_months = 0
    zombies = 0
    relisting = 0
    for name, markets in sorted(entries.items()):
        for entry in markets.values():
            if entry.get("zombie_tail_bars"):
                zombies += 1
            if entry.get("relisting_gap_runs"):
                relisting += 1
            funding = entry.get("funding") or {}
            for record in funding.get("unpublished_funding_months", []):
                if not record.get("admissible"):
                    inadmissible.append(f"{name}:{record['month']}")
                if "rest_events" in record:
                    rest_months += 1
    return {
        "inadmissible_unpublished_funding_months": {
            "count": len(inadmissible),
            "months": inadmissible,
        },
        "rest_funding_months": rest_months,
        "zombie_tail_legs": zombies,
        "relisting_gap_legs": relisting,
    }


def _rest_source(
    args: argparse.Namespace, rest_fetch: RestFetch | None
) -> RestFundingSource:
    """The REST funding fallback: a raw cache under the archive dir, the network only with
    ``--allow-network``."""
    fetch: RestFetch | None = None
    if args.allow_network:
        fetch = binance_rest_fetch if rest_fetch is None else rest_fetch
    return RestFundingSource(args.archive_dir / "rest" / "fundingRate", fetch)


def _coverage_main(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    state: GitState,
    rows_factory: RowSourceFactory | None,
    moment: datetime,
    gh: g4.GhRunner,
    rest_fetch: RestFetch | None = None,
) -> int:
    try:
        if args.stage != protocol.STAGE_FORWARD and (
            args.through is not None or args.forward_symbols is not None
        ):
            raise ValueError("--through and --forward-symbols are for FORWARD only")
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        window = data.frozen_window(args.stage)
        symbols = stage_symbols(args.stage, args.forward_symbols)
        mode = (
            runner.MODE_INTERIM
            if args.stage == protocol.STAGE_FORWARD
            else runner.MODE_DECISION
        )
        limit = _stage_limit(args, window, moment, mode)
        data_auth: g4.CommentSnapshot | None = None
        if _auth_url(args.stage) is not None:
            data_auth = g4.fetch_comment(protocol.DATA_AUTHORIZATION_URL, gh)
        commit, clean = state()
        policy = embargo_for(limit, stage=args.stage)
        plan = build_plan(window, limit, moment, policy, symbols)
        if rows_factory is not None:
            detect = reader = rows_factory(plan)
            log: PlaceholderLog = list(getattr(detect, "placeholders", []))
        else:
            detect, reader, log = _coverage_sources(args, plan, policy)
        try:
            report = coverage_report(
                detect,
                reader,
                log,
                window=window,
                limit_close=limit,
                symbols=symbols,
                rest=_rest_source(args, rest_fetch),
            )
        finally:
            for source in {id(detect): detect, id(reader): reader}.values():
                close = getattr(source, "close", None)
                if callable(close):
                    close()
        document = {
            "schema_version": COVERAGE_SCHEMA,
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_digest": protocol.protocol_digest(),
            "stage": args.stage,
            "window": window.to_dict(),
            "read_limit_close": limit.isoformat(),
            "source": "vision_usdm_1h_and_funding",
            "symbols_sha256": hashlib.sha256(
                json.dumps(list(symbols)).encode("utf-8")
            ).hexdigest(),
            "daily_fallback_enabled": bool(args.daily_fallback),
            "allow_network": bool(args.allow_network),
            "data_authorization": (
                None
                if data_auth is None
                else {
                    "url": protocol.DATA_AUTHORIZATION_URL,
                    "author_login": data_auth.login,
                    "body_sha256": data_auth.body_sha256,
                    "updated_at": data_auth.updated_at,
                    "embargo_policy": type(policy).__name__,
                }
            ),
            "code_commit": commit,
            "working_tree_clean": clean,
            "run_at": moment.isoformat(),
            "not_computed": "no universe, signal, replay, return or label; rows were "
            "read only to validate and count them",
            **report,
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / COVERAGE_NAME).write_bytes(canonical_json_bytes(document))
    except g4.G4AuthError as error:
        err.write(f"authorization failed: {error}\n")
        return EXIT_UNAUTHENTICATED
    except (ValueError, SealedWindowEmbargoError, FileExistsError, OSError) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED
    entries = document["symbols"]
    errored = _coverage_read_errors(entries)
    out.write(
        json.dumps(
            {
                "stage": args.stage,
                "coverage": str(args.output_dir / COVERAGE_NAME),
                "legs_with_missing_hours": sorted(
                    f"{n}:{m}"
                    for n, markets in entries.items()
                    for m, e in markets.items()
                    if e.get("missing_hours")
                ),
                "exchange_wide_perp_gap_hours": document["exchange_wide_perp_gaps"][
                    "hours"
                ],
                "legs_with_read_errors": errored,
                **coverage_summary_fields(entries),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return EXIT_HALTED if errored else EXIT_OK


# ----------------------------------------------------------- public delisting notices

CMS_ROOT = "https://www.binance.com/bapi/composite/v1/public/cms/article"
NOTICE_CATALOGS: tuple[int, ...] = (161, 49, 48, 157)
NOTICE_PAGE_SIZE = 20
NOTICE_SINCE = datetime(2020, 10, 1, tzinfo=UTC)
NOTICE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_NOTICE_THROTTLE_SECONDS = 0.8
NoticeFetch = Callable[[str], bytes]

_USDT_TOKEN = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{2,30}USDT)(?![A-Z0-9])")
_BASE_WORD = re.compile(r"[A-Za-z0-9]+")
_TITLE_STOP = frozenset(
    {"USDT", "USDC", "BUSD", "USD", "USDS", "COIN", "AND", "FUTURES", "BINANCE"}
)
_FUTURES_TITLE = re.compile(r"futures|perpetual|usdⓈ-m|usds-m|usdt-margined", re.I)
_USDM_TITLE = re.compile(r"usdt-margined|usdⓈ-m|usds-m|perpetual", re.I)
_DELIST_TITLE = re.compile(r"delist", re.I)
_OTHER_PRODUCT_DELIST = re.compile(
    r"margin|alpha|loan|savings|earn|options|card|convert|flexible|lending|vip", re.I
)
_SWAP_TITLE = re.compile(r"(token|mainnet|coin) swap", re.I)
_MIGRATION_TITLE = re.compile(r"migrat|token merge", re.I)
_REBRAND_TITLE = re.compile(r"rebrand|redenominat|renam", re.I)
_TOKEN_EVENT_EXCLUDE = re.compile(
    r"system|account|wallet|bep2|convert|margin|loan|earn|has completed", re.I
)
_COMPLETED_TITLE = re.compile(r"has completed|completes|completed", re.I)
_COIN_MARGINED = re.compile(r"coin-?m(argined)?", re.I)
_SYMBOL_OK = re.compile(r"[A-Z0-9]+USDT")
_CONTRACT_LINE = re.compile(r"perpetual contract", re.I)
_FUTURES_ACTION = re.compile(
    r"close all|automatic settlement|suspend trading|will delist|delist this contract",
    re.I,
)
_USDT_MARGINED_BASE = re.compile(
    r"USDT-Margined\s+((?-i:[A-Z0-9]{2,15}))\s+Perpetual", re.I
)
_SLASH_PAIR_BASE = re.compile(
    r"(?<![A-Za-z0-9])((?-i:[A-Z0-9]{2,15}))\s*/\s*USDT\s+futures", re.I
)
_TITLE_BASE_STOP = frozenset(
    {"ON", "AND", "MORE", "USDT", "USD", "BUSD", "USDC", "FDUSD"}
)


def binance_cms_fetch(url: str) -> bytes:
    """One GET of the public Binance CMS API (no key): a browser User-Agent, bounded retries."""
    request = urllib.request.Request(
        url, headers={"User-Agent": NOTICE_USER_AGENT, "Accept": "application/json"}
    )
    last: Exception | None = None
    wait = 5.0
    for _ in range(8):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = bytes(response.read())
            time.sleep(_NOTICE_THROTTLE_SECONDS)
            return payload
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in (418, 429) and error.code < 500:
                break
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
        time.sleep(wait)
        wait = min(wait * 2.0, 120.0)
    raise ValueError(f"the Binance CMS request failed: {url}: {last}")


def _cms_json(fetch: NoticeFetch, url: str) -> dict[str, Any]:
    try:
        document = json.loads(fetch(url))
    except ValueError as error:
        raise ValueError(f"the Binance CMS response is not JSON: {url}") from error
    if not isinstance(document, dict) or not document.get("success", True):
        raise ValueError(f"the Binance CMS response is not a success: {url}")
    payload = document.get("data")
    if not isinstance(payload, dict):
        raise ValueError(f"the Binance CMS response has no data: {url}")
    return payload


def list_catalog(
    fetch: NoticeFetch, catalog: int, *, since_ms: int
) -> list[dict[str, Any]]:
    """Every article of a catalog released at or after ``since_ms`` (newest first)."""
    found: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"{CMS_ROOT}/list/query?type=1&catalogId={catalog}"
            f"&pageNo={page}&pageSize={NOTICE_PAGE_SIZE}"
        )
        catalogs = _cms_json(fetch, url).get("catalogs")
        if not isinstance(catalogs, list) or not catalogs:
            raise ValueError(f"catalog {catalog} page {page} has no catalog entry")
        articles = catalogs[0].get("articles")
        if not isinstance(articles, list):
            raise ValueError(f"catalog {catalog} page {page} has no article list")
        if not articles:
            break
        for item in articles:
            if not (
                isinstance(item, dict)
                and isinstance(item.get("code"), str)
                and isinstance(item.get("title"), str)
                and isinstance(item.get("releaseDate"), int)
            ):
                raise ValueError(
                    f"catalog {catalog} page {page} holds a malformed article"
                )
            found.append(item)
        if min(int(a["releaseDate"]) for a in articles) < since_ms:
            break
        total = catalogs[0].get("total")
        if isinstance(total, int) and page * NOTICE_PAGE_SIZE >= total:
            break
        page += 1
    return [a for a in found if int(a["releaseDate"]) >= since_ms]


def notice_kind(title: str) -> str | None:
    """The kind of a notice title that may announce a perpetual's end, or ``None``.

    ``delist``: a futures delisting, or a spot delisting whose body may carry the futures
    settlement (a postponement concerns the same event); ``swap`` / ``migration`` /
    ``rebrand``: a token swap, migration, redenomination or rename. Completion notices follow
    the event and are skipped. Margin-, loan-, Alpha- and earn-only titles are not notices.
    """
    if _COMPLETED_TITLE.search(title):
        return None
    futures_title = bool(_FUTURES_TITLE.search(title))
    if _DELIST_TITLE.search(title):
        if futures_title or not _OTHER_PRODUCT_DELIST.search(title):
            return "delist"
        return None
    if futures_title or not _TOKEN_EVENT_EXCLUDE.search(title):
        if _SWAP_TITLE.search(title):
            return "swap"
        if _MIGRATION_TITLE.search(title):
            return "migration"
        if _REBRAND_TITLE.search(title):
            return "rebrand"
    return None


def symbols_from_title(title: str, *, kind: str = "delist") -> list[str]:
    """USDⓈ-M perpetual symbols named by a title (explicit ``XUSDT`` tokens first).

    For a swap, migration or rename ``A to B`` only the old contract ``A`` (before the first
    standalone ``to``) is the symbol whose archive ends.
    """
    if not _FUTURES_TITLE.search(title):
        return []
    if kind != "delist":
        title = re.split(r"\bto\b", title, maxsplit=1, flags=re.I)[0]
    explicit = sorted(set(_USDT_TOKEN.findall(title)))
    if explicit:
        return explicit
    if _COIN_MARGINED.search(title) or not _USDM_TITLE.search(title):
        return []
    tail = re.split(r"delist|swap|migrat", title, maxsplit=1, flags=re.I)
    if len(tail) < 2:
        return []
    words = [
        w
        for w in _BASE_WORD.findall(tail[1])
        if len(w) >= 2
        and w == w.upper()
        and any(c.isalpha() for c in w)
        and w not in _TITLE_STOP
        and not w.endswith(("USD", "USDC", "USDT", "BUSD"))
    ]
    return [f"{base}USDT" for base in sorted(set(words))]


def _body_lines(node: Any) -> list[str]:
    """The text of each block of a CMS article body (a JSON tree of elements)."""
    blocks: list[str] = []

    def text(n: Any) -> str:
        if not isinstance(n, dict):
            return ""
        if n.get("node") == "text":
            return str(n.get("text", ""))
        return "".join(text(child) for child in n.get("child", []))

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        if n.get("tag") in ("p", "li", "td", "h1", "h2", "h3", "h4"):
            blocks.append(text(n))
            return
        for child in n.get("child", []):
            walk(child)

    walk(node)
    return blocks


def _body_lines_of(body: str) -> list[str]:
    try:
        return _body_lines(json.loads(body))
    except ValueError:
        return []


def title_bases(title: str) -> list[str]:
    """Upper-case tokens of a ``Will Delist A, B and C on <date>`` title."""
    tail = re.split(r"will delist", title, maxsplit=1, flags=re.I)[-1]
    tail = tail.split(" on ", 1)[0]
    words = [
        w
        for w in _BASE_WORD.findall(tail)
        if w == w.upper() and any(c.isalpha() for c in w) and w not in _TITLE_BASE_STOP
    ]
    return sorted(set(words))


def title_tickers(title: str) -> list[str]:
    """Parenthesised tickers of a swap title before its first standalone ``to``."""
    head = re.split(r"\bto\b", title, maxsplit=1, flags=re.I)[0]
    return re.findall(r"\(([A-Z0-9]{1,15})\)", head)


def symbols_from_body(body: str, *, title: str = "") -> list[str]:
    """``XUSDT`` symbols a body announces for a futures delisting, suspension or settlement.

    Three sources: ``Perpetual Contract`` lines naming ``XUSDT`` symbols; a ``Binance Futures
    will close all positions / suspend trading / delist`` statement naming ``XUSDT``,
    ``USDT-Margined X Perpetual`` or ``X/USDT futures``; and the statement about the
    ``aforementioned token(s)`` / ``contract(s)``, which stands for the tokens of a spot
    delisting title or the old ticker of a token-swap title.
    """
    found: set[str] = set()
    for line in _body_lines_of(body):
        if _CONTRACT_LINE.search(line):
            found.update(_USDT_TOKEN.findall(line))
        if re.search(r"futures", line, re.I) and _FUTURES_ACTION.search(line):
            found.update(_USDT_TOKEN.findall(line))
            found.update(
                f"{base.upper()}USDT" for base in _USDT_MARGINED_BASE.findall(line)
            )
            found.update(
                f"{base.upper()}USDT" for base in _SLASH_PAIR_BASE.findall(line)
            )
            if re.search(r"aforementioned (token|contract)", line, re.I):
                bases = (
                    title_bases(title)
                    if _DELIST_TITLE.search(title)
                    else title_tickers(title)[:1]
                )
                found.update(f"{base}USDT" for base in bases)
    return sorted(s for s in found if _SYMBOL_OK.fullmatch(s))


def fetch_article(fetch: NoticeFetch, code: str) -> dict[str, Any]:
    payload = _cms_json(fetch, f"{CMS_ROOT}/detail/query?articleCode={code}")
    if payload.get("code") != code or not isinstance(payload.get("title"), str):
        raise ValueError(f"article {code}: the detail response does not match the code")
    if not isinstance(payload.get("publishDate"), int) or not isinstance(
        payload.get("body"), str
    ):
        raise ValueError(f"article {code}: the detail response is malformed")
    return payload


def build_notice_document(
    fetch: NoticeFetch,
    *,
    catalogs: Sequence[int] = NOTICE_CATALOGS,
    since: datetime = NOTICE_SINCE,
    overrides: Sequence[dict[str, Any]] = (),
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """The canonical notice document: matched articles plus verified overrides.

    Result-blind: it reads only public announcements, never a market series or a result.
    """
    since_ms = int(since.timestamp()) * 1000
    entries: dict[tuple[str, str], notices.Notice] = {}
    scanned = fetched = 0
    seen: set[str] = set()
    for catalog in catalogs:
        articles = list_catalog(fetch, catalog, since_ms=since_ms)
        progress(f"catalog {catalog}: {len(articles)} articles")
        for article in articles:
            code = str(article["code"])
            if code in seen:
                continue
            seen.add(code)
            scanned += 1
            title = str(article["title"])
            kind = notice_kind(title)
            if kind is None:
                continue
            symbols = symbols_from_title(title, kind=kind)
            if not symbols:
                detail = fetch_article(fetch, code)
                fetched += 1
                symbols = symbols_from_body(str(detail["body"]), title=title)
            for symbol in symbols:
                entries[(symbol, code)] = notices.Notice(
                    symbol=symbol,
                    release_ms=int(article["releaseDate"]),
                    article_code=code,
                    title=title,
                    kind=kind,
                )
    override_entries: list[dict[str, object]] = []
    for item in overrides:
        override_entries.append(_verified_override(fetch, item))
        fetched += 1
    ordered = sorted(
        entries.values(), key=lambda n: (n.symbol, n.release_ms, n.article_code)
    )
    return {
        "schema_version": notices.SCHEMA_VERSION,
        "source": "binance_cms_public_article_api",
        "catalogs": list(catalogs),
        "since_utc": since.isoformat(),
        "articles_scanned": scanned,
        "details_fetched": fetched,
        "entries": [n.to_dict() for n in ordered],
        "overrides": sorted(
            override_entries,
            key=lambda o: (
                str(o["symbol"]),
                str(o["release_utc"]),
                str(o["article_code"]),
            ),
        ),
    }


def _verified_override(fetch: NoticeFetch, item: dict[str, Any]) -> dict[str, object]:
    """An override entry whose code, title and release time come from the real article."""
    if set(item) != {"symbol", "article_code", "kind", "evidence"} or not all(
        isinstance(v, str) and v for v in item.values()
    ):
        raise ValueError(
            "an override needs exactly symbol, article_code, kind and evidence strings"
        )
    detail = fetch_article(fetch, item["article_code"])
    base = item["symbol"].removesuffix("USDT")
    haystack = str(detail["title"]) + " " + " ".join(_body_lines_of(detail["body"]))
    if not re.search(rf"(?<![A-Za-z0-9]){re.escape(base)}(?![A-Za-z0-9])", haystack):
        raise ValueError(
            f"override {item['symbol']}: the article {item['article_code']} does not name it"
        )
    entry = notices.Notice(
        symbol=item["symbol"],
        release_ms=int(detail["publishDate"]),
        article_code=item["article_code"],
        title=str(detail["title"]),
        kind=item["kind"],
        override=True,
    ).to_dict()
    entry["evidence"] = item["evidence"]
    return entry


def notice_document_bytes(document: dict[str, Any]) -> bytes:
    """Deterministic UTF-8, LF, sorted-key bytes (the recorded file the digest pins)."""
    text = json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True)
    return (text + "\n").encode("utf-8")


def _notices_main(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    notice_fetch: NoticeFetch | None,
) -> int:
    """``notices``: record the public notice file (result-blind; network needs a flag)."""
    try:
        if notice_fetch is None and not args.allow_network:
            raise ValueError("notices needs --allow-network (public CMS API, no key)")
        if args.output.exists():
            raise FileExistsError(f"{args.output} exists; the file is write-once")
        overrides: list[dict[str, Any]] = []
        if args.overrides is not None:
            loaded = json.loads(args.overrides.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("--overrides must hold a JSON list")
            overrides = loaded
        document = build_notice_document(
            binance_cms_fetch if notice_fetch is None else notice_fetch,
            since=args.since,
            overrides=overrides,
            progress=lambda line: print(line, file=err),
        )
        raw = notice_document_bytes(document)
        notices.parse_notices(
            raw
        )  # the reader must accept the file before it is written
        args.output.write_bytes(raw)
    except (ValueError, OSError) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED
    out.write(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "entries": len(document["entries"]),
                "overrides": len(document["overrides"]),
                "articles_scanned": document["articles_scanned"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: GitState = tierd_driver.git_state,
    rows_factory: RowSourceFactory | None = None,
    now: Clock = lambda: datetime.now(UTC),
    gh: g4.GhRunner = g4.default_gh,
    rest_fetch: RestFetch | None = None,
    notice_fetch: NoticeFetch | None = None,
    notice_book: notices.NoticeBook | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    if args.command == "notices":
        return _notices_main(args, out=out, err=err, notice_fetch=notice_fetch)
    # Amendment 5: a real run closes a delisted name only from recorded public notices. The
    # notice file must verify against its pin before any clock, G4, file, row or output
    # access; otherwise the real run is refused. Injected synthetic row factories pass their
    # own (fake) notice book and never read a real archive.
    book = notice_book
    if args.command == "run" and rows_factory is None:
        try:
            book = notices.load_notices(args.delisting_notices)
        except (ValueError, OSError) as error:
            err.write(
                "refused: hindsight delisting forced closes are not authorized without "
                f"the pinned public notice file ({error}); no real economic run\n"
            )
            return EXIT_REFUSED
    moment = now()
    if args.command == "plan":
        try:
            document = plan_document(args.stage, moment)
        except (ValueError, protocol.ProtocolBindingError) as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    if args.command == "halts-from-coverage":
        try:
            result = halts_from_coverage(args.coverage)
        except (ValueError, OSError, KeyError) as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    if args.command == "coverage":
        return _coverage_main(
            args,
            out=out,
            err=err,
            state=state,
            rows_factory=rows_factory,
            moment=moment,
            gh=gh,
            rest_fetch=rest_fetch,
        )
    return _run_main(
        args,
        out=out,
        err=err,
        state=state,
        rows_factory=rows_factory,
        moment=moment,
        gh=gh,
        rest_fetch=rest_fetch,
        book=book,
    )


def _require_code_commit(snapshot: g4.CommentSnapshot, commit: str) -> None:
    """The G4 relay must quote the exact code commit the run is made at."""
    if not commit or commit not in snapshot.body:
        raise g4.G4AuthError("comment does not quote the exact code commit")


def _run_main(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    state: GitState,
    rows_factory: RowSourceFactory | None,
    moment: datetime,
    gh: g4.GhRunner,
    rest_fetch: RestFetch | None = None,
    book: notices.NoticeBook | None = None,
) -> int:
    try:
        mode = _check_mode(args, moment)
        core7_protocol.validate_authorization_url(args.authorization_url)
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        protocol.verify_candidates_binding()
        window = data.frozen_window(args.stage)
        limit = _stage_limit(args, window, moment, mode)
        dev = None if args.dev_result is None else _load_dev_result(args.dev_result)
        disjoint = (
            None
            if args.disjoint_decision is None
            else _load_disjoint_decision(args.disjoint_decision)
        )
        symbols = stage_symbols(args.stage, args.forward_symbols)
        # authenticate the G4 relay before any file, row or data is read
        digest = protocol.protocol_digest()
        first_fetch = g4.fetch_comment(args.authorization_url, gh)
        g4.require_protocol_digest(first_fetch, digest)
        g4.require_protocol_id(first_fetch, protocol.PROTOCOL_ID)
        commit, clean = state()
        _require_code_commit(first_fetch, commit)
        data_auth: g4.CommentSnapshot | None = None
        if _auth_url(args.stage) is not None:
            data_auth = g4.fetch_comment(protocol.DATA_AUTHORIZATION_URL, gh)
        if not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        perp_payload, perp_sha = tierd_driver._read_payload(args.exchange_info_perp)
        rules, fallback_rules = load_perp_rules(perp_payload, symbols)
        policy = embargo_for(limit, stage=args.stage)
        plan = build_plan(window, limit, moment, policy, symbols)
        if rows_factory is not None:
            detect = reader = rows_factory(plan)
        else:
            detect, reader = _sources(args, plan, policy)
        listing_sha = (
            read_symbol_file(
                protocol.forward_symbols_path()
                if args.forward_symbols is None
                else args.forward_symbols
            )[1]
            if args.stage == protocol.STAGE_FORWARD
            else None
        )
        rest = _rest_source(args, rest_fetch)
        try:
            hourly, coverage = load_stage_hourly(
                detect,
                reader,
                window=window,
                limit_close=limit,
                symbols=symbols,
                rest=rest,
            )
            evidence_fn = getattr(reader, "evidence", None)
            evidence = list(evidence_fn()) if callable(evidence_fn) else []
            fills_fn = getattr(reader, "fills", None)
            fills = list(fills_fn()) if callable(fills_fn) else []
            unusable_fn = getattr(reader, "unusable_days", None)
            unusable = list(unusable_fn()) if callable(unusable_fn) else []
        finally:
            for source in {id(detect): detect, id(reader): reader}.values():
                close = getattr(source, "close", None)
                if callable(close):
                    close()
        stage = build_stage(
            hourly,
            window=window,
            limit_close=limit,
            rules=rules,
            data_authorization_url=_auth_url(args.stage),
            listing=(
                None
                if listing_sha is None
                else {"symbol_file_sha256": listing_sha, "symbols": len(symbols)}
            ),
            notice_book=book,
        )
        data.validate_stage_dataset(stage, window)
        # re-fetch immediately before the run: an edited comment authorizes nothing
        auth_receipt = g4.confirm_unedited(
            args.authorization_url, first_fetch, gh, protocol_digest=digest
        )
        if data_auth is not None:
            again = g4.fetch_comment(protocol.DATA_AUTHORIZATION_URL, gh)
            if (
                again.body_sha256 != data_auth.body_sha256
                or again.updated_at != data_auth.updated_at
            ):
                raise g4.G4AuthError(
                    "the data-authorization comment was edited after it was first fetched"
                )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        dataset = stage.dataset
        data_receipt: dict[str, object] = {
            "schema_version": DATA_RECEIPT_SCHEMA,
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_digest": digest,
            "candidates_sha256": protocol.CANDIDATES_SHA256,
            "harness_source_digest": protocol.harness_source_digest(),
            "stage": args.stage,
            "mode": mode,
            "window": window.to_dict(),
            "read_limit_close": limit.isoformat(),
            "dataset_id": dataset.dataset_id,
            "legs": list(dataset.symbols),
            "membership_sha256": stage.member_sha256(),
            "universe_months": {
                start.isoformat(): month.to_dict()
                for start, month in sorted(stage.universe.items())
            },
            "symbols_requested": len(symbols),
            "symbols_with_data": len(hourly),
            "source": "vision_usdm_1h_and_funding",
            "exchange_info_sha256": {"perp": perp_sha},
            "exchange_rules_note": RULES_NOTE,
            "symbols_on_fallback_rules": fallback_rules,
            "forward_symbol_file_sha256": listing_sha,
            "listing_coverage": coverage,
            "unpublished_funding_months": [
                record
                for entry in coverage.values()
                for record in entry.get("unpublished_funding_months", [])  # type: ignore[attr-defined]
            ],
            "archive_evidence_count": len(evidence),
            "archive_evidence_digest": (
                tierd_driver._digest(evidence) if evidence else None
            ),
            "zombie_tail": {
                symbol: {
                    "zombie_tail_bars": entry.get("zombie_tail_bars"),
                    "effective_end": entry.get("effective_end"),
                }
                for symbol, entry in coverage.items()
                if entry.get("zombie_tail_bars")
            },
            "relisting_gaps": {
                symbol: entry["relisting_gap_runs"]
                for symbol, entry in coverage.items()
                if entry.get("relisting_gap_runs")
            },
            "rest_funding": {
                "months": rest.evidence(),
                "digest": (
                    tierd_driver._digest(rest.evidence()) if rest.evidence() else None
                ),
            },
            "daily_fallback": {
                "enabled": bool(args.daily_fallback),
                "filled_hours": len(fills),
                "fills_digest": tierd_driver._digest(fills) if fills else None,
                "fills": fills,
                "forward_fill": False,
                "rest_fallback": False,
                "unusable_daily_archives": unusable,
            },
            "delisting_notices": {
                "sha256": protocol.DELISTING_NOTICES_SHA256,
                "events": [dict(item) for item in stage.notice_evidence],
            },
            "declared_halts": {
                "intervals": [list(pair) for pair in protocol.DECLARED_HALTS],
                "coverage_json_sha256": protocol.COVERAGE_EVIDENCE_SHA256,
            },
            "cash_account": {
                "initial_margin_rate": protocol.INITIAL_MARGIN_RATE,
                "limit": protocol.CASH_LIMIT,
                "net_exposure_limit": protocol.NET_EXPOSURE_LIMIT,
                "invalid_on": protocol.REASON_NEGATIVE_CASH,
            },
            "data_authorization": (
                None
                if data_auth is None
                else {
                    "url": protocol.DATA_AUTHORIZATION_URL,
                    "author_login": data_auth.login,
                    "author_id": data_auth.user_id,
                    "body_sha256": data_auth.body_sha256,
                    "updated_at": data_auth.updated_at,
                    "embargo_policy": type(policy).__name__,
                }
            ),
            "dev_result": dev,
            "disjoint_decision": disjoint,
            "code_commit": commit,
            "working_tree_clean": True,
            "run_at": moment.isoformat(),
        }
        (args.output_dir / DATA_RECEIPT_NAME).write_bytes(
            canonical_json_bytes(data_receipt)
        )
        result = runner.run_stage(
            stage,
            window=window,
            output_dir=args.output_dir,
            expected_protocol_digest=digest,
            mode=mode,
            disjoint_label=None if disjoint is None else disjoint["label"],
            code_commit=commit,
            authorization={
                **auth_receipt.to_dict(),
                "g4_comment_updated_at": first_fetch.updated_at,
            },
            run_date=moment.date(),
        )
    except g4.G4AuthError as error:
        err.write(f"authorization failed: {error}\n")
        return EXIT_UNAUTHENTICATED
    except CarryScreenHalt as error:
        state_word = "resumable" if error.resumable else "not resumable"
        err.write(f"halted ({error.kind}, {state_word}): {error}\n")
        return EXIT_HALTED
    except (
        ValueError,
        protocol.ProtocolBindingError,
        SealedWindowEmbargoError,
        FileExistsError,
        ExchangeInfoError,
        OSError,
    ) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED
    label = result["label"]
    run_receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": digest,
        "stage": args.stage,
        "mode": mode,
        "dataset_id": stage.dataset.dataset_id,
        "authorization_receipt": auth_receipt.to_dict(),
        "g4_comment_updated_at": first_fetch.updated_at,
        "code_commit": commit,
        "label": label,
        "status": result.get("status"),
    }
    (args.output_dir / RUN_RECEIPT_NAME).write_bytes(canonical_json_bytes(run_receipt))
    out.write(
        json.dumps(
            {
                "stage": args.stage,
                "mode": mode,
                "label": label,
                "status": result.get("status"),
                "failed_conditions": result.get("failed_conditions"),
                "invalid_cells": result.get("invalid_cells"),
                "dataset_id": stage.dataset.dataset_id,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if label == protocol.LABEL_INVALID or result.get("invalid_cells"):
        err.write("not successful: a cell is invalid (INVALID)\n")
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
