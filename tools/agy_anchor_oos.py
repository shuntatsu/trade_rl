"""Driver for the out-of-sample test ``agy_anchor_realistic_v1``.

``python -m tools.agy_anchor_oos plan --stage DEV|BACKCAST_2020|FORWARD``
``python -m tools.agy_anchor_oos coverage --stage ...``
``python -m tools.agy_anchor_oos halts-from-coverage --coverage coverage.json``
``python -m tools.agy_anchor_oos run --stage DEV``
``python -m tools.agy_anchor_oos run --stage BACKCAST_2020 --dev-result DEV/dev-result.json``
``python -m tools.agy_anchor_oos run --stage FORWARD --interim | --final``

The wiring mirrors :mod:`tools.trend_oos` (whose fetchers, embargo policy and fallback source
it imports) with spot **and** USD-M perpetual klines, perpetual funding and the taker-buy
quote volume of the same 1h klines.

``plan`` prints the frozen protocol, its digest and the windows; it loads no data.

``coverage`` is the result-blind data step before any run: it acquires the stage's archives
through the same source path and writes ``coverage.json`` (listing start, missing hour ranges
after the daily fallback, fills, placeholders, checksums, funding counts, exchange-wide spot
gaps). It builds no signal, replay or return and never refuses on a gap.

``run`` authenticates the owner authorization comment (G4: owner author, this protocol's id
and digest quoted, unedited between two fetches), requires a clean checkout and a new output
directory, builds the stage dataset, records a data receipt and writes every cell before the
result. The run order is enforced: BACKCAST_2020 needs the DEV result, FORWARD final needs the
BACKCAST_2020 decision of the same digest (whatever its label). ``--final`` is refused before
2027-05-02T00:00Z; ``--interim`` writes a dated ``INTERIM_NOT_A_DECISION`` report. The driver
authorizes nothing beyond data reads and the evaluation: no orders, no API keys.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from tools import core7_g4_comment as g4
from tools import core7_mtf_tierd as tierd_driver
from tools import trend_oos as base
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.agy_anchor_oos import data, decision, protocol, runner
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.trend_oos.data import StageEmbargoError
from trade_rl.integrations.binance import vision_funding_url
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryScreenHalt,
    parse_carry_perp_rows,
    parse_carry_spot_rows,
)
from trade_rl.integrations.market_store.carry_source import (
    CarryRowSource,
    VisionArchiveSource,
    _check_order,
    _gap_error,
    _missing_hours,
    _vision_rows,
    funding_coverage_gap,
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
from trade_rl.integrations.market_store.parse import (
    ArchiveContentError,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import (
    DEFAULT_MONTHLY_PUBLICATION_LAG,
    IngestionPlan,
    _month_start,
    _next_month,
    plan_ingestion,
)
from trade_rl.integrations.market_store.rules import ExchangeInfoError

EXIT_OK = base.EXIT_OK
EXIT_REFUSED = base.EXIT_REFUSED
EXIT_HALTED = base.EXIT_HALTED
EXIT_INVALID = base.EXIT_INVALID
EXIT_UNAUTHENTICATED = base.EXIT_UNAUTHENTICATED
COVERAGE_NAME = "coverage.json"
COVERAGE_SCHEMA = "agy_anchor_oos_coverage_v1"
DATA_RECEIPT_NAME = "agy-anchor-oos-data-receipt.json"
RUN_RECEIPT_NAME = "agy-anchor-oos-run-receipt.json"
DATA_RECEIPT_SCHEMA = "agy_anchor_oos_data_receipt_v1"
RUN_RECEIPT_SCHEMA = "agy_anchor_oos_run_receipt_v1"
RULES_NOTE = (
    "exchange rules are one static retrieval-time snapshot applied to every row of the "
    "stage; they are not point-in-time"
)
_MAX_REPORTED_GAP_RUNS = 25
_MARKETS = (StoreMarket.SPOT, StoreMarket.USDS_M)

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


# ----------------------------------------------------------------------- loading


def _ceil_month(value: datetime) -> datetime:
    floor = _month_start(value)
    return floor if floor == value else _next_month(floor)


@dataclass(frozen=True, slots=True)
class AnchorForwardEmbargoPolicy(EmbargoPolicy):
    """Market-store embargo for FORWARD reads, bound to this protocol's owner relay.

    The Tier D / core7 embargo and ``tools.trend_oos.ForwardEmbargoPolicy`` (bound to the
    older relay) are untouched. It admits archives strictly before ``admit_before``, a UTC
    midnight: the day after the read limit, or the end of the month holding it when the
    monthly funding archive needs that (see :func:`embargo_for`).
    """

    admit_before: datetime = datetime(2026, 8, 1, tzinfo=UTC)

    def __post_init__(self) -> None:
        EmbargoPolicy.__post_init__(self)
        if self.warmup_authorization_url != protocol.DATA_AUTHORIZATION_URL:
            raise ValueError(
                "the FORWARD embargo path is bound to the owner authorization "
                f"{protocol.DATA_AUTHORIZATION_URL}"
            )
        moment = self.admit_before
        if moment.tzinfo is None or moment != moment.replace(
            hour=0, minute=0, second=0, microsecond=0
        ):
            raise ValueError("admit_before must be a UTC midnight")

    @property
    def sealed_start(self) -> datetime:
        return self.admit_before

    @property
    def warmup_start(self) -> datetime:
        return self.admit_before


def embargo_for(limit_close: datetime, *, stage: str) -> EmbargoPolicy:
    """The stage's embargo policy: the enforced default before 2026-08-01, else FORWARD's.

    Funding exists only as monthly archives, and the month that holds the read limit ends
    after it. The FORWARD policy therefore admits archives up to the end of that month (never
    further); the reader discards every row after the limit, and the row guard refuses them.
    """
    admit_before = _ceil_day(limit_close)
    if admit_before <= data.DEFAULT_EMBARGO:
        return EmbargoPolicy.enforced()
    if stage != protocol.STAGE_FORWARD:
        raise StageEmbargoError("only FORWARD may read past the default embargo")
    return AnchorForwardEmbargoPolicy(
        warmup_authorization_url=protocol.DATA_AUTHORIZATION_URL,
        admit_before=max(admit_before, _ceil_month(limit_close)),
    )


def _auth_url(stage: str) -> str | None:
    """The owner relay a stage's reads are bound to (DEV reads known-used data: none)."""
    return (
        protocol.DATA_AUTHORIZATION_URL if stage in protocol.AUTHORIZED_STAGES else None
    )


def funding_boundary(now: datetime) -> datetime:
    """The end of the last funding month that has a published monthly archive.

    Binance Vision publishes ``fundingRate`` only as monthly archives (there is no daily
    funding archive), and the market-store planner uses a month once it is complete and
    ``DEFAULT_MONTHLY_PUBLICATION_LAG`` old. Funding after this instant does not exist in
    the archives; nothing is invented or fetched from elsewhere to cover it.
    """
    return _month_start(_floor_day(now) - DEFAULT_MONTHLY_PUBLICATION_LAG)


def require_funding_published(limit_close: datetime, now: datetime) -> None:
    """Refuse a read limit whose last funding month is not published yet."""
    needed = _ceil_month(limit_close)
    boundary = funding_boundary(now)
    if needed > boundary:
        raise StageEmbargoError(
            f"the USD-M funding archive of the month ending {needed.isoformat()} is not "
            f"published yet (monthly archives are used {DEFAULT_MONTHLY_PUBLICATION_LAG.days} "
            f"days after the month ends; funding is published through {boundary.isoformat()}); "
            "funding is never invented or taken from another source"
        )


def build_plan(
    window: data.StageWindow,
    limit_close: datetime,
    now: datetime,
    policy: EmbargoPolicy,
) -> IngestionPlan:
    """Hourly spot and USD-M klines, and monthly funding of every perp over the read range.

    The market-store planner plans a funding month only when it ends inside
    ``[start, ceil_day(limit))``. The stage reads funding up to the read limit, which usually
    lies inside a month, so the month that holds the limit is planned here as well (its
    rows after the limit are discarded by the reader). A month that is not published yet
    refuses (:func:`require_funding_published`).
    """
    universe = Universe(
        symbols=protocol.ROSTER, interval=data.BASE_INTERVAL, markets=_MARKETS
    )
    plan = plan_ingestion(
        universe,
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
    for symbol in protocol.ROSTER:
        month = _month_start(window.load_start)
        while month < needed_end:
            following = _next_month(month)
            if (symbol, month) not in have:
                url = vision_funding_url(StoreMarket.USDS_M.binance, symbol, month)
                extra.append(
                    ArchiveSpec(
                        kind=ArchiveKind.FUNDING_RATE,
                        market=StoreMarket.USDS_M,
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


def _probe(reader: Any, market: StoreMarket, symbol: str, day_ms: int) -> bool:
    probe = getattr(reader, "probe_published_day", None)
    return bool(probe(market, symbol, day_ms)) if callable(probe) else False


def _hourly_taker(rows: Sequence[Any], start_ms: int, end_ms: int) -> np.ndarray:
    """Taker-buy quote volume per hour of ``[start_ms, end_ms)``; 0 for an absent hour."""
    by_open = {int(row.open_time_ms): float(row.taker_buy_quote_volume) for row in rows}
    return np.asarray(
        [by_open.get(h, 0.0) for h in range(start_ms, end_ms, data.HOUR_MS)],
        dtype=np.float64,
    )


def load_leg(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    market: StoreMarket,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
) -> tuple[data.HourlyLeg | None, dict[str, object]]:
    """The complete hourly series (and taker-buy) of one leg from its first listed 4h bar.

    ``detect`` finds the first listed hour: only a *leading* run of unpublished monthly
    archives (HTTP 404) counts as "not listed yet"; an unpublished archive after the first
    row, an unexplained gap or any other failure raises. ``reader`` then serves the complete
    range from the first 4h-aligned listed hour (it may fill hours from checksum-verified
    daily archives). Spot halts are the only explained gaps; a perpetual is strict. Nothing is
    filled forward or invented; a leg with no rows in the window returns ``None``.
    """
    is_perp = market is StoreMarket.USDS_M
    start_ms, end_ms = ms_from_utc(window.load_start), ms_from_utc(limit_close)
    first_open: int | None = None
    unpublished_months = 0
    leading: list[tuple[int, int]] = []
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(market, symbol, data.BASE_INTERVAL, lo, hi)
        except CarryCoverageError as error:
            if (
                first_open is None
                and (error.evidence or {}).get("kind") == "archive_not_published"
            ):
                unpublished_months += 1
                leading.append((lo, hi))
                continue
            raise
        except CarryIntegrityHalt as error:
            if (error.evidence or {}).get("kind") != "malformed_archive":
                raise
            rows = reader.read_klines(market, symbol, data.BASE_INTERVAL, lo, hi)
        if rows:
            first_open = int(rows[0].open_time_ms)
            break
        unpublished_months += 1
    for lo, hi in leading:
        day = hi - 24 * data.HOUR_MS
        if _probe(reader, market, symbol, day):
            raise data.StageDataError(
                f"{symbol}:{market.value}: the monthly archive of the leading month "
                f"starting {_iso(lo)} is unpublished but a daily archive exists "
                f"({_iso(day)}); the listing would silently move"
            )
    coverage: dict[str, object] = {
        "unpublished_leading_months": unpublished_months,
        "first_listed_open": None,
        "first_bar_open": None,
        "dropped_leading_partial_hours": 0,
        "rows": 0,
    }
    if first_open is None:
        return None, coverage
    bar_ms = data.BAR_MS
    aligned = (
        first_open if first_open % bar_ms == 0 else (first_open // bar_ms + 1) * bar_ms
    )
    coverage["first_listed_open"] = _iso(first_open)
    coverage["dropped_leading_partial_hours"] = (aligned - first_open) // data.HOUR_MS
    if aligned >= end_ms:
        return None, coverage
    coverage["first_bar_open"] = _iso(aligned)
    rows = reader.read_klines(market, symbol, data.BASE_INTERVAL, aligned, end_ms)
    opens = [int(row.open_time_ms) for row in rows]
    _check_order(opens, market=market, symbol=symbol, what="klines")
    explained: list[tuple[int, int]] = []
    if not is_perp:
        for h_lo, h_hi in protocol.declared_halt_intervals():
            first = bisect.bisect_left(opens, ms_from_utc(h_lo))
            if first < len(opens) and opens[first] + data.HOUR_MS <= ms_from_utc(h_hi):
                raise data.StageDataError(
                    f"declared halt {h_lo.isoformat()}..{h_hi.isoformat()} is not "
                    f"exchange-wide: {symbol} has a published row at {_iso(opens[first])}"
                )
        explained = [(ms_from_utc(lo), ms_from_utc(hi)) for lo, hi in halt_intervals]
    gaps = _missing_hours(opens, aligned, end_ms, explained=explained)
    if gaps:
        gap = _gap_error(market, symbol, gaps)
        evidence = dict(gap.evidence or {})
        evidence["gap_runs"] = tierd_driver._ranges(gaps)[:_MAX_REPORTED_GAP_RUNS]
        gap.evidence = evidence
        raise gap
    start_time = datetime.fromtimestamp(aligned / 1_000, UTC)
    funding_events = 0
    try:
        if is_perp:
            funding = [
                row
                for row in reader.read_funding(market, symbol, aligned, end_ms)
                if row.funding_time_ms < end_ms
            ]
            _check_order(
                [row.funding_time_ms for row in funding],
                market=market,
                symbol=symbol,
                what="funding",
            )
            events = [(row.funding_time_ms, float(row.rate)) for row in funding]
            funding_events = len(events)
            series: RawMarketSeries = parse_carry_perp_rows(
                _vision_rows(rows), events, start_time=start_time, end_time=limit_close
            )
        else:
            series = parse_carry_spot_rows(
                _vision_rows(rows),
                start_time=start_time,
                end_time=limit_close,
                halt_intervals=halt_intervals,
            )
    except ValueError as error:
        raise CarryIntegrityHalt(
            f"{symbol}: {error}",
            {"kind": "parse_failure", "symbol": symbol, "detail": str(error)},
        ) from error
    if is_perp:
        counts = np.asarray(series.funding_event_count)
        if np.any(counts > 1):
            raise CarryIntegrityHalt(
                f"{symbol}: two funding events settle in one hourly bar",
                {"kind": "duplicate_funding", "symbol": symbol},
            )
        reason = funding_coverage_gap(counts)
        if reason is not None:
            raise CarryCoverageError(
                f"{symbol}: {reason}",
                {"kind": "funding_gap", "symbol": symbol, "detail": reason},
            )
        coverage["funding_events"] = funding_events
    coverage["rows"] = len(rows)
    taker = _hourly_taker(rows, aligned, end_ms)
    return data.HourlyLeg(series=series, taker_buy_quote=taker), coverage


def load_stage_data(
    detect: CarryRowSource,
    reader: CarryRowSource,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    rules: dict[str, Any],
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    data_authorization_url: str | None = None,
) -> tuple[data.StageData, dict[str, dict[str, object]]]:
    """Read every roster leg through the stage guard and assemble the stage."""
    if halt_intervals is None:
        halt_intervals = protocol.stage_halt_intervals()
    guarded_detect = data.StageRowSource(detect, window=window, limit_close=limit_close)
    guarded_reader = data.StageRowSource(reader, window=window, limit_close=limit_close)
    legs: dict[StoreMarket, dict[str, data.HourlyLeg | None]] = {
        StoreMarket.SPOT: {},
        StoreMarket.USDS_M: {},
    }
    coverage: dict[str, dict[str, object]] = {}
    for symbol in protocol.ROSTER:
        coverage[symbol] = {}
        for market in _MARKETS:
            leg, entry = load_leg(
                guarded_detect,
                guarded_reader,
                symbol,
                market,
                window=window,
                limit_close=limit_close,
                halt_intervals=halt_intervals,
            )
            legs[market][symbol] = leg
            coverage[symbol][market.value] = entry
    stage = data.assemble_stage_dataset(
        window,
        legs[StoreMarket.SPOT],
        legs[StoreMarket.USDS_M],
        rules=rules,
        limit_close=limit_close,
        halt_intervals=halt_intervals,
        data_authorization_url=data_authorization_url,
    )
    return stage, coverage


# ---------------------------------------------------------------- sources


class _AnchorFallbackSource(base._MonthFallbackSource):
    """The month/day fallback with a market-aware published-day probe."""

    def probe_published_day(  # type: ignore[override]
        self, market: StoreMarket, symbol: str, day_start_ms: int
    ) -> bool:
        got = self._fill_hours(
            market,
            symbol,
            data.BASE_INTERVAL,
            [day_start_ms],
            source="probe",
            tolerate_malformed=True,
            record=False,
        )
        return bool(got)


def _sources(
    args: argparse.Namespace, plan: IngestionPlan, policy: EmbargoPolicy
) -> tuple[CarryRowSource, CarryRowSource]:
    """``(detect, reader)``: the primary source and, with the fallback, its wrapper."""
    factory = base._vision_factory(args.archive_dir, args.allow_network, policy)
    primary = factory(plan)
    if not args.daily_fallback:
        return primary, primary
    daily = base._vision_factory(args.archive_dir, args.allow_network, policy)
    return primary, _AnchorFallbackSource(
        primary, daily, halts=protocol.stage_halt_intervals(), embargo=policy
    )


# ----------------------------------------------------------------- CLI plumbing


def plan_document(stage: str, now: datetime) -> dict[str, Any]:
    """Protocol, digest and windows; reads no data."""
    window = data.frozen_window(stage)
    protocol.verify_fixture_binding()
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "port_source_digest": protocol.port_source_digest(),
        "parity_fixture_sha256": protocol.FIXTURE_SHA256,
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
        prog="python -m tools.agy_anchor_oos",
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
        help="result-blind data-coverage report; builds no signal, replay or return",
    )
    cov.add_argument("--stage", choices=protocol.STAGES, required=True)
    cov.add_argument("--through", type=_parse_instant, default=None)
    cov.add_argument("--archive-dir", type=Path, required=True)
    cov.add_argument("--allow-network", action="store_true")
    cov.add_argument("--daily-fallback", action="store_true")
    cov.add_argument("--output-dir", type=Path, required=True)
    halts = commands.add_parser(
        "halts-from-coverage",
        help="derive the exchange-wide halt list (and its evidence SHA-256) from a "
        "coverage.json; reads no market data",
    )
    halts.add_argument("--coverage", type=Path, required=True)
    run = commands.add_parser(
        "run", help="run one stage (needs the owner authorization comment)"
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
        help="BACKCAST_2020: the dev-result.json of the DEV run (the run order)",
    )
    run.add_argument(
        "--backcast-decision",
        type=Path,
        default=None,
        help="FORWARD --final: the BACKCAST_2020 decision.json of the same digest",
    )
    run.add_argument("--archive-dir", type=Path, required=True)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--daily-fallback", action="store_true")
    run.add_argument("--exchange-info-spot", type=Path, required=True)
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
            f"the DEV result has invalid cells {invalid}: the port is not fit to run "
            "(DEV may stop the line); BACKCAST_2020 is refused"
        )
    return {"sha256": document["__sha256"], "status": document.get("status")}


def _load_backcast_decision(path: Path) -> dict[str, Any]:
    document = _load_prior(
        path, stage=protocol.STAGE_BACKCAST, what="BACKCAST_2020 decision"
    )
    verdict = document.get("stage_verdict")
    if verdict not in decision.VERDICTS:
        raise ValueError("the BACKCAST_2020 decision carries no stage verdict")
    return {"stage_verdict": verdict, "sha256": document["__sha256"]}


def _check_mode(args: argparse.Namespace, now: datetime) -> str:
    """The run mode; refuses every stage/mode/order combination the protocol forbids."""
    if args.stage != protocol.STAGE_FORWARD:
        if args.final or args.interim or args.through is not None:
            raise ValueError(f"{args.stage} takes no --final/--interim/--through")
        if args.backcast_decision is not None:
            raise ValueError("--backcast-decision belongs to FORWARD --final")
        if args.stage == protocol.STAGE_DEV:
            if args.dev_result is not None:
                raise ValueError("--dev-result belongs to BACKCAST_2020")
        elif args.dev_result is None:
            raise ValueError("BACKCAST_2020 needs --dev-result (DEV runs first)")
        return runner.MODE_DECISION
    if args.dev_result is not None:
        raise ValueError("--dev-result belongs to BACKCAST_2020")
    if args.final == args.interim:
        raise ValueError("FORWARD needs exactly one of --final and --interim")
    if args.interim:
        if args.backcast_decision is not None:
            raise ValueError("--backcast-decision is for --final only")
        return runner.MODE_INTERIM
    if args.through is not None:
        raise ValueError("--through is for --interim only")
    if args.backcast_decision is None:
        raise ValueError("FORWARD --final needs --backcast-decision")
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


class _CoverageVisionSource(VisionArchiveSource):
    """``VisionArchiveSource`` that also logs the placeholder rows the parser skips."""

    def __init__(self, fetcher: Any, plan: IngestionPlan, log: PlaceholderLog) -> None:
        super().__init__(fetcher, plan)
        self._log = log

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        rows: list[Any] = []
        for spec in self._selected(
            ArchiveKind.KLINES, market, symbol, start_ms, end_ms
        ):
            if spec.interval != interval:
                raise CarryIntegrityHalt(
                    f"planned interval differs: {spec.url}",
                    {"kind": "plan_interval_mismatch", "url": spec.url},
                )
            payload = self._payload(spec)
            try:
                parsed = parse_kline_archive(payload, spec, embargo=self._embargo)
            except ArchiveContentError as error:
                raise self._malformed(spec, error) from error
            for item in parsed.placeholders:
                self._log.append(
                    (spec.symbol, item.open_time_ms, item.close_time_ms, spec.url)
                )
            rows.extend(
                row for row in parsed.rows if start_ms <= row.open_time_ms < end_ms
            )
        return rows


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
        return _CoverageVisionSource(
            base.ArchiveDirFetcher(args.archive_dir, inner), inner_plan, log
        )

    primary = factory(plan)
    if not args.daily_fallback:
        return primary, primary, log
    fallback = _AnchorFallbackSource(
        primary, factory, halts=protocol.stage_halt_intervals(), embargo=policy
    )
    return primary, fallback, log


def _leg_coverage(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    market: StoreMarket,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
) -> tuple[dict[str, Any], int | None, set[int]]:
    """Coverage facts of one leg; never raises on a gap, a 404 or a bad archive."""
    is_perp = market is StoreMarket.USDS_M
    start_ms, end_ms = ms_from_utc(window.load_start), ms_from_utc(limit_close)
    errors: list[dict[str, object]] = []
    first_open: int | None = None
    unpublished_leading = 0
    leading_with_dailies: list[str] = []
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(market, symbol, data.BASE_INTERVAL, lo, hi)
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") == "archive_not_published":
                unpublished_leading += 1
                if _probe(reader, market, symbol, hi - 24 * data.HOUR_MS):
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
                        market, symbol, data.BASE_INTERVAL, lo, hi
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
        return entry, None, set()
    present: set[int] = set()
    zero_volume: list[int] = []
    unread: list[dict[str, object]] = []
    for lo, hi in _month_chunks(first_open, end_ms):
        span = {"from": _iso(lo), "to": _iso(hi)}
        try:
            rows = reader.read_klines(market, symbol, data.BASE_INTERVAL, lo, hi)
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
    halts = (
        []
        if is_perp
        else [(ms_from_utc(lo), ms_from_utc(hi)) for lo, hi in halt_intervals]
    )
    missing: list[int] = []
    explained = 0
    for index, opened in enumerate(range(first_open, end_ms, data.HOUR_MS)):
        if opened in present:
            continue
        if index > 0 and any(
            lo <= opened and opened + data.HOUR_MS <= hi for lo, hi in halts
        ):
            explained += 1
        else:
            missing.append(opened)
    entry.update(
        {
            "expected_hours": (end_ms - first_open) // data.HOUR_MS,
            "present_hours": len(present),
            "missing_hours": len(missing),
            "missing_ranges": _hour_ranges(missing),
            "missing_explained_by_declared_halt_hours": explained,
            "zero_volume_regular_rows": _hour_ranges(zero_volume),
            "unread_chunks": unread,
        }
    )
    if is_perp:
        entry["funding"] = _funding_coverage(reader, symbol, first_open, end_ms, errors)
    return entry, first_open, set(missing)


def _funding_coverage(
    reader: CarryRowSource,
    symbol: str,
    first_open: int,
    end_ms: int,
    errors: list[dict[str, object]],
) -> dict[str, object]:
    try:
        rows = [
            r
            for r in reader.read_funding(StoreMarket.USDS_M, symbol, first_open, end_ms)
            if r.funding_time_ms < end_ms
        ]
    except CarryScreenHalt as error:
        errors.append({"phase": "funding", **_halt_error(error)})
        return {"events": None}
    stamps = [int(r.funding_time_ms) for r in rows]
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    return {
        "events": len(stamps),
        "first_event": None if not stamps else _iso(stamps[0]),
        "last_event": None if not stamps else _iso(stamps[-1]),
        "max_gap_hours": None if not gaps else max(gaps) / data.HOUR_MS,
    }


def _exchange_wide(
    firsts: dict[str, int], missing: dict[str, set[int]]
) -> dict[str, object]:
    """Spot hours missing for every symbol listed then (at least two listed)."""
    hours: list[int] = []
    union: set[int] = set().union(*missing.values()) if missing else set()
    for hour in sorted(union):
        listed = [name for name, first in firsts.items() if first <= hour]
        if len(listed) >= 2 and all(hour in missing[name] for name in listed):
            hours.append(hour)
    return {
        "rule": "unexplained missing spot hours shared by every symbol listed then "
        "(at least two listed)",
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
    halt_intervals: Sequence[tuple[datetime, datetime]],
) -> dict[str, Any]:
    """Per-leg listing, gaps, fills, placeholders, funding and checksum results.

    Rows are read through the stage guard only to validate and count them; this builds no
    dataset, no signal, no replay and no return, and never refuses because of a gap.
    """
    guarded_detect = data.StageRowSource(detect, window=window, limit_close=limit_close)
    guarded_reader = data.StageRowSource(reader, window=window, limit_close=limit_close)
    symbols: dict[str, Any] = {}
    firsts: dict[str, int] = {}
    missing: dict[str, set[int]] = {}
    for symbol in protocol.ROSTER:
        symbols[symbol] = {}
        for market in _MARKETS:
            entry, first, gone = _leg_coverage(
                guarded_detect,
                guarded_reader,
                symbol,
                market,
                window=window,
                limit_close=limit_close,
                halt_intervals=halt_intervals,
            )
            symbols[symbol][market.value] = entry
            if market is StoreMarket.SPOT and first is not None:
                firsts[symbol] = first
                missing[symbol] = gone
    evidence_fn = getattr(reader, "evidence", None)
    evidence = list(evidence_fn()) if callable(evidence_fn) else []
    fills_fn = getattr(reader, "fills", None)
    fills = list(fills_fn()) if callable(fills_fn) else []
    unusable_fn = getattr(reader, "unusable_days", None)
    unusable = list(unusable_fn()) if callable(unusable_fn) else []
    for symbol, markets in symbols.items():
        for market_name, entry in markets.items():
            entry["unusable_daily_archives"] = [
                u
                for u in unusable
                if u.get("symbol") == symbol and u.get("market") == market_name
            ]
            mine = [
                e
                for e in evidence
                if e.get("symbol") == symbol
                and (
                    market_name == StoreMarket.USDS_M.value
                    if "/futures/um/" in str(e.get("url"))
                    else market_name == StoreMarket.SPOT.value
                )
            ]
            failures = [
                c
                for c in entry.get("unread_chunks", [])
                if c.get("detail") in base._INTEGRITY_DETAILS
            ] + [
                e for e in entry["errors"] if e.get("detail") in base._INTEGRITY_DETAILS
            ]
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
            filled = [
                f
                for f in fills
                if f.get("symbol") == symbol and f.get("market") == market_name
            ]
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
    placeholder_rows = sorted(set(placeholders))
    for symbol, markets in symbols.items():
        for entry in markets.values():
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
        "symbols": symbols,
        "exchange_wide_spot_gaps": _exchange_wide(firsts, missing),
        "archive_evidence_count": len(evidence),
        "archive_evidence_digest": tierd_driver._digest(evidence) if evidence else None,
        "daily_fallback_filled_hours": len(fills),
        "registered_halts": [
            [_iso(ms_from_utc(a)), _iso(ms_from_utc(b))]
            for a, b in protocol.stage_halt_intervals()
        ],
    }


def halts_from_coverage(path: Path) -> dict[str, Any]:
    """The exchange-wide spot ranges of a coverage.json as a declared halt list.

    Only the report's exchange-wide spot ranges qualify. It also lists, per leg, the missing
    ranges that would remain (and make a run refuse). It refuses a report with read errors
    and computes nothing from prices.
    """
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != COVERAGE_SCHEMA:
        raise ValueError("not an agy_anchor_oos coverage report")
    if document.get("stage") != protocol.STAGE_BACKCAST:
        raise ValueError("declared halts are derived from the BACKCAST_2020 coverage")
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

    shared = [span(r) for r in document["exchange_wide_spot_gaps"]["ranges"]]

    def inside(moment: datetime) -> bool:
        return any(lo <= moment < hi for lo, hi in shared)

    remaining: dict[str, list[dict[str, object]]] = {}
    for name, market, entry in legs:
        left: list[int] = []
        for item in entry.get("missing_ranges", []):
            lo, hi = span(item)
            moment = lo
            while moment < hi:
                if market == StoreMarket.USDS_M.value or not inside(moment):
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
        "gaps_that_would_remain_and_refuse_the_run": remaining,
    }


def _coverage_read_errors(entries: dict[str, Any]) -> list[str]:
    return sorted(
        f"{n}:{m}"
        for n, markets in entries.items()
        for m, e in markets.items()
        if _read_errors(e)
    )


def _coverage_main(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    state: GitState,
    rows_factory: RowSourceFactory | None,
    moment: datetime,
    gh: g4.GhRunner,
) -> int:
    try:
        if args.stage != protocol.STAGE_FORWARD and args.through is not None:
            raise ValueError("--through is for FORWARD only")
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        window = data.frozen_window(args.stage)
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
        plan = build_plan(window, limit, moment, policy)
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
                halt_intervals=protocol.stage_halt_intervals(),
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
            "source": "vision_spot_and_usdm_1h_and_funding",
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
            "not_computed": "no signal, replay, return or label; rows were read "
            "only to validate and count them",
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
                "exchange_wide_spot_gap_hours": document["exchange_wide_spot_gaps"][
                    "hours"
                ],
                "legs_with_read_errors": errored,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return EXIT_HALTED if errored else EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: GitState = tierd_driver.git_state,
    rows_factory: RowSourceFactory | None = None,
    now: Clock = lambda: datetime.now(UTC),
    gh: g4.GhRunner = g4.default_gh,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
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
        )
    return _run_main(
        args,
        out=out,
        err=err,
        state=state,
        rows_factory=rows_factory,
        moment=moment,
        gh=gh,
    )


def _run_main(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    state: GitState,
    rows_factory: RowSourceFactory | None,
    moment: datetime,
    gh: g4.GhRunner,
) -> int:
    try:
        mode = _check_mode(args, moment)
        core7_protocol.validate_authorization_url(args.authorization_url)
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        protocol.verify_fixture_binding()
        window = data.frozen_window(args.stage)
        limit = _stage_limit(args, window, moment, mode)
        dev = None if args.dev_result is None else _load_dev_result(args.dev_result)
        backcast = (
            None
            if args.backcast_decision is None
            else _load_backcast_decision(args.backcast_decision)
        )
        # authenticate the owner comment before any file, row or data is read
        digest = protocol.protocol_digest()
        first_fetch = g4.fetch_comment(args.authorization_url, gh)
        g4.require_protocol_digest(first_fetch, digest)
        g4.require_protocol_id(first_fetch, protocol.PROTOCOL_ID)
        data_auth: g4.CommentSnapshot | None = None
        if _auth_url(args.stage) is not None:
            data_auth = g4.fetch_comment(protocol.DATA_AUTHORIZATION_URL, gh)
        commit, clean = state()
        if not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        spot_payload, spot_sha = tierd_driver._read_payload(args.exchange_info_spot)
        perp_payload, perp_sha = tierd_driver._read_payload(args.exchange_info_perp)
        rules = tierd_driver.load_rules(spot_payload, perp_payload)
        policy = embargo_for(limit, stage=args.stage)
        plan = build_plan(window, limit, moment, policy)
        if rows_factory is not None:
            detect = reader = rows_factory(plan)
        else:
            detect, reader = _sources(args, plan, policy)
        try:
            stage, coverage = load_stage_data(
                detect,
                reader,
                window=window,
                limit_close=limit,
                rules=rules,
                data_authorization_url=_auth_url(args.stage),
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
            "port_source_digest": protocol.PORT_SOURCE_DIGEST,
            "parity_fixture_sha256": protocol.FIXTURE_SHA256,
            "harness_source_digest": protocol.harness_source_digest(),
            "stage": args.stage,
            "mode": mode,
            "window": window.to_dict(),
            "read_limit_close": limit.isoformat(),
            "dataset_id": dataset.dataset_id,
            "activation_rows": list(stage.activation_rows),
            "source": "vision_spot_and_usdm_1h_and_funding",
            "taker_buy": "sum of the four hourly taker_buy_quote_volume values per 4h "
            "bar, carried as a (T, legs) array beside the dataset; bound into the "
            "dataset identity by its SHA-256",
            "exchange_info_sha256": {"spot": spot_sha, "perp": perp_sha},
            "exchange_rules_note": RULES_NOTE,
            "listing_coverage": coverage,
            "archive_evidence_count": len(evidence),
            "archive_evidence_digest": (
                tierd_driver._digest(evidence) if evidence else None
            ),
            "daily_fallback": {
                "enabled": bool(args.daily_fallback),
                "filled_hours": len(fills),
                "fills_digest": tierd_driver._digest(fills) if fills else None,
                "fills": fills,
                "forward_fill": False,
                "rest_fallback": False,
                "unusable_daily_archives": unusable,
            },
            "declared_halts": {
                "intervals": [list(pair) for pair in protocol.DECLARED_HALTS],
                "coverage_json_sha256": protocol.COVERAGE_EVIDENCE_SHA256,
            },
            "cash_account": {
                "initial_margin_rate": protocol.INITIAL_MARGIN_RATE,
                "limit": protocol.CASH_LIMIT,
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
            "backcast_decision": backcast,
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
            backcast_verdict=None if backcast is None else backcast["stage_verdict"],
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
        if error.evidence.get("gap_runs"):
            err.write(json.dumps(error.evidence["gap_runs"]) + "\n")
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
        gap_runs = (getattr(error, "evidence", None) or {}).get("gap_runs")
        if gap_runs:
            err.write(json.dumps(gap_runs) + "\n")
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
        "stage_verdict": result.get("stage_verdict"),
        "status": result.get("status"),
    }
    (args.output_dir / RUN_RECEIPT_NAME).write_bytes(canonical_json_bytes(run_receipt))
    out.write(
        json.dumps(
            {
                "stage": args.stage,
                "mode": mode,
                "label": label,
                "stage_verdict": result.get("stage_verdict"),
                "status": result.get("status"),
                "failed_conditions": result.get("failed_conditions"),
                "dataset_id": stage.dataset.dataset_id,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    invalid = label == protocol.LABEL_INVALID or (
        result.get("stage_verdict") == decision.VERDICT_INVALID
    )
    if invalid:
        err.write("not successful: a cell is invalid (INVALID)\n")
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
