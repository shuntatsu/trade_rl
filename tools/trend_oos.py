"""Driver for the out-of-sample test ``trend_long_spot_oos_v1``.

``python -m tools.trend_oos plan --stage BACKCAST|FORWARD``
``python -m tools.trend_oos coverage --stage BACKCAST|FORWARD``
``python -m tools.trend_oos run --stage BACKCAST``
``python -m tools.trend_oos run --stage FORWARD --interim | --final``

``plan`` prints the frozen protocol, its digest and the stage windows. It loads no data
and computes no return.

``coverage`` is the result-blind data step before any run: it acquires the stage's spot
archives through the same source path and writes ``coverage.json`` (listing start, missing
hour ranges after the daily fallback, fallback fills, placeholder rows, checksum results,
exchange-wide gaps). It builds no signal, replay or return and never refuses on a gap.

``run`` builds the stage dataset (spot only, 22-leg layout, unlisted markets
unavailable; :mod:`trade_rl.evaluation.trend_oos.data`) from checksum-verified Binance
Vision archives (offline unless ``--allow-network``; ``--daily-fallback`` fills hours
missing from the primary source from checksum-verified daily archives with provenance),
authenticates the owner authorization comment (``--authorization-url``) with
``tools.core7_g4_comment`` (owner author, this protocol's id and digest quoted, unedited
between two fetches), requires a clean checkout and a new output directory, records the
code commit, the dataset id and a data receipt, and writes every cell and the result
before it prints.

* BACKCAST runs once and decides (``OOS_SUPPORTED`` / ``OOS_NOT_SUPPORTED`` /
  ``INVALID``).
* FORWARD needs ``--backcast-decision`` (the stop rule: it is refused unless BACKCAST was
  ``OOS_SUPPORTED``). ``--final`` is refused before 2027-05-02T00:00Z; ``--interim``
  runs at any time up to the latest fully closed bar, writes a dated
  ``INTERIM_NOT_A_DECISION`` report and never a label.
* The default embargo of ``tools.core7_mtf_tierd`` (2026-08-01) is untouched. FORWARD
  reads past it only through :class:`ForwardEmbargoPolicy`, bound to the exact owner
  authorization URL of Issue #810 and to the stage's read limit, which is verified by
  fetching that comment as the repository owner.

The driver authorizes nothing beyond data reads and the evaluation: no orders, no API
keys, no FS1 scoring.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from tools import core7_g4_comment as g4
from tools import core7_mtf_tierd as tierd_driver
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.carry_multisymbol.protocol import SPOT_HALTS
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.trend_oos import data, decision, protocol, runner
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryScreenHalt,
    parse_carry_spot_rows,
)
from trade_rl.integrations.market_store.carry_source import (
    ArchiveDirFetcher,
    CarryRowSource,
    VisionArchiveSource,
    _check_order,
    _gap_error,
    _missing_hours,
    _vision_rows,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.fetch import VisionFetcher
from trade_rl.integrations.market_store.parse import (
    ArchiveContentError,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion
from trade_rl.integrations.market_store.rules import ExchangeInfoError

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3
EXIT_INVALID = 4
EXIT_UNAUTHENTICATED = 5
COVERAGE_NAME = "coverage.json"
COVERAGE_SCHEMA = "trend_oos_coverage_v1"
DATA_RECEIPT_NAME = "trend-oos-data-receipt.json"
RUN_RECEIPT_NAME = "trend-oos-run-receipt.json"
DATA_RECEIPT_SCHEMA = "trend_oos_data_receipt_v1"
RUN_RECEIPT_SCHEMA = "trend_oos_run_receipt_v1"
RULES_NOTE = (
    "exchange rules are one static retrieval-time snapshot applied to every row of the "
    "stage; they are not point-in-time (for BACKCAST this approximates 2018-2020)"
)
_MAX_REPORTED_GAP_RUNS = 25

GitState = Callable[[], tuple[str, bool]]
RowSourceFactory = Callable[[IngestionPlan], CarryRowSource]
Clock = Callable[[], datetime]


# ----------------------------------------------------------------- the FORWARD embargo


@dataclass(frozen=True, slots=True)
class ForwardEmbargoPolicy(EmbargoPolicy):
    """Market-store embargo for the FORWARD stage only, bound to one authorization.

    It is a *separate* policy: ``EmbargoPolicy.enforced()``, the module constants of the
    market store and the Tier D / core7 embargo are untouched. It exists only for the
    exact owner authorization URL of Issue #810 (``protocol.DATA_AUTHORIZATION_URL``)
    and admits archives and rows strictly before ``admit_before`` (the UTC midnight at
    or after the stage's read limit, never in the future), so no archive, row or request
    reaches past the stage's last row or the latest fully closed day.
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


class _StageVisionFetcher(VisionFetcher):
    """``VisionFetcher`` whose URL check is the FORWARD policy instead of the default."""

    def __init__(self, policy: EmbargoPolicy, **kwargs: Any) -> None:
        super().__init__(
            warmup_authorization_url=policy.warmup_authorization_url, **kwargs
        )
        self._policy = policy


def _ceil_day(value: datetime) -> datetime:
    floor = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return floor if floor == value else floor + timedelta(days=1)


def _floor_day(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def embargo_for(limit_close: datetime, *, stage: str) -> EmbargoPolicy:
    """The market-store policy of a stage read: the enforced default before 2026-08-01."""
    admit_before = _ceil_day(limit_close)
    if admit_before <= data.DEFAULT_EMBARGO:
        return EmbargoPolicy.enforced()
    if stage != protocol.STAGE_FORWARD:
        raise data.StageEmbargoError("only FORWARD may read past the default embargo")
    return ForwardEmbargoPolicy(
        warmup_authorization_url=protocol.DATA_AUTHORIZATION_URL,
        admit_before=admit_before,
    )


# ----------------------------------------------------------------------- loading


def build_plan(
    window: data.StageWindow,
    limit_close: datetime,
    now: datetime,
    policy: EmbargoPolicy,
) -> IngestionPlan:
    """Hourly spot klines of the roster for ``[load_start, ceil_day(limit))``."""
    universe = Universe(
        symbols=protocol.ROSTER,
        interval=data.BASE_INTERVAL,
        markets=(StoreMarket.SPOT,),
    )
    return plan_ingestion(
        universe,
        start=window.load_start,
        end=_ceil_day(limit_close),
        now=now,
        embargo=policy,
    )


def _month_chunks(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    cursor = datetime.fromtimestamp(start_ms / 1_000, UTC)
    while ms_from_utc(cursor) < end_ms:
        following = (
            cursor.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            + timedelta(days=32)
        ).replace(day=1)
        chunks.append((ms_from_utc(cursor), min(ms_from_utc(following), end_ms)))
        cursor = following
    return chunks


def load_spot_series(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
) -> tuple[RawMarketSeries | None, dict[str, object]]:
    """The complete hourly spot series of one symbol from its first listed 4h bar.

    ``detect`` finds the first listed hour: only a *leading* run of unpublished monthly
    archives (HTTP 404) counts as "not listed yet"; an unpublished archive after the
    first row, an unexplained gap or any other failure raises. ``reader`` then serves
    the complete range from the first 4h-aligned listed hour (it may fill hours from
    checksum-verified daily archives). The 4h bar holding the first listed hour is
    unavailable. Nothing is filled forward or invented; a symbol with no spot rows in
    the window returns ``None``.
    """
    if halt_intervals is None:
        halt_intervals = protocol.stage_halt_intervals()
    start_ms = ms_from_utc(window.load_start)
    end_ms = ms_from_utc(limit_close)
    first_open: int | None = None
    unpublished_months = 0
    leading: list[tuple[int, int]] = []
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(
                StoreMarket.SPOT, symbol, data.BASE_INTERVAL, lo, hi
            )
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
            # the listing month's monthly archive is unusable: the reader rebuilds it
            rows = reader.read_klines(
                StoreMarket.SPOT, symbol, data.BASE_INTERVAL, lo, hi
            )
        if rows:
            first_open = int(rows[0].open_time_ms)
            break
        if first_open is None:
            unpublished_months += 1
    # a leading month whose monthly archive is unpublished may not hide a listing:
    # if its last day is published as a daily archive the run refuses instead of
    # silently moving the listing to a later month
    for lo, hi in leading:
        day = hi - 24 * data.HOUR_MS
        if reader.probe_published_day(symbol, day):  # type: ignore[attr-defined]
            raise data.StageDataError(
                f"{symbol}: the monthly archive of the leading month starting "
                f"{_iso(lo)} is unpublished but a daily archive exists ({_iso(day)}); "
                "the listing would silently move"
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
    coverage["first_listed_open"] = datetime.fromtimestamp(
        first_open / 1_000, UTC
    ).isoformat()
    coverage["dropped_leading_partial_hours"] = (aligned - first_open) // data.HOUR_MS
    if aligned >= end_ms:
        return None, coverage
    coverage["first_bar_open"] = datetime.fromtimestamp(
        aligned / 1_000, UTC
    ).isoformat()
    rows = reader.read_klines(
        StoreMarket.SPOT, symbol, data.BASE_INTERVAL, aligned, end_ms
    )
    opens = [int(row.open_time_ms) for row in rows]
    _check_order(opens, market=StoreMarket.SPOT, symbol=symbol, what="klines")
    for h_lo, h_hi in protocol.declared_halt_intervals():
        first = bisect.bisect_left(opens, ms_from_utc(h_lo))
        if first < len(opens) and opens[first] + data.HOUR_MS <= ms_from_utc(h_hi):
            raise data.StageDataError(
                f"declared halt {h_lo.isoformat()}..{h_hi.isoformat()} is not "
                f"exchange-wide: {symbol} has a published row at {_iso(opens[first])}"
            )
    halts = [(ms_from_utc(lo), ms_from_utc(hi)) for lo, hi in halt_intervals]
    gaps = _missing_hours(opens, aligned, end_ms, explained=halts)
    if gaps:
        gap = _gap_error(StoreMarket.SPOT, symbol, gaps)
        evidence = dict(gap.evidence or {})
        evidence["gap_runs"] = tierd_driver._ranges(gaps)[:_MAX_REPORTED_GAP_RUNS]
        gap.evidence = evidence
        raise gap
    start_time = datetime.fromtimestamp(aligned / 1_000, UTC)
    try:
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
    coverage["rows"] = len(rows)
    return series, coverage


def load_stage_dataset(
    detect: CarryRowSource,
    reader: CarryRowSource,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    rules: dict[str, Any],
    halt_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    data_authorization_url: str | None = None,
) -> tuple[MarketDataset, dict[str, dict[str, object]]]:
    """Read every roster symbol through the stage guard and assemble the dataset."""
    if halt_intervals is None:
        halt_intervals = protocol.stage_halt_intervals()
    guarded_detect = data.StageRowSource(detect, window=window, limit_close=limit_close)
    guarded_reader = data.StageRowSource(reader, window=window, limit_close=limit_close)
    spot: dict[str, RawMarketSeries | None] = {}
    coverage: dict[str, dict[str, object]] = {}
    for symbol in protocol.ROSTER:
        spot[symbol], coverage[symbol] = load_spot_series(
            guarded_detect,
            guarded_reader,
            symbol,
            window=window,
            limit_close=limit_close,
            halt_intervals=halt_intervals,
        )
    dataset = data.assemble_stage_dataset(
        window,
        spot,
        rules=rules,
        limit_close=limit_close,
        halt_intervals=halt_intervals,
        data_authorization_url=data_authorization_url,
    )
    return dataset, coverage


# ----------------------------------------------------------------- CLI plumbing


def _parse_instant(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an ISO-8601 UTC instant, got {text!r}"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def plan_document(stage: str, now: datetime) -> dict[str, Any]:
    """Protocol, digest and windows; reads no data."""
    window = data.frozen_window(stage)
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "lowturn_protocol_id": protocol.LOWTURN_PROTOCOL_ID,
        "lowturn_protocol_digest": protocol.LOWTURN_PROTOCOL_DIGEST,
        "stage": stage,
        "window": window.to_dict(),
        "windows": {
            name: data.frozen_window(name).to_dict() for name in protocol.STAGES
        },
        "final_not_before": protocol.FINAL_NOT_BEFORE,
        "latest_closed_bar": data.latest_closed_bar(now).isoformat(),
        "forward_data_authorization_url": protocol.DATA_AUTHORIZATION_URL,
        "default_embargo": data.DEFAULT_EMBARGO.isoformat(),
        "protocol": protocol.protocol_payload(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.trend_oos",
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
    cov.add_argument(
        "--through",
        type=_parse_instant,
        default=None,
        help="FORWARD only: an earlier 4h-grid row close to stop at",
    )
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
    mode.add_argument(
        "--final", action="store_true", help="FORWARD final scoring (2027-05-02T00:00Z)"
    )
    mode.add_argument(
        "--interim", action="store_true", help="FORWARD interim report, no label"
    )
    run.add_argument(
        "--through",
        type=_parse_instant,
        default=None,
        help="FORWARD interim only: an earlier 4h-grid row close to stop at",
    )
    run.add_argument(
        "--backcast-decision",
        type=Path,
        default=None,
        help="FORWARD: the BACKCAST decision.json (the stop rule needs OOS_SUPPORTED)",
    )
    run.add_argument("--archive-dir", type=Path, required=True)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument(
        "--daily-fallback",
        action="store_true",
        help="fill hours missing from the primary source from checksum-verified "
        "Vision daily archives; provenance is recorded",
    )
    run.add_argument("--exchange-info-spot", type=Path, required=True)
    run.add_argument("--exchange-info-perp", type=Path, required=True)
    run.add_argument("--authorization-url", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def _vision_factory(
    archive_dir: Path, allow_network: bool, policy: EmbargoPolicy
) -> RowSourceFactory:
    def factory(plan: IngestionPlan) -> CarryRowSource:
        inner = None
        if allow_network:
            inner = _StageVisionFetcher(policy, max_attempts=1)
        return VisionArchiveSource(ArchiveDirFetcher(archive_dir, inner), plan)

    return factory


def _monthly_unusable(error: Exception) -> bool:
    """A monthly archive that is unpublished (404) or fails the strict parse."""
    kind = (getattr(error, "evidence", None) or {}).get("kind")
    return kind in ("archive_not_published", "malformed_archive")


class _MonthFallbackSource(tierd_driver.DailyFallbackSource):
    """Daily fallback that also rebuilds an unusable monthly archive from dailies.

    A month whose monthly archive is unpublished or malformed (never a checksum
    fault) is rebuilt from checksum-verified daily archives with the same strict parse;
    every filled hour records its day archive and SHA-256 (source
    ``vision_daily_unusable_month``). A day that is unpublished or malformed stays
    missing and is recorded in ``unusable_days()``. Nothing is filled forward.
    """

    def probe_published_day(self, symbol: str, day_start_ms: int) -> bool:
        """Whether the daily archive of that day publishes its first hour (nothing recorded)."""
        got = self._fill_hours(
            StoreMarket.SPOT,
            symbol,
            data.BASE_INTERVAL,
            [day_start_ms],
            source="probe",
            tolerate_malformed=True,
            record=False,
        )
        return bool(got)

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        rows: list[Any] = []
        for lo, hi in _month_chunks(start_ms, end_ms):
            try:
                rows.extend(super().read_klines(market, symbol, interval, lo, hi))
            except (CarryCoverageError, CarryIntegrityHalt) as error:
                if not _monthly_unusable(error):
                    raise
                missing = [
                    hour
                    for hour in range(lo, hi, data.HOUR_MS)
                    if not self._explained(market, lo, hour)
                ]
                rows.extend(
                    self._fill_hours(
                        market,
                        symbol,
                        interval,
                        missing,
                        source="vision_daily_unusable_month",
                        tolerate_malformed=True,
                    )
                )
        return sorted(rows, key=lambda row: int(row.open_time_ms))


def _sources(
    args: argparse.Namespace, plan: IngestionPlan, policy: EmbargoPolicy
) -> tuple[CarryRowSource, CarryRowSource]:
    """``(detect, reader)``: the primary source and, with the fallback, its wrapper."""
    base = _vision_factory(args.archive_dir, args.allow_network, policy)
    primary = base(plan)
    if not args.daily_fallback:
        return primary, primary
    daily = _vision_factory(args.archive_dir, args.allow_network, policy)
    return primary, _MonthFallbackSource(
        primary, daily, halts=protocol.stage_halt_intervals(), embargo=policy
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_backcast_decision(path: Path) -> dict[str, Any]:
    """The BACKCAST decision the stop rule needs; anything else refuses FORWARD."""
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise ValueError("the BACKCAST decision is not JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("protocol_id") != protocol.PROTOCOL_ID
        or document.get("protocol_digest") != protocol.protocol_digest()
        or document.get("stage") != protocol.STAGE_BACKCAST
        or document.get("mode") != runner.MODE_DECISION
    ):
        raise ValueError(
            "the file is not a BACKCAST decision of this exact protocol digest"
        )
    if document.get("label") != protocol.LABEL_SUPPORTED:
        raise ValueError(
            f"BACKCAST was {document.get('label')!r}: the TS_TREND_LONG_SPOT line "
            "ended and no FORWARD scoring is done"
        )
    return {"label": document["label"], "sha256": _sha256(raw)}


def _check_mode(args: argparse.Namespace, now: datetime) -> str:
    """The run mode; refuses every stage/mode combination the protocol forbids."""
    if args.stage == protocol.STAGE_BACKCAST:
        if args.final or args.interim or args.through is not None:
            raise ValueError("BACKCAST takes no --final/--interim/--through")
        if args.backcast_decision is not None:
            raise ValueError("--backcast-decision belongs to FORWARD")
        return runner.MODE_DECISION
    if args.final == args.interim:
        raise ValueError("FORWARD needs exactly one of --final and --interim")
    if args.backcast_decision is None:
        raise ValueError("FORWARD needs --backcast-decision (the stop rule)")
    if args.final:
        if args.through is not None:
            raise ValueError("--through is for --interim only")
        gate = datetime.fromisoformat(protocol.FINAL_NOT_BEFORE.replace("Z", "+00:00"))
        if now < gate:
            raise ValueError(
                f"--final is refused before {protocol.FINAL_NOT_BEFORE} (now "
                f"{now.isoformat()})"
            )
        return runner.MODE_DECISION
    return runner.MODE_INTERIM


def _stage_limit(
    args: argparse.Namespace, window: data.StageWindow, now: datetime, mode: str
) -> datetime:
    limit = data.read_limit(
        window,
        now=now,
        authorization_url=(
            protocol.DATA_AUTHORIZATION_URL
            if args.stage == protocol.STAGE_FORWARD
            else None
        ),
        through=args.through,
    )
    if mode == runner.MODE_INTERIM and args.through is None:
        # Vision publishes a day archive only after the day closed
        limit = min(limit, _floor_day(now))
    if limit <= window.first_decision + 2 * data.BAR:
        raise data.StageEmbargoError("no closed day beyond the first decision")
    if _ceil_day(limit) > _floor_day(now):
        raise data.StageEmbargoError(
            "the day archive holding the last read row is not closed yet"
        )
    return limit


# --------------------------------------------------------------------- coverage

PlaceholderLog = list[tuple[str, int, int, str]]


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
            _StageVisionFetcher(policy, max_attempts=1) if args.allow_network else None
        )
        return _CoverageVisionSource(
            ArchiveDirFetcher(args.archive_dir, inner), inner_plan, log
        )

    primary = factory(plan)
    if not args.daily_fallback:
        return primary, primary, log
    fallback = _MonthFallbackSource(primary, factory, halts=SPOT_HALTS, embargo=policy)
    return primary, fallback, log


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, UTC).isoformat()


def _hour_ranges(hours: Sequence[int]) -> list[dict[str, object]]:
    return tierd_driver._ranges(sorted(hours))


def _halt_error(error: CarryScreenHalt) -> dict[str, object]:
    return {
        "kind": error.kind,
        "detail": error.evidence.get("kind"),
        "resumable": error.resumable,
        "message": str(error),
    }


def _symbol_coverage(
    detect: CarryRowSource,
    reader: CarryRowSource,
    symbol: str,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
) -> tuple[dict[str, Any], int | None, set[int]]:
    """Coverage facts of one symbol; never raises on a gap, a 404 or a bad archive."""
    start_ms, end_ms = ms_from_utc(window.load_start), ms_from_utc(limit_close)
    errors: list[dict[str, object]] = []
    first_open: int | None = None
    unpublished_leading = 0
    leading_with_dailies: list[str] = []
    for lo, hi in _month_chunks(start_ms, end_ms):
        try:
            rows = detect.read_klines(
                StoreMarket.SPOT, symbol, data.BASE_INTERVAL, lo, hi
            )
        except CarryCoverageError as error:
            if (error.evidence or {}).get("kind") == "archive_not_published":
                unpublished_leading += 1
                probe = getattr(reader, "probe_published_day", None)
                if callable(probe) and probe(symbol, hi - 24 * data.HOUR_MS):
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
                        StoreMarket.SPOT, symbol, data.BASE_INTERVAL, lo, hi
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
            rows = reader.read_klines(
                StoreMarket.SPOT, symbol, data.BASE_INTERVAL, lo, hi
            )
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
    halts = [(ms_from_utc(lo), ms_from_utc(hi)) for lo, hi in halt_intervals]
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
    return entry, first_open, set(missing)


def _exchange_wide(
    firsts: dict[str, int], missing: dict[str, set[int]]
) -> dict[str, object]:
    """Hours missing for every symbol listed at that hour (at least two listed)."""
    hours: list[int] = []
    union: set[int] = set().union(*missing.values()) if missing else set()
    for hour in sorted(union):
        listed = [name for name, first in firsts.items() if first <= hour]
        if len(listed) >= 2 and all(hour in missing[name] for name in listed):
            hours.append(hour)
    return {
        "rule": "unexplained missing hours shared by every symbol listed then "
        "(at least two listed)",
        "hours": len(hours),
        "ranges": _hour_ranges(hours),
    }


_INTEGRITY_DETAILS = ("checksum_mismatch", "checksum_missing", "malformed_archive")


def coverage_report(
    detect: CarryRowSource,
    reader: CarryRowSource,
    placeholders: PlaceholderLog,
    *,
    window: data.StageWindow,
    limit_close: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
) -> dict[str, Any]:
    """Per-symbol listing, gaps, fills, placeholders and checksum results.

    Rows are read through the stage guard only to validate and count them; this builds
    no dataset, no signal, no replay and no return, and never refuses because of a gap.
    """
    guarded_detect = data.StageRowSource(detect, window=window, limit_close=limit_close)
    guarded_reader = data.StageRowSource(reader, window=window, limit_close=limit_close)
    symbols: dict[str, Any] = {}
    firsts: dict[str, int] = {}
    missing: dict[str, set[int]] = {}
    for symbol in protocol.ROSTER:
        entry, first, gone = _symbol_coverage(
            guarded_detect,
            guarded_reader,
            symbol,
            window=window,
            limit_close=limit_close,
            halt_intervals=halt_intervals,
        )
        symbols[symbol] = entry
        if first is not None:
            firsts[symbol] = first
            missing[symbol] = gone
    evidence_fn = getattr(reader, "evidence", None)
    evidence = list(evidence_fn()) if callable(evidence_fn) else []
    fills_fn = getattr(reader, "fills", None)
    fills = list(fills_fn()) if callable(fills_fn) else []
    unusable_fn = getattr(reader, "unusable_days", None)
    unusable = list(unusable_fn()) if callable(unusable_fn) else []
    for symbol, entry in symbols.items():
        entry["unusable_daily_archives"] = [
            u for u in unusable if u.get("symbol") == symbol
        ]
        mine = [e for e in evidence if e.get("symbol") == symbol]
        failures = [
            c
            for c in entry.get("unread_chunks", [])
            if c.get("detail") in _INTEGRITY_DETAILS
        ] + [e for e in entry["errors"] if e.get("detail") in _INTEGRITY_DETAILS]
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
            for sym, open_ms, close_ms, url in sorted(set(placeholders))
            if sym == symbol
        ]
    return {
        "symbols": symbols,
        "exchange_wide_gaps": _exchange_wide(firsts, missing),
        "archive_evidence_count": len(evidence),
        "archive_evidence_digest": tierd_driver._digest(evidence) if evidence else None,
        "daily_fallback_filled_hours": len(fills),
        "declared_halts": [
            [_iso(ms_from_utc(a)), _iso(ms_from_utc(b))] for a, b in SPOT_HALTS
        ],
    }


def halts_from_coverage(path: Path) -> dict[str, Any]:
    """The exchange-wide ranges of a coverage.json as a declared halt list.

    Only the report's exchange-wide ranges qualify. The result also lists, per symbol,
    the missing ranges that would remain (and make a run refuse). It refuses a report
    with read errors, and it computes nothing from prices.
    """
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != COVERAGE_SCHEMA:
        raise ValueError("not a trend_oos coverage report")
    if document.get("stage") != protocol.STAGE_BACKCAST:
        raise ValueError("declared halts are derived from the BACKCAST coverage")
    bad = sorted(n for n, e in document["symbols"].items() if _read_errors(e))
    if bad:
        raise ValueError(f"the coverage report has read errors for {bad}")
    hidden = sorted(
        n
        for n, e in document["symbols"].items()
        if e.get("leading_unpublished_months_with_daily_archives")
    )
    if hidden:
        raise ValueError(
            f"leading months with daily archives (listing would move): {hidden}"
        )

    def span(item: dict[str, Any]) -> tuple[datetime, datetime]:
        lo = datetime.fromisoformat(item["first_missing_open"])
        return lo, lo + timedelta(hours=int(item["hours"]))

    shared = [span(r) for r in document["exchange_wide_gaps"]["ranges"]]

    def inside(moment: datetime) -> bool:
        return any(lo <= moment < hi for lo, hi in shared)

    remaining: dict[str, list[dict[str, object]]] = {}
    for name, entry in document["symbols"].items():
        left: list[int] = []
        for item in entry.get("missing_ranges", []):
            lo, hi = span(item)
            moment = lo
            while moment < hi:
                if not inside(moment):
                    left.append(int(moment.timestamp() * 1_000))
                moment += timedelta(hours=1)
        if left:
            remaining[name] = _hour_ranges(left)
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
            n: e["unread_chunks"]
            for n, e in document["symbols"].items()
            if e.get("unread_chunks")
        },
        "gaps_that_would_remain_and_refuse_the_run": remaining,
    }


def _read_errors(entry: dict[str, Any]) -> bool:
    """True when something other than a documented absence went wrong for a symbol."""
    return bool(entry["errors"]) or any(
        c.get("kind") in ("operational", "integrity")
        for c in entry.get("unread_chunks", [])
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
        if args.stage == protocol.STAGE_BACKCAST and args.through is not None:
            raise ValueError("--through is for FORWARD only")
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        window = data.frozen_window(args.stage)
        mode = (
            runner.MODE_DECISION
            if args.stage == protocol.STAGE_BACKCAST
            else runner.MODE_INTERIM
        )
        limit = _stage_limit(args, window, moment, mode)
        data_auth: g4.CommentSnapshot | None = None
        if args.stage == protocol.STAGE_FORWARD:
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
                detect, reader, log, window=window, limit_close=limit
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
            "source": "vision_spot_1h",
            "daily_fallback_enabled": bool(args.daily_fallback),
            "allow_network": bool(args.allow_network),
            "forward_data_authorization": (
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
            "note": "a month whose archive is unpublished after listing is reported "
            "in unread_chunks and counted missing; the daily fallback fills hours "
            "inside published archives only",
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
    errored = sorted(n for n, e in entries.items() if _read_errors(e))
    out.write(
        json.dumps(
            {
                "stage": args.stage,
                "coverage": str(args.output_dir / COVERAGE_NAME),
                "symbols_with_missing_hours": sorted(
                    n for n, e in entries.items() if e.get("missing_hours")
                ),
                "exchange_wide_gap_hours": document["exchange_wide_gaps"]["hours"],
                "symbols_with_read_errors": errored,
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
        except ValueError as error:
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
    try:
        mode = _check_mode(args, moment)
        # the authorization shape is checked before any file or row is touched
        core7_protocol.validate_authorization_url(args.authorization_url)
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        window = data.frozen_window(args.stage)
        limit = _stage_limit(args, window, moment, mode)
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
        if args.stage == protocol.STAGE_FORWARD:
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
            dataset, coverage = load_stage_dataset(
                detect,
                reader,
                window=window,
                limit_close=limit,
                rules=rules,
                data_authorization_url=(
                    protocol.DATA_AUTHORIZATION_URL
                    if args.stage == protocol.STAGE_FORWARD
                    else None
                ),
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
        data.validate_stage_dataset(dataset, window)
        # re-fetch immediately before the run: an edited comment authorizes nothing
        auth_receipt = g4.confirm_unedited(
            args.authorization_url, first_fetch, gh, protocol_digest=digest
        )
        if data_auth is not None:
            # the bound data-authorization comment must not change between the fetches
            again = g4.fetch_comment(protocol.DATA_AUTHORIZATION_URL, gh)
            if (
                again.body_sha256 != data_auth.body_sha256
                or again.updated_at != data_auth.updated_at
            ):
                raise g4.G4AuthError(
                    "the data-authorization comment was edited after it was first fetched"
                )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        data_receipt: dict[str, object] = {
            "schema_version": DATA_RECEIPT_SCHEMA,
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_digest": digest,
            "lowturn_protocol_digest": protocol.LOWTURN_PROTOCOL_DIGEST,
            "stage": args.stage,
            "mode": mode,
            "window": window.to_dict(),
            "read_limit_close": limit.isoformat(),
            "dataset_id": dataset.dataset_id,
            "source": "vision_spot_1h",
            "perp_legs": "not_loaded_unavailable_in_every_row",
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
                "coverage_code_commit": protocol.COVERAGE_EVIDENCE_CODE_COMMIT,
                "coverage_protocol_digest": protocol.COVERAGE_EVIDENCE_PROTOCOL_DIGEST,
            },
            "spot_cash_account": {
                "gross_limit": protocol.SPOT_GROSS_LIMIT,
                "invalid_on": protocol.REASON_NEGATIVE_CASH,
            },
            "forward_data_authorization": (
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
            "backcast_decision": backcast,
            "code_commit": commit,
            "working_tree_clean": True,
            "run_at": moment.isoformat(),
        }
        (args.output_dir / DATA_RECEIPT_NAME).write_bytes(
            canonical_json_bytes(data_receipt)
        )
        result = runner.run_stage(
            dataset,
            window=window,
            output_dir=args.output_dir,
            expected_protocol_digest=digest,
            mode=mode,
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
    claim = (
        decision.combined_claim(backcast["label"], label)
        if backcast is not None and label is not None
        else None
    )
    run_receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": digest,
        "stage": args.stage,
        "mode": mode,
        "dataset_id": dataset.dataset_id,
        "authorization_receipt": auth_receipt.to_dict(),
        "g4_comment_updated_at": first_fetch.updated_at,
        "code_commit": commit,
        "label": label,
        "status": result.get("status"),
        "combined_claim": claim,
    }
    (args.output_dir / RUN_RECEIPT_NAME).write_bytes(canonical_json_bytes(run_receipt))
    out.write(
        json.dumps(
            {
                "stage": args.stage,
                "mode": mode,
                "label": label,
                "status": result.get("status"),
                "combined_claim": claim,
                "failed_conditions": result.get("failed_conditions"),
                "dataset_id": dataset.dataset_id,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if label == protocol.LABEL_INVALID:
        err.write("not successful: a cell is invalid (label INVALID)\n")
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
