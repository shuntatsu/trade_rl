"""Phase 1 Tier D driver for the Core7 MTF study: dry check and (separately authorized) run.

``python -m tools.core7_mtf_tierd dry-check | run``

The wiring lives here, outside ``trade_rl``, because the market-data store package may
be imported by nothing else under ``trade_rl`` (an architecture test pins that). The
pure assembly (embargo guard, 4h aggregation, 22-leg Dataset) is
``trade_rl.evaluation.core7_mtf_study.tierd``; the economics are ``run_study``.

``dry-check`` only loads the Dataset and prints its shape, time range, per-leg
availability counts and content hash. It never calls ``run_study`` and computes no
return, signal or decision. ``run`` additionally requires an owner G4
authorization comment URL (no default), a clean checkout and a *new* output directory,
and calls ``run_study`` exactly once. A passing dry check authorizes nothing.

Rows come from checksum-verified Binance Vision archives (``--source vision``, offline
unless ``--allow-network``) or from a populated market store (``--source store``,
connection URL only from ``TRADE_RL_MARKET_STORE_URL``). Exchange lot/min-notional rules
come from two raw ``exchangeInfo`` payload files whose SHA-256 is recorded; they are a
static snapshot, not point-in-time rules.

``--load-start`` defaults to 2021-10-01 (the warm-up allowed before the 2022-01-01 replay
start). Earlier starts run into unregistered 2021 spot maintenance gaps (2021-08-13,
2021-09-29) that no declared halt explains, so the strict loader refuses them; the
protocol registers only the 2023-03-24 spot halt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from tools import core7_g4_comment as g4
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.carry_multisymbol.protocol import SPOT_HALTS
from trade_rl.evaluation.core7_mtf_study import protocol, tierd
from trade_rl.evaluation.core7_mtf_study.runner import run_study
from trade_rl.integrations.binance import BinanceMarket
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryScreenHalt,
)
from trade_rl.integrations.binance.vision import vision_kline_url
from trade_rl.integrations.market_store.carry_source import (
    ArchiveDirFetcher,
    CarryRowSource,
    VisionArchiveSource,
    load_carry_pair,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion
from trade_rl.integrations.market_store.rules import (
    ExchangeInfoError,
    parse_symbol_rules,
)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3
EXIT_NOT_EVIDENCE = 4
EXIT_UNAUTHENTICATED = 5
RUN_RECEIPT_NAME = "tierd-run-receipt.json"
RECEIPT_NAME = "tierd-data-receipt.json"
#: Earlier starts hit unregistered 2021 spot maintenance gaps (2021-08-13, 2021-09-29).
DEFAULT_LOAD_START = datetime(2021, 10, 1, tzinfo=UTC)

GitState = Callable[[], tuple[str, bool]]
RowSourceFactory = Callable[[IngestionPlan], CarryRowSource]


def _parse_day(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD, got {text!r}"
        ) from error


def build_plan(load_start: datetime, now: datetime) -> IngestionPlan:
    """Hourly klines for both markets and perpetual funding, ``[load_start, 2026-08-01)``."""
    universe = Universe(
        symbols=protocol.ROSTER,
        interval=tierd.BASE_INTERVAL,
        markets=(StoreMarket.USDS_M, StoreMarket.SPOT),
    )
    return plan_ingestion(
        universe, start=load_start, end=tierd.DATA_END_EXCLUSIVE, now=now
    )


def load_rules(
    spot_payload: bytes, perp_payload: bytes
) -> dict[str, tuple[tierd.LegRules, tierd.LegRules]]:
    """Per-symbol ``(spot, perpetual)`` rules from raw ``exchangeInfo`` payloads."""
    spot = parse_symbol_rules(
        spot_payload, market=StoreMarket.SPOT, symbols=protocol.ROSTER
    )
    perp = parse_symbol_rules(
        perp_payload, market=StoreMarket.USDS_M, symbols=protocol.ROSTER
    )
    result: dict[str, tuple[tierd.LegRules, tierd.LegRules]] = {}
    for spot_rule, perp_rule in zip(spot, perp, strict=True):
        if spot_rule.symbol != perp_rule.symbol:
            raise ExchangeInfoError("exchange rules are not in roster order")
        result[spot_rule.symbol] = (
            tierd.LegRules(float(spot_rule.lot_size), float(spot_rule.min_notional)),
            tierd.LegRules(float(perp_rule.lot_size), float(perp_rule.min_notional)),
        )
    return result


def _day_record(
    market: StoreMarket,
    symbol: str,
    day_start_ms: int,
    url: str,
    reason: str,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "market": market.value,
        "symbol": symbol,
        "day": datetime.fromtimestamp(day_start_ms / 1_000, UTC).date().isoformat(),
        "archive_url": url,
        "reason": reason,
        "detail": detail,
    }


class DailyFallbackSource:
    """Fill hourly klines missing from the primary source from Vision *daily* archives.

    Only an hour that the primary source lacks and that no declared spot halt wholly
    contains is ever requested, and only from the checksum-verified daily archive of its
    day (the same ``VisionArchiveSource`` verification, cache and retry rules). A daily
    archive that is not published (HTTP 404) leaves its hours missing, so the strict
    pair loader still refuses them: there is no forward fill and no REST path. An hour
    present in the primary source is never replaced. Every filled hour is recorded in
    :meth:`fills` with its archive URL and SHA-256, in ascending order.
    """

    def __init__(
        self,
        primary: CarryRowSource,
        daily_factory: RowSourceFactory,
        *,
        halts: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
        embargo: EmbargoPolicy | None = None,
    ) -> None:
        self._primary = primary
        self._factory = daily_factory
        # default: the enforced FS1 embargo; only a caller with its own explicit,
        # authorized policy (``tools.trend_oos`` FORWARD) passes another one
        self._embargo = EmbargoPolicy.enforced() if embargo is None else embargo
        self._halts = [
            (int(lo.timestamp() * 1_000), int(hi.timestamp() * 1_000))
            for lo, hi in halts
        ]
        self._fills: dict[tuple[str, str, int], dict[str, object]] = {}
        self._unusable_days: list[dict[str, object]] = []
        self._daily_sources: list[Any] = []

    def _explained(self, market: StoreMarket, start_ms: int, open_ms: int) -> bool:
        return (
            market is StoreMarket.SPOT
            and open_ms != start_ms
            and any(
                lo <= open_ms and open_ms + tierd.HOUR_MS <= hi
                for lo, hi in self._halts
            )
        )

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        rows = list(
            self._primary.read_klines(market, symbol, interval, start_ms, end_ms)
        )
        present = {int(row.open_time_ms) for row in rows}
        missing = [
            stamp
            for stamp in range(start_ms, end_ms, tierd.HOUR_MS)
            if stamp not in present and not self._explained(market, start_ms, stamp)
        ]
        if not missing:
            return rows
        filled = self._fill_hours(market, symbol, interval, missing)
        return sorted([*rows, *filled], key=lambda row: int(row.open_time_ms))

    def _fill_hours(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        missing: list[int],
        *,
        source: str = "vision_daily",
        tolerate_malformed: bool = False,
        record: bool = True,
    ) -> list[Any]:
        """Hours of ``missing`` that checksum-verified daily archives publish.

        A day that is not published stays missing. With ``tolerate_malformed`` a day
        whose archive fails the strict parse (a checksum fault never qualifies) also
        stays missing and is recorded in :meth:`unusable_days`.
        """
        day_ms = 24 * tierd.HOUR_MS
        days = sorted({stamp - stamp % day_ms for stamp in missing})
        specs: list[ArchiveSpec] = []
        for day_start in days:
            day = datetime.fromtimestamp(day_start / 1_000, UTC)
            url = vision_kline_url(BinanceMarket(market.binance), symbol, interval, day)
            specs.append(
                ArchiveSpec(
                    kind=ArchiveKind.KLINES,
                    market=market,
                    symbol=symbol,
                    interval=interval,
                    period_kind=PeriodKind.DAILY,
                    period_start=day,
                    period_end=datetime.fromtimestamp(
                        (day_start + day_ms) / 1_000, UTC
                    ),
                    url=url,
                    checksum_url=f"{url}.CHECKSUM",
                )
            )
        plan = IngestionPlan(
            universe=Universe(symbols=(symbol,), interval=interval, markets=(market,)),
            start=specs[0].period_start,
            end=specs[-1].period_end,
            embargo=self._embargo,
            specs=tuple(specs),
            skipped=(),
        )
        daily: Any = self._factory(plan)
        self._daily_sources.append(daily)
        wanted = set(missing)
        filled: list[Any] = []
        for spec, day_start in zip(specs, days, strict=True):
            hours = [
                stamp for stamp in missing if day_start <= stamp < day_start + day_ms
            ]
            try:
                got = daily.read_klines(
                    market, symbol, interval, day_start, max(hours) + tierd.HOUR_MS
                )
            except CarryCoverageError:
                if tolerate_malformed and record:
                    self._unusable_days.append(
                        _day_record(
                            market, symbol, day_start, spec.url, "archive_not_published"
                        )
                    )
                continue  # not published: the hours stay missing and the load refuses
            except CarryIntegrityHalt as error:
                if (
                    not tolerate_malformed
                    or error.evidence.get("kind") != "malformed_archive"
                ):
                    raise
                if not record:
                    continue
                self._unusable_days.append(
                    _day_record(
                        market,
                        symbol,
                        day_start,
                        spec.url,
                        "malformed_archive",
                        str(error),
                    )
                )
                continue
            digest = next(
                (
                    item["archive_sha256"]
                    for item in daily.evidence()
                    if item["url"] == spec.url
                ),
                None,
            )
            for row in got:
                opened = int(row.open_time_ms)
                if opened in wanted:
                    filled.append(row)
                    if not record:
                        continue
                    self._fills[(market.value, symbol, opened)] = {
                        "market": market.value,
                        "symbol": symbol,
                        "open_time": datetime.fromtimestamp(
                            opened / 1_000, UTC
                        ).isoformat(),
                        "source": source,
                        "archive_url": spec.url,
                        "archive_sha256": digest,
                    }
        return filled

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        return list(self._primary.read_funding(market, symbol, start_ms, end_ms))

    def unusable_days(self) -> list[dict[str, object]]:
        """Days a tolerant fill could not use (unpublished or malformed daily archive)."""
        return list(self._unusable_days)

    def fills(self) -> list[dict[str, object]]:
        """Provenance of every filled hour, ascending by (market, symbol, hour)."""
        return [self._fills[key] for key in sorted(self._fills)]

    def evidence(self) -> list[dict[str, Any]]:
        primary = getattr(self._primary, "evidence", None)
        items: list[dict[str, Any]] = list(primary()) if callable(primary) else []
        for daily in self._daily_sources:
            items.extend(daily.evidence())
        return sorted(
            {item["url"]: item for item in items}.values(),
            key=lambda item: item["url"],
        )

    def close(self) -> None:
        close = getattr(self._primary, "close", None)
        if callable(close):
            close()


def load_tierd_dataset(
    source: CarryRowSource,
    *,
    rules: dict[str, tuple[tierd.LegRules, tierd.LegRules]],
    load_start: datetime,
    load_end: datetime = tierd.LOAD_END,
    halt_intervals: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
) -> MarketDataset:
    """Read every roster pair through the embargo guard and assemble the Dataset.

    Any missing symbol, unexplained gap or incomplete funding calendar raises; no
    symbol is excluded and nothing is filled.
    """
    start, end = tierd.validate_load_window(load_start, load_end)
    guarded = tierd.GuardedRowSource(source)
    pairs = {
        symbol: load_carry_pair(
            guarded,
            symbol,
            start_time=start,
            end_time=end,
            halt_intervals=halt_intervals,
        )
        for symbol in protocol.ROSTER
    }
    return tierd.assemble_tierd_dataset(
        pairs, rules=rules, spot_halt_intervals=halt_intervals
    )


def _ranges(open_times_ms: list[int]) -> list[dict[str, object]]:
    """Merge ascending hourly opens into ``{first_missing_open, hours}`` runs."""
    runs: list[tuple[int, int]] = []
    for stamp in open_times_ms:
        if runs and runs[-1][0] + runs[-1][1] * tierd.HOUR_MS == stamp:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((stamp, 1))
    return [
        {
            "first_missing_open": datetime.fromtimestamp(
                first / 1_000, UTC
            ).isoformat(),
            "hours": hours,
        }
        for first, hours in runs
    ]


def coverage_report(
    source: CarryRowSource,
    *,
    load_start: datetime,
    load_end: datetime = tierd.LOAD_END,
    halt_intervals: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
) -> dict[str, object]:
    """Per-leg hourly row coverage and funding counts; reads through the embargo guard.

    Builds no Dataset and no return. A missing hour is *unexplained* unless a declared
    spot halt wholly contains it. ``complete`` is true only if no leg has an
    unexplained gap and every perpetual has funding events.
    """
    start, end = tierd.validate_load_window(load_start, load_end)
    start_ms, end_ms = int(start.timestamp() * 1_000), int(end.timestamp() * 1_000)
    halts = [
        (int(lo.timestamp() * 1_000), int(hi.timestamp() * 1_000))
        for lo, hi in halt_intervals
    ]
    guarded = tierd.GuardedRowSource(source)
    legs: dict[str, dict[str, object]] = {}
    complete = True
    for symbol in protocol.ROSTER:
        for market, name in ((StoreMarket.SPOT, "spot"), (StoreMarket.USDS_M, "perp")):
            rows = guarded.read_klines(
                market, symbol, tierd.BASE_INTERVAL, start_ms, end_ms
            )
            present = {int(row.open_time_ms) for row in rows}
            expected = range(start_ms, end_ms, tierd.HOUR_MS)
            missing = [stamp for stamp in expected if stamp not in present]
            explained = [
                stamp
                for stamp in missing
                if market is StoreMarket.SPOT
                and stamp != start_ms
                and any(lo <= stamp and stamp + tierd.HOUR_MS <= hi for lo, hi in halts)
            ]
            unexplained = [stamp for stamp in missing if stamp not in set(explained)]
            entry: dict[str, object] = {
                "rows": len(rows),
                "expected_rows": len(expected),
                "missing_hours": len(missing),
                "explained_by_halt_hours": len(explained),
                "unexplained_missing_hours": len(unexplained),
                "unexplained_gaps": _ranges(unexplained),
            }
            if market is StoreMarket.USDS_M:
                funding = guarded.read_funding(market, symbol, start_ms, end_ms)
                stamps = [
                    int(row.funding_time_ms)
                    for row in funding
                    if int(row.funding_time_ms) < end_ms
                ]
                entry["funding_events"] = len(stamps)
                if not stamps:
                    complete = False
            if unexplained:
                complete = False
            legs[f"{symbol}:{name}"] = entry
    return {
        "load_start": start.isoformat(),
        "load_end": end.isoformat(),
        "complete": complete,
        "daily_fallback_filled_hours": (
            len(source.fills()) if hasattr(source, "fills") else 0
        ),
        "legs": legs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.core7_mtf_tierd",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def source_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--load-start",
            type=_parse_day,
            default=DEFAULT_LOAD_START,
            help="warm-up start (default 2021-10-01; replay starts 2022-01-01)",
        )
        sub.add_argument("--source", choices=("vision", "store"), default="vision")
        sub.add_argument("--archive-dir", type=Path, default=None)
        sub.add_argument("--allow-network", action="store_true")
        sub.add_argument(
            "--daily-fallback",
            action="store_true",
            help="fill hours missing from the primary source from checksum-verified "
            "Vision daily archives (needs --archive-dir); provenance is recorded",
        )

    def common(sub: argparse.ArgumentParser) -> None:
        source_args(sub)
        sub.add_argument("--exchange-info-spot", type=Path, required=True)
        sub.add_argument("--exchange-info-perp", type=Path, required=True)

    coverage = commands.add_parser(
        "coverage",
        help="read-only per-leg row/gap/funding report; builds no Dataset",
    )
    source_args(coverage)
    dry = commands.add_parser(
        "dry-check", help="load the Dataset only; print shape, range, counts, hash"
    )
    common(dry)
    run = commands.add_parser(
        "run", help="run_study on the loaded Dataset (needs owner G4 authorization)"
    )
    common(run)
    run.add_argument("--authorization-url", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--fit-stop-index",
        type=int,
        default=None,
        help="default: the replay start index (zero-residual RL arms fit nothing)",
    )
    return parser


def git_state() -> tuple[str, bool]:
    """The checked-out commit and whether the working tree is clean."""
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, not status.strip()


def _vision_rows(archive_dir: Path, allow_network: bool) -> RowSourceFactory:
    def factory(plan: IngestionPlan) -> CarryRowSource:
        inner = None
        if allow_network:
            from trade_rl.integrations.market_store.fetch import VisionFetcher

            inner = VisionFetcher(max_attempts=1)
        return VisionArchiveSource(ArchiveDirFetcher(archive_dir, inner), plan)

    return factory


def _store_rows(plan: IngestionPlan) -> CarryRowSource:
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    return PostgresMarketStore.open()


def _resolve_rows(args: argparse.Namespace) -> RowSourceFactory:
    if args.source == "vision":
        if args.archive_dir is None:
            raise ValueError("--source vision requires --archive-dir")
        base = _vision_rows(args.archive_dir, args.allow_network)
    else:
        base = _store_rows
    if not args.daily_fallback:
        return base
    if args.archive_dir is None:
        raise ValueError("--daily-fallback requires --archive-dir")
    daily = _vision_rows(args.archive_dir, args.allow_network)

    def with_fallback(plan: IngestionPlan) -> CarryRowSource:
        return DailyFallbackSource(base(plan), daily)

    return with_fallback


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_payload(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _receipt(
    *,
    dataset: MarketDataset,
    commit: str,
    clean: bool,
    args: argparse.Namespace,
    load_end: datetime,
    rules_sha: dict[str, str],
    evidence: list[dict[str, Any]],
    fills: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "core7_mtf_tierd_receipt_v1",
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "expected_protocol_digest": protocol.protocol_digest(),
        "tierd_contract": tierd.TIERD_CONTRACT,
        # decided by run_study; recorded in the run receipt written after the run
        "evidence_eligible": None,
        "dataset_id": dataset.dataset_id,
        "load_start": args.load_start.isoformat(),
        "load_end": load_end.isoformat(),
        "source": args.source,
        "exchange_info_sha256": rules_sha,
        "archive_evidence_count": len(evidence),
        "archive_evidence_digest": _digest(evidence) if evidence else None,
        "daily_fallback": {
            "enabled": bool(args.daily_fallback),
            "filled_hours": len(fills),
            "fills_digest": _digest(fills) if fills else None,
            "fills": fills,
            "forward_fill": False,
            "rest_fallback": False,
        },
        "code_commit": commit,
        "working_tree_clean": clean,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: GitState = git_state,
    rows_factory: RowSourceFactory | None = None,
    load_end: datetime = tierd.LOAD_END,
    gh: g4.GhRunner = g4.default_gh,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            # the authorization shape is checked before any file or row is touched
            protocol.validate_authorization_url(args.authorization_url)
            if args.output_dir.exists():
                raise FileExistsError(
                    f"{args.output_dir} exists; outputs are write-once"
                )
            # authenticate the owner comment before any file, row or data is read
            first_fetch = g4.fetch_comment(args.authorization_url, gh)
            g4.require_protocol_digest(first_fetch)
        tierd.validate_load_window(args.load_start, load_end)
        commit, clean = state()
        if args.command == "run" and not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        if args.command == "coverage":
            if rows_factory is None:
                rows_factory = _resolve_rows(args)
            source = rows_factory(build_plan(args.load_start, datetime.now(UTC)))
            try:
                report = coverage_report(
                    source, load_start=args.load_start, load_end=load_end
                )
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()
            out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return EXIT_OK if report["complete"] else EXIT_REFUSED
        spot_payload, spot_sha = _read_payload(args.exchange_info_spot)
        perp_payload, perp_sha = _read_payload(args.exchange_info_perp)
        rules = load_rules(spot_payload, perp_payload)
        if rows_factory is None:
            rows_factory = _resolve_rows(args)
        source = rows_factory(build_plan(args.load_start, datetime.now(UTC)))
        try:
            dataset = load_tierd_dataset(
                source,
                rules=rules,
                load_start=args.load_start,
                load_end=load_end,
            )
            evidence_fn = getattr(source, "evidence", None)
            evidence = list(evidence_fn()) if callable(evidence_fn) else []
            fills_fn = getattr(source, "fills", None)
            fills = list(fills_fn()) if callable(fills_fn) else []
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()
        receipt = _receipt(
            dataset=dataset,
            commit=commit,
            clean=clean,
            args=args,
            load_end=load_end,
            rules_sha={"spot": spot_sha, "perp": perp_sha},
            evidence=evidence,
            fills=fills,
        )
        if args.command == "dry-check":
            document = {"receipt": receipt, "summary": tierd.dataset_summary(dataset)}
            out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            return EXIT_OK
        start_index, _ = tierd.replay_indices(dataset)
        g4.require_dataset_id(first_fetch, dataset.dataset_id)
        # re-fetch immediately before the study: an edited comment authorizes nothing
        auth_receipt = g4.confirm_unedited(args.authorization_url, first_fetch, gh)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / RECEIPT_NAME).write_bytes(canonical_json_bytes(receipt))
        decision = run_study(
            dataset,
            output_dir=args.output_dir,
            authorization_url=args.authorization_url,
            authorization_receipt=auth_receipt,
            expected_protocol_digest=protocol.protocol_digest(),
            allow_synthetic_unauthenticated=False,
            require_full_roster=True,
            start_index=start_index,
            fit_stop_index=(
                start_index if args.fit_stop_index is None else args.fit_stop_index
            ),
            enforce_pre_fs1=True,
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
        SealedWindowEmbargoError,
        FileExistsError,
        ExchangeInfoError,
        OSError,
    ) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED
    eligible = decision.get("evidence_eligible") is True
    run_receipt = {
        "schema_version": "core7_mtf_tierd_run_receipt_v1",
        "dataset_id": dataset.dataset_id,
        "expected_protocol_digest": protocol.protocol_digest(),
        "authorization_receipt": auth_receipt.to_dict(),
        "g4_comment_updated_at": first_fetch.updated_at,
        "evidence_eligible": eligible,
        "synthetic_only": decision.get("synthetic_only"),
        "outcome": decision.get("outcome"),
        "underlying_outcome": decision.get("underlying_outcome"),
    }
    (args.output_dir / RUN_RECEIPT_NAME).write_bytes(canonical_json_bytes(run_receipt))
    out.write(json.dumps(decision, indent=2, sort_keys=True, default=str) + "\n")
    if not eligible:
        err.write(
            "not successful: the result is not evidence_eligible "
            f"(outcome={decision.get('outcome')!r})\n"
        )
        return EXIT_NOT_EVIDENCE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
