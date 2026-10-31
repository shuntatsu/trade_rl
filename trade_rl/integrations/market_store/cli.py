"""Command line for the reusable market-data store.

``python -m trade_rl.integrations.market_store migrate | plan | ingest | export | status``

The connection URL comes only from ``TRADE_RL_MARKET_STORE_URL`` (never an argument, so
it does not enter shell history or the recorded run command line). ``plan`` and
``ingest`` without ``--allow-network`` are offline and need neither PostgreSQL nor the
``store`` extra. Data on or after 2026-11-01T00:00Z (the FS1 sealed window) is refused
unconditionally: there is no flag that lifts it. The FS1 warm-up span
[2026-08-01, 2026-11-01) is refused by default and admitted only with BOTH
``--allow-fs1-warmup-span`` and ``--fs1-warmup-authorization-url`` naming the Issue #810
comment in which the user authorized it; even then FS1 scoring re-acquires its own data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from trade_rl.integrations.market_store.checksum import ChecksumError
from trade_rl.integrations.market_store.contracts import (
    DEFAULT_BASE_ASSETS,
    DEFAULT_INTERVAL,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.ingest import (
    ArchiveFetcher,
    ConflictError,
    IngestError,
    run_ingestion,
)
from trade_rl.integrations.market_store.parse import ArchiveContentError
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

DEFAULT_START = "2020-01-01"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFLICT = 3
EXIT_EMBARGO = 4


def _parse_day(text: str) -> datetime:
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD, got {text!r}"
        ) from error
    return parsed.replace(tzinfo=UTC)


def _csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma separated list")
    return values


def _add_universe(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bases",
        type=_csv,
        default=DEFAULT_BASE_ASSETS,
        help="comma separated base assets (default: %(default)s); quote is USDT",
    )
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--markets",
        type=_csv,
        default=("usds_m", "spot"),
        help="comma separated: usds_m, spot (default: both)",
    )


def _add_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=_parse_day, default=_parse_day(DEFAULT_START))
    parser.add_argument(
        "--end",
        type=_parse_day,
        default=None,
        help="exclusive end day; default is today (UTC), never past the sealed window",
    )


def _add_warmup(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-fs1-warmup-span",
        action="store_true",
        help=(
            "admit the FS1 warm-up span [2026-08-01, 2026-11-01); requires "
            "--fs1-warmup-authorization-url and a separate user authorization. "
            "The sealed window (>= 2026-11-01) can never be admitted."
        ),
    )
    parser.add_argument(
        "--fs1-warmup-authorization-url",
        default=None,
        help="Issue #810 comment URL in which the user authorized the warm-up fetch",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.integrations.market_store",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("migrate", help="apply pending SQL migrations")
    commands.add_parser("status", help="schema, embargo, coverage and recent runs")

    plan = commands.add_parser("plan", help="print the archives that would be ingested")
    _add_universe(plan)
    _add_range(plan)
    _add_warmup(plan)

    ingest = commands.add_parser(
        "ingest",
        help="ingest planned archives (a dry-run plan unless --allow-network)",
    )
    _add_universe(ingest)
    _add_range(ingest)
    _add_warmup(ingest)
    ingest.add_argument(
        "--allow-network",
        action="store_true",
        help="fetch from data.binance.vision; without it only the plan is printed",
    )
    ingest.add_argument(
        "--refetch",
        action="store_true",
        help="re-download already ingested archives and compare their sha256",
    )
    ingest.add_argument(
        "--skip-exchange-info",
        action="store_true",
        help="do not snapshot exchangeInfo (rules) during this run",
    )

    export = commands.add_parser(
        "export", help="build a MarketDataset artifact from the store"
    )
    export.add_argument(
        "--market", choices=[m.value for m in StoreMarket], required=True
    )
    export.add_argument("--bases", type=_csv, default=DEFAULT_BASE_ASSETS)
    export.add_argument("--interval", default=DEFAULT_INTERVAL)
    export.add_argument("--start", type=_parse_day, required=True)
    export.add_argument("--end", type=_parse_day, required=True)
    export.add_argument(
        "--feature-timeframes",
        type=_csv,
        default=("4h", "1d"),
        help="aggregated from the stored base interval (default: %(default)s)",
    )
    export.add_argument(
        "--gap-policy",
        choices=("strict", "non_tradable_rows"),
        default="non_tradable_rows",
    )
    export.add_argument(
        "--rules-snapshot-sha256",
        default=None,
        help="pin the exchangeInfo snapshot (default: latest; recorded in the receipt)",
    )
    export.add_argument("--output-dir", type=Path, required=True)
    _add_warmup(export)
    return parser


def _universe(args: argparse.Namespace) -> Universe:
    markets = tuple(StoreMarket(value) for value in args.markets)
    return Universe.from_bases(args.bases, interval=args.interval, markets=markets)


def _embargo(args: argparse.Namespace, err: TextIO) -> EmbargoPolicy:
    flag = bool(getattr(args, "allow_fs1_warmup_span", False))
    url = getattr(args, "fs1_warmup_authorization_url", None)
    if flag != (url is not None):
        raise ValueError(
            "--allow-fs1-warmup-span and --fs1-warmup-authorization-url must be "
            "given together"
        )
    if not flag:
        return EmbargoPolicy.enforced()
    assert url is not None
    policy = EmbargoPolicy.with_warmup(url)
    print(
        "WARNING: the FS1 warm-up span is ADMITTED for this command "
        f"(authorization: {url}). The sealed window stays closed, and FS1 scoring "
        "still re-acquires and re-verifies its own data through its sealed tool.",
        file=err,
    )
    return policy


def _default_end(now: datetime, policy: EmbargoPolicy) -> datetime:
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return min(today, policy.start)


def _code_commit() -> str:
    override = os.environ.get("TRADE_RL_CODE_COMMIT")
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dump(payload: object, out: TextIO) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), file=out)


def _make_plan(
    args: argparse.Namespace, policy: EmbargoPolicy, now: datetime
) -> IngestionPlan:
    end = args.end if args.end is not None else _default_end(now, policy)
    return plan_ingestion(
        _universe(args), start=args.start, end=end, now=now, embargo=policy
    )


def _ingest_exchange_info(store: Any, plan: IngestionPlan, out: TextIO) -> None:
    from trade_rl.integrations.binance import BinancePublicTransport
    from trade_rl.integrations.market_store.contracts import StoredExchangeInfo
    from trade_rl.integrations.market_store.rules import parse_symbol_rules

    transport = BinancePublicTransport(allow_network=True)
    for market in plan.universe.markets:
        snapshot = transport.load_exchange_information_snapshot(
            market=market.binance, mode="rest"
        )
        info = StoredExchangeInfo(
            market=market,
            raw_payload=snapshot.raw_payload,
            raw_payload_sha256=snapshot.raw_payload_sha256,
            source_uri=snapshot.source_uri,
            retrieved_at=snapshot.retrieved_at,
        )
        rules = parse_symbol_rules(
            info.raw_payload, market=market, symbols=plan.universe.symbols
        )
        stored = store.store_exchange_info(info, rules, run_id=None)
        print(
            f"exchangeInfo {market.value}: "
            f"{'stored' if stored else 'already stored'} {info.raw_payload_sha256}",
            file=out,
        )


def run(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    clock: Callable[[], datetime] | None = None,
    fetcher: ArchiveFetcher | None = None,
) -> int:
    stdout = sys.stdout if out is None else out
    stderr = sys.stderr if err is None else err
    now = (clock or (lambda: datetime.now(UTC)))()
    args = build_parser().parse_args(argv)
    command_line = "python -m trade_rl.integrations.market_store " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    try:
        if args.command == "plan":
            policy = _embargo(args, stderr)
            _dump(_make_plan(args, policy, now).to_dict(), stdout)
            return EXIT_OK

        if args.command == "ingest":
            policy = _embargo(args, stderr)
            plan = _make_plan(args, policy, now)
            if not args.allow_network:
                _dump(
                    {"dry_run": True, "network": False, "plan": plan.to_dict()}, stdout
                )
                print(
                    "dry run: nothing was fetched or written (pass --allow-network "
                    "to ingest)",
                    file=stderr,
                )
                return EXIT_OK
            from trade_rl.integrations.market_store.store import PostgresMarketStore

            if fetcher is None:
                from trade_rl.integrations.market_store.fetch import VisionFetcher

                fetcher = VisionFetcher(
                    warmup_authorization_url=plan.embargo.warmup_authorization_url
                )
            with PostgresMarketStore.open() as store:
                if not store.schema_versions():
                    raise IngestError("the store is not migrated; run `migrate` first")
                if not args.skip_exchange_info:
                    _ingest_exchange_info(store, plan, stderr)
                report = run_ingestion(
                    store,
                    fetcher,
                    plan,
                    code_commit=_code_commit(),
                    command_line=command_line,
                    refetch=args.refetch,
                    clock=clock,
                )
            _dump(report.to_dict() | {"skipped_plan_items": len(plan.skipped)}, stdout)
            return EXIT_OK

        if args.command == "migrate":
            from trade_rl.integrations.market_store.store import PostgresMarketStore

            with PostgresMarketStore.open() as store:
                applied = store.migrate()
                _dump(
                    {
                        "applied": [f"{m.version:04d}_{m.name}" for m in applied],
                        "schema_versions": [list(v) for v in store.schema_versions()],
                    },
                    stdout,
                )
            return EXIT_OK

        if args.command == "status":
            from trade_rl.integrations.market_store.store import PostgresMarketStore

            with PostgresMarketStore.open() as store:
                _dump(store.status(), stdout)
            return EXIT_OK

        if args.command == "export":
            from trade_rl.integrations.binance.dataset import BinanceGapPolicy
            from trade_rl.integrations.market_store.export import (
                export_market_dataset,
                publish_export,
            )
            from trade_rl.integrations.market_store.store import PostgresMarketStore

            policy = _embargo(args, stderr)
            market = StoreMarket(args.market)
            universe = Universe.from_bases(
                args.bases, interval=args.interval, markets=(market,)
            )
            with PostgresMarketStore.open() as store:
                export = export_market_dataset(
                    store,
                    market=market,
                    symbols=universe.symbols,
                    interval=args.interval,
                    start_time=args.start,
                    end_time=args.end,
                    feature_timeframes=args.feature_timeframes,
                    gap_policy=BinanceGapPolicy(args.gap_policy),
                    rules_snapshot_sha256=args.rules_snapshot_sha256,
                    embargo=policy,
                )
            _dump(publish_export(export, args.output_dir), stdout)
            return EXIT_OK
    except SealedWindowEmbargoError as error:
        print(f"REFUSED (sealed window): {error}", file=stderr)
        return EXIT_EMBARGO
    except ConflictError as error:
        print(f"REFUSED (conflict): {error}", file=stderr)
        return EXIT_CONFLICT
    except (
        ArchiveContentError,
        ChecksumError,
        IngestError,
        FileExistsError,
        LookupError,
        ValueError,
        RuntimeError,
        ImportError,
    ) as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=stderr)
        return EXIT_FAILURE
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


__all__ = [
    "EXIT_CONFLICT",
    "EXIT_EMBARGO",
    "EXIT_FAILURE",
    "EXIT_OK",
    "build_parser",
    "main",
    "run",
]
