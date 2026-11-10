"""Command line of the pre-registered multi-symbol funding-carry development screen.

``python -m tools.carry_multisymbol_screen plan | screen``

The wiring lives here, outside ``trade_rl``, because the market-data store package
may be imported by nothing else under ``trade_rl`` (an architecture test pins that).
The economics, protocol and decision rule are ``trade_rl.evaluation.carry_multisymbol``.

``plan`` is offline: it prints the registered protocol, its digest, the archive
counts and the exact owner authorization text. ``screen`` refuses to start without
``--g4-authorization-url`` (an issue comment URL), refuses every window except the
registered one (nothing at or after 2026-08-01 is ever admitted) and needs a clean
checkout whose commit it records. Archives are read from ``--archive-dir`` and are
fetched from Binance Vision only when ``--allow-network`` is given; every archive is
verified against its official checksum. ``--source store`` reads a populated market
store instead (connection URL only from ``TRADE_RL_MARKET_STORE_URL``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from trade_rl.evaluation.carry_multisymbol import protocol
from trade_rl.evaluation.carry_multisymbol.runner import (
    ScreenData,
    ScreenRequest,
    run_screen,
)
from trade_rl.integrations.market_store.carry_source import (
    ArchiveDirFetcher,
    CarryRowSource,
    CarryScreenData,
    VisionArchiveSource,
)
from trade_rl.integrations.market_store.contracts import StoreMarket, Universe
from trade_rl.integrations.market_store.embargo import SealedWindowEmbargoError
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

EXIT_OK = 0
EXIT_REFUSED = 2

GitState = Callable[[], tuple[str, bool]]
RowSourceFactory = Callable[[IngestionPlan], CarryRowSource]


def _parse_day(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD, got {text!r}"
        ) from error


def build_plan(now: datetime) -> IngestionPlan:
    """The archive plan: lookback klines, window klines and window funding only."""
    universe = Universe(
        symbols=(*protocol.CANDIDATES, *protocol.REFERENCE),
        interval="1h",
        markets=(StoreMarket.USDS_M, StoreMarket.SPOT),
    )
    plan = plan_ingestion(
        universe,
        start=protocol.LOOKBACK_START,
        end=protocol.WINDOW_END,
        now=now,
    )
    specs = tuple(
        spec
        for spec in plan.specs
        if spec.kind.value == "klines" or spec.period_end > protocol.WINDOW_START
    )
    return replace(plan, specs=specs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.carry_multisymbol_screen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="offline: protocol, digest, archive plan")
    plan.add_argument("--code-commit", default="<CODE_COMMIT>")
    screen = commands.add_parser("screen", help="run the registered development screen")
    screen.add_argument("--start", type=_parse_day, required=True)
    screen.add_argument("--end", type=_parse_day, required=True)
    screen.add_argument("--g4-authorization-url", default=None)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--source", choices=("vision", "store"), default="vision")
    screen.add_argument("--archive-dir", type=Path, default=None)
    screen.add_argument("--allow-network", action="store_true")
    screen.add_argument(
        "--coverage-only",
        action="store_true",
        help="load, check coverage and write the inclusion report; replay nothing",
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

            inner = VisionFetcher()
        return VisionArchiveSource(ArchiveDirFetcher(archive_dir, inner), plan)

    return factory


def _store_rows(plan: IngestionPlan) -> CarryRowSource:
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    return PostgresMarketStore.open()


def _screen_data(rows: CarryRowSource) -> ScreenData:
    return CarryScreenData(
        rows,
        lookback_start=protocol.LOOKBACK_START,
        window_start=protocol.WINDOW_START,
        window_end=protocol.WINDOW_END,
        halt_intervals=protocol.SPOT_HALTS,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: GitState = git_state,
    rows_factory: RowSourceFactory | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan = build_plan(datetime.now(UTC))
        kinds = sorted({spec.kind.value for spec in plan.specs})
        document = {
            "protocol": protocol.protocol_document(),
            "protocol_digest": protocol.protocol_digest(),
            "archives_planned": {
                kind: sum(spec.kind.value == kind for spec in plan.specs)
                for kind in kinds
            },
            "required_symbol_passes": {
                str(count): protocol.required_symbol_passes(count)
                for count in range(protocol.MIN_ADMITTED, len(protocol.CANDIDATES) + 1)
            },
            "g4_authorization_text": protocol.lease_text(code_commit=args.code_commit),
        }
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    try:
        protocol.validate_window(args.start, args.end)
        if args.g4_authorization_url is None:
            raise ValueError(
                "a G4 authorization comment URL is required: --g4-authorization-url"
            )
        protocol.validate_g4_authorization_url(args.g4_authorization_url)
        if rows_factory is None:
            if args.source == "vision":
                if args.archive_dir is None:
                    raise ValueError("--source vision requires --archive-dir")
                rows_factory = _vision_rows(args.archive_dir, args.allow_network)
            else:
                rows_factory = _store_rows
        commit, clean = state()
        if not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        decision = run_screen(
            ScreenRequest(
                start=args.start,
                end=args.end,
                g4_authorization_url=args.g4_authorization_url,
                output_dir=args.output_dir,
                code_commit=commit,
                coverage_only=args.coverage_only,
            ),
            _screen_data(rows_factory(build_plan(datetime.now(UTC)))),
            log=lambda message: print(message, file=err),
        )
    except (ValueError, SealedWindowEmbargoError, FileExistsError) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED
    out.write(json.dumps(decision, indent=2, sort_keys=True, default=str) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
