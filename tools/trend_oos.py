"""Driver for the out-of-sample test ``trend_long_spot_oos_v1``.

``python -m tools.trend_oos plan --stage BACKCAST|FORWARD``
``python -m tools.trend_oos run --stage BACKCAST``
``python -m tools.trend_oos run --stage FORWARD --interim | --final``

``plan`` prints the frozen protocol, its digest and the stage windows. It loads no data
and computes no return.

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
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.fetch import VisionFetcher
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion
from trade_rl.integrations.market_store.rules import ExchangeInfoError

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3
EXIT_INVALID = 4
EXIT_UNAUTHENTICATED = 5
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
    halt_intervals: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
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
    start_ms = ms_from_utc(window.load_start)
    end_ms = ms_from_utc(limit_close)
    first_open: int | None = None
    unpublished_months = 0
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
                continue
            raise
        if rows:
            first_open = int(rows[0].open_time_ms)
            break
        if first_open is None:
            unpublished_months += 1
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
    halt_intervals: Sequence[tuple[datetime, datetime]] = SPOT_HALTS,
    data_authorization_url: str | None = None,
) -> tuple[MarketDataset, dict[str, dict[str, object]]]:
    """Read every roster symbol through the stage guard and assemble the dataset."""
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


def _sources(
    args: argparse.Namespace, plan: IngestionPlan, policy: EmbargoPolicy
) -> tuple[CarryRowSource, CarryRowSource]:
    """``(detect, reader)``: the primary source and, with the fallback, its wrapper."""
    base = _vision_factory(args.archive_dir, args.allow_network, policy)
    primary = base(plan)
    if not args.daily_fallback:
        return primary, primary
    daily = _vision_factory(args.archive_dir, args.allow_network, policy)
    return primary, tierd_driver.DailyFallbackSource(
        primary, daily, halts=SPOT_HALTS, embargo=policy
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
