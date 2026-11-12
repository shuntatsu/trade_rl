"""Command line: ``python -m trade_rl.integrations.cross_asset fetch|build|status``.

The CLI reads and writes only the cross-asset data directory. It never reads
crypto prices and computes no relation to them.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TextIO

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.integrations.cross_asset.catalog import (
    CATALOG,
    SOURCES,
    SeriesSpec,
    series_by_key,
)
from trade_rl.integrations.cross_asset.features import (
    COMPOSITE_FEATURE_NAMES,
    FEATURE_SET_ID,
    definitions_digest,
)
from trade_rl.integrations.cross_asset.fetch import UrllibTransport, fetch_all
from trade_rl.integrations.cross_asset.loader import (
    DEFAULT_ROOT,
    load_cross_asset_features,
    regular_grid,
)
from trade_rl.integrations.cross_asset.pit import knowable_at_utc
from trade_rl.integrations.cross_asset.store import (
    CrossAssetStore,
    captured_at,
    format_capture_utc,
    format_utc,
    latest_records,
    parse_utc,
)


def _utc_date(text: str) -> datetime:
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD, got {text!r}"
        ) from error
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _utc_instant(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected ISO-8601 UTC, got {text!r}"
        ) from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset or Z")
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.integrations.cross_asset",
        description="Point-in-time cross-asset market context (no crypto data).",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="data directory (git-ignored)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download raw series and record vintages")
    fetch.add_argument("--sources", default=",".join(SOURCES), help="comma separated")
    fetch.add_argument("--series", default="", help="comma separated series keys")
    fetch.add_argument(
        "--delay", type=float, default=0.4, help="seconds between requests"
    )

    build = commands.add_parser("build", help="align features to a UTC decision grid")
    build.add_argument("--timeframe", default="1d", help="1d, 4h, ...")
    build.add_argument("--start", type=_utc_date, required=True)
    build.add_argument("--end", type=_utc_date, required=True)
    build.add_argument(
        "--as-of", type=_utc_instant, default=None, help="vintage cutoff"
    )
    build.add_argument(
        "--features",
        choices=("composites", "all"),
        default="composites",
        help="composites only (default) or every per-series feature as well",
    )
    build.add_argument(
        "--opt-in",
        default="",
        help="comma separated series keys with revision_risk other than low that "
        "composites may use (default: none)",
    )

    commands.add_parser("status", help="per-series coverage, vintages and lag audit")
    return parser


def _select(args: argparse.Namespace) -> tuple[SeriesSpec, ...]:
    sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    unknown = sources - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")
    keys = [item.strip() for item in args.series.split(",") if item.strip()]
    if keys:
        specs = tuple(series_by_key(key) for key in keys)
    else:
        specs = CATALOG
    return tuple(spec for spec in specs if spec.source in sources)


def _fetch(args: argparse.Namespace, out: TextIO) -> int:
    store = CrossAssetStore(args.root)
    records = fetch_all(
        store, _select(args), UrllibTransport(), delay_seconds=args.delay
    )
    usable = 0
    for record in records:
        status = str(record["status"])
        usable += status in {"ok", "unchanged"}
        if status in {"ok", "unchanged"}:
            detail = (
                f"rows={record['n_obs']} {record['first_obs']}..{record['last_obs']} "
                f"missing_rows={record['missing_rows']}"
            )
            revision = record.get("revision")
            if isinstance(revision, dict) and (
                revision.get("changed") or revision.get("dropped")
            ):
                detail += f" REVISED {revision}"
        else:
            detail = str(record.get("reason", ""))
        out.write(f"{record['key']:<14} {record['source']:<6} {status:<11} {detail}\n")
    out.write(
        f"{usable}/{len(records)} series usable; manifest {store.manifest_path}\n"
    )
    return 0 if usable else 2


def _expected_frontier(spec: SeriesSpec, fetched_at: datetime) -> date | None:
    """Latest weekday whose configured release time had passed at fetch time."""

    day = fetched_at.astimezone(UTC).date() + timedelta(days=1)
    for _ in range(40):
        day -= timedelta(days=1)
        if day.weekday() >= 5:
            continue
        stamp = knowable_at_utc(
            day,
            venue_tz=spec.venue_tz,
            close_local=spec.close_local,
            rule=spec.release,
        )
        if stamp <= fetched_at:
            return day
    return None


def _was_revised(record: dict[str, object]) -> bool:
    revision = record.get("revision")
    return isinstance(revision, dict) and bool(
        revision.get("changed") or revision.get("dropped")
    )


def status_report(store: CrossAssetStore) -> list[dict[str, object]]:
    records = store.read_manifest()
    latest = latest_records(records)
    rows: list[dict[str, object]] = []
    for spec in CATALOG:
        history = [r for r in records if r["key"] == spec.key]
        row: dict[str, object] = {
            "key": spec.key,
            "source": spec.source,
            "attempts": len(history),
            "vintages": len({r.get("sha256") for r in history if r.get("path")}),
            "revisions_seen": sum(1 for r in history if _was_revised(r)),
        }
        record = latest.get(spec.key)
        if record is None:
            last = history[-1] if history else None
            row["state"] = "never fetched" if last is None else str(last["status"])
            row["reason"] = None if last is None else last.get("reason")
        else:
            fetched_at = parse_utc(captured_at(record))
            frontier = _expected_frontier(spec, fetched_at)
            last_row = record.get("last_row_date")
            behind = (
                frontier is not None
                and isinstance(last_row, str)
                and date.fromisoformat(last_row) < frontier
            )
            row.update(
                state="ok" if store.verify_raw(record) else "RAW FILE FAILS DIGEST",
                n_obs=record["n_obs"],
                first_obs=record["first_obs"],
                last_obs=record["last_obs"],
                missing_rows=record["missing_rows"],
                last_fetch=record["fetched_at"],
                lag_audit=(
                    f"BEHIND configured lag (expected a row for {frontier})"
                    if behind
                    else "consistent"
                ),
            )
        rows.append(row)
    return rows


def _status(args: argparse.Namespace, out: TextIO) -> int:
    store = CrossAssetStore(args.root)
    rows = status_report(store)
    for row in rows:
        if "n_obs" in row:
            out.write(
                f"{row['key']:<14} {row['source']:<6} {row['state']:<8} "
                f"n={row['n_obs']} {row['first_obs']}..{row['last_obs']} "
                f"missing_rows={row['missing_rows']} vintages={row['vintages']} "
                f"revisions={row['revisions_seen']} lag={row['lag_audit']}\n"
            )
        else:
            out.write(
                f"{row['key']:<14} {row['source']:<6} {row['state']:<8} "
                f"{row.get('reason') or ''}\n"
            )
    out.write(f"feature set {FEATURE_SET_ID} definitions {definitions_digest()}\n")
    return 0


def _build(args: argparse.Namespace, out: TextIO) -> int:
    grid = regular_grid(args.start, args.end, args.timeframe)
    opt_in = tuple(item.strip() for item in args.opt_in.split(",") if item.strip())
    frame = load_cross_asset_features(
        args.root, grid, as_of=args.as_of, opt_in_series=opt_in
    )
    if args.features == "composites":
        frame = frame.select(COMPOSITE_FEATURE_NAMES)
    tag = f"{args.timeframe}_{args.start:%Y%m%d}_{args.end:%Y%m%d}_{frame.input_digest[:8]}"
    target = Path(args.root) / "built" / FEATURE_SET_ID / tag
    arrays = {
        "decision_ns": frame.decision_ns,
        "values": frame.values,
        "available": frame.available,
        "staleness_hours": frame.staleness_hours,
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atomic_write_bytes(target.with_suffix(".npz"), buffer.getvalue())
    meta = {
        "feature_set": frame.feature_set,
        "definitions_digest": frame.definitions_digest,
        "input_digest": frame.input_digest,
        "names": list(frame.names),
        "series_present": list(frame.series_present),
        "opt_in_series": list(frame.opt_in_series),
        "rows": int(frame.decision_ns.size),
        "timeframe": args.timeframe,
        "start": format_utc(args.start),
        "end": format_utc(args.end),
        "as_of": None if args.as_of is None else format_capture_utc(args.as_of),
        "built_at": format_capture_utc(datetime.now(UTC)),
    }
    atomic_write_bytes(
        target.with_suffix(".json"),
        (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    out.write(
        f"{len(frame.names)} features x {frame.decision_ns.size} rows -> {target}\n"
    )
    for index, name in enumerate(frame.names[: len(COMPOSITE_FEATURE_NAMES)]):
        share = float(frame.available[:, index].mean())
        out.write(f"  {name:<34} available {share:6.1%}\n")
    return 0


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(argv)
    handlers = {"fetch": _fetch, "build": _build, "status": _status}
    return handlers[args.command](args, stream)


__all__ = ["build_parser", "main", "status_report"]
