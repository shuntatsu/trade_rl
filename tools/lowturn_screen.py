"""Driver for the low-turnover family screen (``lowturn_family_screen_v1``).

``python -m tools.lowturn_screen plan | run``

``plan`` prints the frozen protocol, its digest and the DEV/HOLDOUT segment indices of the
Tier D clock. It loads no data and computes no return.

``run`` builds the Tier D dataset through the same code path and arguments as
``tools.core7_mtf_tierd run`` (``--load-start``, ``--source``, ``--daily-fallback``,
``--archive-dir``, ``--exchange-info-spot/perp``), refuses any dataset whose id is not the
frozen Tier D id, and authenticates the owner authorization comment (``--authorization-url``,
an issue #810 comment) with ``tools.core7_g4_comment``: it must quote this protocol's id
and digest and the dataset id, and must not change between the first fetch and the run.
It requires a clean checkout and a new output directory, and writes every cell, the
manifest and ``decision.json`` before it prints the label. The driver authorizes nothing
on its own: the protocol runs once on known-used data and its strongest label is
``EXPLORATORY_CANDIDATE``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from tools import core7_g4_comment as g4
from tools import core7_mtf_tierd as tierd_driver
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.core7_mtf_study import tierd
from trade_rl.evaluation.lowturn_screen import protocol, runner
from trade_rl.integrations.binance.carry import CarryScreenHalt
from trade_rl.integrations.market_store.embargo import SealedWindowEmbargoError
from trade_rl.integrations.market_store.rules import ExchangeInfoError

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3
EXIT_INVALID = 4
EXIT_UNAUTHENTICATED = 5
DATA_RECEIPT_NAME = "lowturn-data-receipt.json"
RUN_RECEIPT_NAME = "lowturn-run-receipt.json"
DATA_RECEIPT_SCHEMA = "lowturn_screen_data_receipt_v1"
RUN_RECEIPT_SCHEMA = "lowturn_screen_run_receipt_v1"


def plan_document(load_start: datetime) -> dict[str, Any]:
    """Protocol, digest and segment indices of the Tier D clock; reads no data."""
    start, end = tierd.validate_load_window(load_start, tierd.LOAD_END)
    first_close = start + timedelta(hours=tierd.BAR_HOURS)
    n_bars = int((end - first_close) / timedelta(hours=tierd.BAR_HOURS)) + 1
    clock = (
        np.datetime64(first_close.replace(tzinfo=None), "ns")
        + np.arange(n_bars) * np.timedelta64(tierd.BAR_HOURS, "h")
    ).astype("datetime64[ns]")
    bounds = protocol.segment_bounds(clock)
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": protocol.FROZEN_DATASET_ID,
        "claim_ceiling": protocol.CLAIM_CEILING,
        "clock": {
            "load_start": start.isoformat(),
            "first_bar_close": str(clock[0]) + "Z",
            "last_bar_close": str(clock[-1]) + "Z",
            "n_bars": n_bars,
        },
        "segments": {
            name: {
                "start_index": lo,
                "stop_index": hi,
                "start_bar_close": str(clock[lo]) + "Z",
                "stop_bar_close": str(clock[hi]) + "Z",
            }
            for name, (lo, hi) in bounds.items()
        },
        "protocol": protocol.protocol_payload(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.lowturn_screen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="print the protocol, its digest and the segment indices"
    )
    plan.add_argument(
        "--load-start",
        type=tierd_driver._parse_day,
        default=tierd_driver.DEFAULT_LOAD_START,
    )
    run = commands.add_parser(
        "run", help="run the screen on the Tier D dataset (needs owner authorization)"
    )
    run.add_argument(
        "--load-start",
        type=tierd_driver._parse_day,
        default=tierd_driver.DEFAULT_LOAD_START,
        help="warm-up start (default 2021-10-01, as the Phase 1 run)",
    )
    run.add_argument("--source", choices=("vision", "store"), default="vision")
    run.add_argument("--archive-dir", type=Path, default=None)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument(
        "--daily-fallback",
        action="store_true",
        help="fill hours missing from the primary source from checksum-verified "
        "Vision daily archives (needs --archive-dir); provenance is recorded",
    )
    run.add_argument("--exchange-info-spot", type=Path, required=True)
    run.add_argument("--exchange-info-perp", type=Path, required=True)
    run.add_argument("--authorization-url", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def _data_receipt(
    *,
    dataset_id: str,
    commit: str,
    args: argparse.Namespace,
    load_end: datetime,
    rules_sha: dict[str, str],
    evidence: list[dict[str, Any]],
    fills: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": DATA_RECEIPT_SCHEMA,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "tierd_contract": tierd.TIERD_CONTRACT,
        "dataset_id": dataset_id,
        "load_start": args.load_start.isoformat(),
        "load_end": load_end.isoformat(),
        "source": args.source,
        "exchange_info_sha256": rules_sha,
        "archive_evidence_count": len(evidence),
        "archive_evidence_digest": tierd_driver._digest(evidence) if evidence else None,
        "daily_fallback": {
            "enabled": bool(args.daily_fallback),
            "filled_hours": len(fills),
            "fills_digest": tierd_driver._digest(fills) if fills else None,
            "fills": fills,
            "forward_fill": False,
            "rest_fallback": False,
        },
        "code_commit": commit,
        "working_tree_clean": True,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: tierd_driver.GitState = tierd_driver.git_state,
    rows_factory: tierd_driver.RowSourceFactory | None = None,
    load_end: datetime = tierd.LOAD_END,
    gh: g4.GhRunner = g4.default_gh,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        try:
            document = plan_document(args.load_start)
        except ValueError as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    try:
        # the authorization shape is checked before any file or row is touched
        core7_protocol.validate_authorization_url(args.authorization_url)
        if args.output_dir.exists():
            raise FileExistsError(f"{args.output_dir} exists; outputs are write-once")
        # authenticate the owner comment before any file, row or data is read
        first_fetch = g4.fetch_comment(args.authorization_url, gh)
        g4.require_protocol_digest(first_fetch, protocol.protocol_digest())
        g4.require_protocol_id(first_fetch, protocol.PROTOCOL_ID)
        tierd.validate_load_window(args.load_start, load_end)
        commit, clean = state()
        if not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        spot_payload, spot_sha = tierd_driver._read_payload(args.exchange_info_spot)
        perp_payload, perp_sha = tierd_driver._read_payload(args.exchange_info_perp)
        rules = tierd_driver.load_rules(spot_payload, perp_payload)
        if rows_factory is None:
            rows_factory = tierd_driver._resolve_rows(args)
        source = rows_factory(
            tierd_driver.build_plan(args.load_start, datetime.now(UTC))
        )
        try:
            dataset = tierd_driver.load_tierd_dataset(
                source, rules=rules, load_start=args.load_start, load_end=load_end
            )
            evidence_fn = getattr(source, "evidence", None)
            evidence = list(evidence_fn()) if callable(evidence_fn) else []
            fills_fn = getattr(source, "fills", None)
            fills = list(fills_fn()) if callable(fills_fn) else []
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()
        if dataset.dataset_id != protocol.FROZEN_DATASET_ID:
            raise ValueError(
                "dataset id differs from the frozen Tier D dataset id "
                f"{protocol.FROZEN_DATASET_ID}: got {dataset.dataset_id}"
            )
        g4.require_dataset_id(first_fetch, dataset.dataset_id)
        # re-fetch immediately before the screen: an edited comment authorizes nothing
        auth_receipt = g4.confirm_unedited(
            args.authorization_url,
            first_fetch,
            gh,
            protocol_digest=protocol.protocol_digest(),
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / DATA_RECEIPT_NAME).write_bytes(
            canonical_json_bytes(
                _data_receipt(
                    dataset_id=dataset.dataset_id,
                    commit=commit,
                    args=args,
                    load_end=load_end,
                    rules_sha={"spot": spot_sha, "perp": perp_sha},
                    evidence=evidence,
                    fills=fills,
                )
            )
        )
        decision = runner.run_screen(
            dataset,
            output_dir=args.output_dir,
            expected_protocol_digest=protocol.protocol_digest(),
            expected_dataset_id=protocol.FROZEN_DATASET_ID,
            code_commit=commit,
            authorization={
                **auth_receipt.to_dict(),
                "g4_comment_updated_at": first_fetch.updated_at,
            },
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
    run_receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": dataset.dataset_id,
        "authorization_receipt": auth_receipt.to_dict(),
        "g4_comment_updated_at": first_fetch.updated_at,
        "code_commit": commit,
        "label": decision["label"],
        "claim_ceiling": protocol.CLAIM_CEILING,
    }
    (args.output_dir / RUN_RECEIPT_NAME).write_bytes(canonical_json_bytes(run_receipt))
    out.write(
        json.dumps(
            {
                "label": decision["label"],
                "claim_ceiling": protocol.CLAIM_CEILING,
                "dev_pass_families": decision["dev_pass_families"],
                "confirmed_families": decision["confirmed_families"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if decision["label"] == protocol.LABEL_INVALID:
        err.write("not successful: a control cell is invalid (label INVALID)\n")
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
