"""Command line for the FS1 acquisition and the seal-record generator (Issue #810).

``acquire`` plans by default and fetches only with ``--allow-network`` after an Issue
#810 data-fetch authorization comment URL; ``seal-record`` prints the Markdown body of
the future seal comment. Nothing here posts, scores or reads a price. The entrypoint
takes no registry, digest, clock, code-identity or transport seam.

Exit codes: 0 planned/acquired/printed, 2 refused (the reasons are on stderr).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

PROG = "python -m trade_rl.evaluation.forward_acquisition"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "FS1 data acquisition (plan by default; fetch only with --allow-network "
            "and an Issue #810 authorization comment URL) and the seal-record printer."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    acquire = commands.add_parser(
        "acquire",
        help="plan, or with --allow-network fetch, build and write the seal manifest",
    )
    acquire.add_argument(
        "--authorization",
        required=True,
        help="Issue #810 data-fetch authorization comment URL",
    )
    acquire.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new directory; an existing path is refused",
    )
    acquire.add_argument(
        "--allow-network",
        action="store_true",
        help="actually fetch (otherwise only the plan is printed)",
    )
    acquire.add_argument(
        "--dev-archive-root",
        type=Path,
        help="local development archives (klines/<SYMBOL>/1h, fundingRate/<SYMBOL>)",
    )
    acquire.add_argument(
        "--dev-source-evidence",
        type=Path,
        help="source-evidence.json of the development dataset build",
    )
    acquire.add_argument(
        "--dev-dataset",
        type=Path,
        help="published development Dataset artifact directory (read only)",
    )

    record = commands.add_parser(
        "seal-record",
        help="print the Markdown body of the Issue #810 seal comment (never posts)",
    )
    record.add_argument("--output-root", type=Path, required=True)
    record.add_argument("--sealed-code-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the operator's values and call the entrypoints (no seams here)."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    # Imported after parsing so that ``--help`` and a malformed command line stay cheap;
    # neither module imports a network client at import time.
    from trade_rl.evaluation.forward_acquisition.acquire import (
        AcquisitionRefusal,
        run_acquire,
    )
    from trade_rl.evaluation.forward_acquisition.seal_record import render_seal_record

    try:
        if args.command == "seal-record":
            sys.stdout.write(render_seal_record(args.output_root, args.sealed_code_sha))
            return 0
        result = run_acquire(
            authorization=args.authorization,
            output_root=args.output_root,
            allow_network=args.allow_network,
            dev_archive_root=args.dev_archive_root,
            dev_source_evidence=args.dev_source_evidence,
            dev_dataset=args.dev_dataset,
            command_line=f"{PROG} {shlex.join(arguments)}",
            emit=lambda message: print(message, flush=True),
        )
    except AcquisitionRefusal as refusal:
        detail = f" ({refusal.detail})" if refusal.detail else ""
        print(f"refused: {', '.join(refusal.reasons)}{detail}", file=sys.stderr)
        return 2
    if result.planned_only:
        print("\n".join(result.plan_lines))
        return 0
    print(f"manifest={result.manifest_path}")
    print(f"manifest_sha256={result.manifest_sha256}")
    print(f"dataset={result.dataset_path}")
    print(f"dataset_id={result.dataset_id}")
    print(f"fetched_utc={result.fetched_utc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
