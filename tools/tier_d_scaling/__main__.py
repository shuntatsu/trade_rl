"""Artificial-only profiling CLI with fixed per-job process-tree limits."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.tier_d_nonrl.tier_d_protocol import REPO, source_head

from .context import prepare
from .run import measure
from .spec import CASES, validate_profile_source
from .summary import summarize
from .supervision import supervised

ROOT = REPO / "output/tier-d-scaling"


def _root(value: Path) -> Path:
    resolved = value.resolve()
    if (
        resolved == ROOT.resolve()
        or not resolved.is_relative_to(ROOT.resolve())
        or any(p.is_symlink() for p in (value, *value.parents))
    ):
        raise ValueError("artificial outputs must stay below the assigned scope")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "measure"):
        sub = actions.add_parser(action)
        sub.add_argument("--root", required=True, type=Path)
        if action == "measure":
            sub.add_argument("--prepared-root", required=True, type=Path)
            sub.add_argument(
                "--case", required=True, choices=[case[0] for case in CASES]
            )
    summary = actions.add_parser("summary")
    summary.add_argument(
        "--measurement-root", required=True, type=Path, action="append"
    )
    args = parser.parse_args(argv)
    if args.action == "summary":
        print(
            json.dumps(
                summarize([_root(p) for p in args.measurement_root]),
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    root = _root(args.root)
    prepared_root = _root(args.prepared_root) if args.action == "measure" else None
    if prepared_root is not None and (
        root.is_relative_to(prepared_root) or prepared_root.is_relative_to(root)
    ):
        raise ValueError("measurement and prepared inputs must be disjoint")
    validate_profile_source({"source_head": source_head()})
    if args.worker:
        result = (
            prepare(root)
            if prepared_root is None
            else measure(root, prepared_root, args.case)
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0 if result["status"] in {"PREPARED", "PREFIX_OBSERVED"} else 1
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.tier_d_scaling",
        "--worker",
        args.action,
        "--root",
        str(root / "attempt"),
    ]
    if prepared_root is not None:
        command += ["--prepared-root", str(prepared_root), "--case", args.case]
    return supervised(command, root)


if __name__ == "__main__":
    raise SystemExit(main())
