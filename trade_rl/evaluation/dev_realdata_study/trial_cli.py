"""Command line entry for the Phase 2 trial registry run (writes new files only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np

from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import load_dataset
from trade_rl.evaluation.dev_realdata_study.trial_study import (
    compact_phase2_summary,
    run_phase2,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.dev_realdata_study.trial_cli",
        description=(
            "Run the registered Issue #810 Phase 2 trial registry (T1-T8 plus the "
            "five PR-C arms). Every parameter is fixed by trials.py / protocol.py; "
            "nothing is tunable here."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="run every arm in this process instead of one process per arm",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir: Path = args.output_dir
    summary_path: Path = args.summary_path
    if (output_dir / "full_results.json").exists() or summary_path.exists():
        print("refusing to overwrite an existing study output", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    _, dataset = load_published_market_dataset_artifact(args.dataset)
    results, equity = run_phase2(
        dataset,
        PROTOCOL,
        progress=lambda message: print(message, flush=True),
        parallel_loader=None
        if args.sequential
        else partial(load_dataset, str(args.dataset)),
        max_workers=args.workers,
    )
    with (output_dir / "equity.npz").open("xb") as stream:
        np.savez_compressed(stream, **equity)
    with (output_dir / "full_results.json").open("xb") as stream:
        stream.write(_json_bytes(results))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("xb") as stream:
        stream.write(_json_bytes(compact_phase2_summary(results)))
    print(f"label={results['label']} selected={results['selection']['selected_arm']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
