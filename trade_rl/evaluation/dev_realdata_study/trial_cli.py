"""Command line entry for the Phase 2 trial registry run (writes new files only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np

from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import _git_head, load_dataset
from trade_rl.evaluation.dev_realdata_study.trial_study import (
    PROBE_MATRIX_CUT,
    compact_phase2_summary,
    run_phase2,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
        "--causality-matrix-cut",
        default=PROBE_MATRIX_CUT,
        help="cut of the per-arm, per-scenario future-perturbation probe",
    )
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
    started_utc = _now()
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
        matrix_cut=args.causality_matrix_cut,
    )
    with (output_dir / "equity.npz").open("xb") as stream:
        np.savez_compressed(stream, **equity)
    with (output_dir / "full_results.json").open("xb") as stream:
        stream.write(_json_bytes(results))
    metadata = {
        "started_utc": started_utc,
        "finished_utc": _now(),
        "git_head": _git_head(),
        "argv": list(sys.argv if argv is None else ["trial_cli", *argv]),
        "python": platform.python_version(),
        "workers": None if args.sequential else args.workers,
        "sequential": bool(args.sequential),
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in ("equity.npz", "full_results.json")
        },
    }
    with (output_dir / "run_metadata.json").open("xb") as stream:
        stream.write(_json_bytes(metadata))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("xb") as stream:
        stream.write(
            _json_bytes({**compact_phase2_summary(results), "run_metadata": metadata})
        )
    print(
        f"label={results['label']} (PR-C probe rule, primary) "
        f"amended_rule_label={results['label_amended_rule']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
