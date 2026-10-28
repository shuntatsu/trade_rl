"""Command line entry for the development real-data study (writes new files only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import run_study

_SUMMARY_DROP = ("protocol",)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def compact_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Small tracked summary: no per-interval data, only the headline evidence."""

    summary = {key: value for key, value in results.items() if key not in _SUMMARY_DROP}
    summary["protocol_deviations"] = list(PROTOCOL.deviations)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.dev_realdata_study",
        description=(
            "Run the preregistered Issue #810 P2 development real-data study. "
            "Parameters are fixed by the protocol module; nothing is tunable here."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
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
    results, equity = run_study(
        dataset, PROTOCOL, progress=lambda message: print(message, flush=True)
    )
    with (output_dir / "equity.npz").open("xb") as stream:
        np.savez_compressed(stream, **equity)
    with (output_dir / "full_results.json").open("xb") as stream:
        stream.write(_json_bytes(results))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("xb") as stream:
        stream.write(_json_bytes(compact_summary(results)))
    print(f"label={results['label']} selected={results['selection']['selected_arm']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
