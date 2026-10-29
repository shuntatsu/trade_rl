"""Command line entry for the sealed FS1 evaluator (refuses unless every seal holds).

Nothing is tunable here: the parameters are the frozen registry, and every value the
operator supplies is compared with the registry, the checked-out commit or the dataset
seal manifest. Exit codes: 0 scored and written, 2 refused (nothing was scored and no
dataset was read unless all earlier preconditions held).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from trade_rl.evaluation.forward_study.fs1 import run_fs1
from trade_rl.evaluation.forward_study.seal import Fs1Refusal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.forward_study",
        description=(
            "Score the sealed FS1 forward study ONCE. Refuses before the scoring end "
            "date, without an Issue #810 authorization comment URL, or on any params, "
            "code or dataset mismatch."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--params-digest", required=True)
    parser.add_argument("--sealed-code-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the operator's values and call the sealed entrypoint (no seams here)."""

    args = build_parser().parse_args(argv)
    try:
        result = run_fs1(
            authorization=args.authorization,
            expected_params_digest=args.params_digest,
            sealed_code_sha=args.sealed_code_sha,
            manifest_path=args.manifest,
            expected_manifest_sha256=args.manifest_sha256,
            dataset_path=args.dataset,
            output_path=args.output,
            progress=lambda message: print(message, flush=True),
        )
    except Fs1Refusal as refusal:
        print(f"refused: {', '.join(refusal.reasons)}", file=sys.stderr)
        return 2
    labels = {arm: row["label"] for arm, row in result["labels"].items()}
    print(f"labels={labels}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
