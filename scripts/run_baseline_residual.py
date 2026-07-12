"""Run baseline-anchored residual RL as an explicit research workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from mars_lite.pipeline.cli import build_parser
from mars_lite.pipeline.residual_pipeline import run_baseline_residual


def main() -> int:
    parser = build_parser()
    parser.description = "Baseline-Anchored Residual RL training"
    parser.set_defaults(
        phase="train",
        target="cs_demean",
        min_trade_delta=0.0,
        lambda_turnover=0.0,
        ensemble=3,
    )
    parser.add_argument(
        "--run-tier",
        choices=["smoke", "research", "release"],
        default="research",
    )
    parser.add_argument(
        "--baseline-max-drawdown",
        type=float,
        default=0.30,
    )
    args = parser.parse_args()
    run_baseline_residual(args, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
