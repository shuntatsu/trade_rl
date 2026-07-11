"""Run the production control-plane training and candidate-registration pipeline."""

from __future__ import annotations

import sys

from mars_lite.pipeline.cli import build_parser
from mars_lite.pipeline.production_pipeline import run


def main() -> int:
    parser = build_parser()
    parser.add_argument("--skip-p0", action="store_true")
    parser.add_argument("--skip-pbt", action="store_true")
    parser.add_argument("--skip-wf", action="store_true")
    parser.add_argument("--wf-cost-gate", type=float, default=0.0)
    parser.add_argument("--require-significant", action="store_true")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="run validation/training without constructing and registering a candidate",
    )
    parser.add_argument("--registry-dir", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
