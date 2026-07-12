"""Run the production control-plane training and candidate-registration pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

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
    parser.add_argument(
        "--risk-config",
        type=Path,
        default=None,
        help=(
            "validated JSON risk policy required whenever a release candidate may "
            "be registered"
        ),
    )
    args = parser.parse_args()
    release_disqualifying_override = any(
        (args.force, args.skip_p0, args.skip_wf, args.skip_gate)
    )
    release_intent = not args.no_register and not release_disqualifying_override
    if release_intent and args.risk_config is None:
        parser.error("--risk-config is required unless the run is research-only")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
