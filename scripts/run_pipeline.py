"""Run the production control-plane training and candidate-registration pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from mars_lite.eval.residual_walk_forward import run_residual_walk_forward
from mars_lite.pipeline.cli import build_parser
from mars_lite.pipeline.production_pipeline import run
from mars_lite.pipeline.residual_pipeline import run_baseline_residual
from mars_lite.pipeline.residual_release_boundary import validate_residual_invocation


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
    parser.add_argument(
        "--action-mode",
        choices=["direct", "baseline-residual"],
        default="direct",
        help=(
            "direct=legacy per-symbol action; baseline-residual=two-dimensional "
            "trend/alpha residual action"
        ),
    )
    parser.add_argument(
        "--run-tier",
        choices=["smoke", "research", "release"],
        default="research",
        help="minimum PPO update and seed contract for baseline-residual runs",
    )
    parser.add_argument("--baseline-max-drawdown", type=float, default=0.30)
    args = parser.parse_args()

    validate_residual_invocation(
        action_mode=args.action_mode,
        no_register=bool(args.no_register),
    )
    if args.action_mode == "baseline-residual":
        if args.phase == "wf":
            run_residual_walk_forward(args, Path(args.output))
        else:
            run_baseline_residual(args, Path(args.output))
        return 0

    release_disqualifying_override = any(
        (args.force, args.skip_p0, args.skip_wf, args.skip_gate)
    )
    release_intent = not args.no_register and not release_disqualifying_override
    if release_intent and args.risk_config is None:
        parser.error("--risk-config is required unless the run is research-only")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
