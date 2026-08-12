"""Evaluate random, BC, critic-warm, and rollout policies on one market segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_rl.workflows.universal_policy_stage_evaluation import (
    evaluate_universal_policy_stages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--frozen-metadata-root", type=Path, required=True)
    parser.add_argument("--member-root", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=("ppo", "lagrangian", "discounted"), required=True
    )
    parser.add_argument("--member-seed", type=int, required=True)
    parser.add_argument("--decisions", type=int, default=384)
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_universal_policy_stages(
        config_path=args.config,
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
        member_root=args.member_root,
        algorithm=args.algorithm,
        member_seed=args.member_seed,
        decisions=args.decisions,
        output_path=args.output,
        symbols=args.symbols,
        device=args.device,
    )
    print(json.dumps(result["stages"], sort_keys=True))


if __name__ == "__main__":
    main()
