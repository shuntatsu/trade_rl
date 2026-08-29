"""CLI for Causal Alpha V10 hierarchical-wave research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_rl.learning.causal_alpha_v10_hierarchy import CausalAlphaV10BoundaryMode
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    run_causal_alpha_v10_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Causal Alpha V10 research")
    for name in (
        "config",
        "run-config",
        "runtime-manifest",
        "v4-context-manifest",
        "frozen-metadata-root",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument(
        "--boundary-mode",
        choices=tuple(mode.value for mode in CausalAlphaV10BoundaryMode),
        default=CausalAlphaV10BoundaryMode.INHERIT_CONFIRM.value,
    )
    args = parser.parse_args()
    try:
        evidence = run_causal_alpha_v10_selection(
            config_path=args.config,
            run_config_path=args.run_config,
            runtime_manifest_path=args.runtime_manifest,
            v4_context_manifest_path=args.v4_context_manifest,
            frozen_metadata_root=args.frozen_metadata_root,
            output_root=args.output_root,
            boundary_mode=CausalAlphaV10BoundaryMode(args.boundary_mode),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"error": f"{type(error).__name__}:{error}", "status": "failed"},
                sort_keys=True,
            )
        )
        return 5
    print(
        json.dumps(
            {
                "artifact_digest": evidence.digest,
                "promotion_eligible": False,
                "status": "selection_passed" if evidence.passed else "selection_rejected",
            },
            sort_keys=True,
        )
    )
    return 0 if evidence.passed else 3


if __name__ == "__main__":
    raise SystemExit(main())

