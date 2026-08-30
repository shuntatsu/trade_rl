"""CLI for one independent Causal Alpha V11 policy study arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_rl.learning.causal_alpha_v11 import CausalAlphaV11StudyArm
from trade_rl.workflows.universal_causal_alpha_v11_stage_entry import (
    run_causal_alpha_v11_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Causal Alpha V11 research")
    for name in (
        "config",
        "run-config",
        "runtime-manifest",
        "v4-context-manifest",
        "frozen-metadata-root",
        "r21-output-root",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument(
        "--study-arm",
        required=True,
        choices=tuple(arm.value for arm in CausalAlphaV11StudyArm),
    )
    args = parser.parse_args()
    try:
        evidence = run_causal_alpha_v11_selection(
            config_path=args.config,
            run_config_path=args.run_config,
            runtime_manifest_path=args.runtime_manifest,
            v4_context_manifest_path=args.v4_context_manifest,
            frozen_metadata_root=args.frozen_metadata_root,
            r21_output_root=args.r21_output_root,
            output_root=args.output_root,
            study_arm=CausalAlphaV11StudyArm(args.study_arm),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"error": f"{type(error).__name__}:{error}", "status": "failed"},
                sort_keys=True,
            )
        )
        return 5
    stopped = evidence.source_v8 is None
    status = (
        "preflight_stopped"
        if stopped
        else "selection_passed"
        if evidence.passed
        else "selection_rejected"
    )
    print(
        json.dumps(
            {
                "artifact_digest": evidence.digest,
                "promotion_eligible": False,
                "status": status,
                "study_arm": evidence.study_arm.value,
            },
            sort_keys=True,
        )
    )
    return 4 if stopped else 0 if evidence.passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
