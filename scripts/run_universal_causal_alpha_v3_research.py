"""Run artifact-bound real-data Causal Alpha V3 selection and admission."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.binance_universal_runtime import build_runtime
from trade_rl.workflows.universal_causal_alpha_v3_research import (
    run_causal_alpha_v3_research,
)
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run research-only Causal Alpha V3 selection and exact-once admission. "
            "This command never changes the reward or claims promotion eligibility."
        )
    )
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--frozen-metadata-root", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stage-limit", choices=("selection", "admission"), default="admission"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_config = TrainingRunConfig.from_json(args.run_config)
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
    )
    runtime = build_runtime(
        algorithm=FullResearchAlgorithm.PPO,
        run_config=run_config,
        context=context,
    )
    result = run_causal_alpha_v3_research(
        runtime=runtime,
        fold_train_range=context.manifest.fold_train_range,
        output_root=args.output_root,
        stage_limit=args.stage_limit,
    )
    print(
        json.dumps(
            {
                "package_digest": result.package_digest,
                "package_path": (
                    None if result.package_path is None else str(result.package_path)
                ),
                "phase": result.phase,
                "selection_digest": result.selection_digest,
                "selection_path": str(result.selection_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
