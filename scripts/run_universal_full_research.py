from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
    load_universal_runtime_factory,
    run_universal_full_research_training,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Universal U6 PPO/Lagrangian/discounted training for one U5-selected "
            "architecture. This command does not claim research success until paired "
            "sealed evidence is completed."
        )
    )
    parser.add_argument(
        "--selected-architecture",
        required=True,
        choices=tuple(item.value for item in UniversalArchitectureName),
    )
    parser.add_argument("--ppo-config", required=True, type=Path)
    parser.add_argument("--lagrangian-config", required=True, type=Path)
    parser.add_argument("--discounted-config", required=True, type=Path)
    parser.add_argument(
        "--runtime-factory",
        required=True,
        help=(
            "Import target module:function. The callable receives keyword arguments "
            "algorithm, run_config, and context and must return UniversalTrainingRuntime."
        ),
    )
    parser.add_argument("--instrument-artifact-root", required=True, type=Path)
    parser.add_argument("--postgres-url", required=True)
    parser.add_argument("--dataset-artifact-root", required=True, type=Path)
    parser.add_argument("--fold-train-start", required=True, type=int)
    parser.add_argument("--fold-train-stop", required=True, type=int)
    parser.add_argument("--normalizer-digest", required=True)
    parser.add_argument("--feature-schema-digest", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--fold", action="append", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    authored_configs = {
        FullResearchAlgorithm.PPO: TrainingRunConfig.from_json(args.ppo_config),
        FullResearchAlgorithm.LAGRANGIAN: TrainingRunConfig.from_json(
            args.lagrangian_config
        ),
        FullResearchAlgorithm.DISCOUNTED: TrainingRunConfig.from_json(
            args.discounted_config
        ),
    }
    context = UniversalRuntimeFactoryContext(
        instrument_artifact_root=args.instrument_artifact_root,
        postgres_url=args.postgres_url,
        dataset_artifact_root=args.dataset_artifact_root,
        fold_train_range=(args.fold_train_start, args.fold_train_stop),
        normalizer_digest=args.normalizer_digest,
        feature_schema_digest=args.feature_schema_digest,
    )
    raw_runtime_factory = load_universal_runtime_factory(args.runtime_factory)

    def runtime_factory(*, algorithm, run_config):
        return raw_runtime_factory(
            algorithm=algorithm,
            run_config=run_config,
            context=context,
        )

    result = run_universal_full_research_training(
        selected_architecture=UniversalArchitectureName(args.selected_architecture),
        run_configs=authored_configs,
        runtime_factory=runtime_factory,
        fold_train_range=context.fold_train_range,
        normalizer_digest=context.normalizer_digest,
        feature_schema_digest=context.feature_schema_digest,
        baseline_names=tuple(args.baseline),
        folds=tuple(args.fold),
        output_root=args.output_root,
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "comparison_digest": result.comparison_digest,
                "manifest_digest": result.manifest_digest,
                "manifest_path": str(result.manifest_path),
                "research_success": result.research_success,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
