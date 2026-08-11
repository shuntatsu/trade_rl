from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
    load_universal_runtime_factory,
    run_universal_full_research_training,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime


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
        default="trade_rl.integrations.binance_universal_runtime:build_runtime",
        help=(
            "Import target module:function. The callable receives keyword arguments "
            "algorithm, run_config, and context and must return UniversalTrainingRuntime. "
            "Defaults to trade_rl.integrations.binance_universal_runtime:build_runtime."
        ),
    )
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--frozen-metadata-root", required=True, type=Path)
    parser.add_argument("--instrument-artifact-root", type=Path)
    parser.add_argument("--dataset-artifact-root", type=Path)
    parser.add_argument("--fold-train-start", type=int)
    parser.add_argument("--fold-train-stop", type=int)
    parser.add_argument("--normalizer-digest")
    parser.add_argument("--feature-schema-digest")
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--fold", action="append", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    authored_configs: Mapping[FullResearchAlgorithm | str, TrainingRunConfig] = {
        FullResearchAlgorithm.PPO: TrainingRunConfig.from_json(args.ppo_config),
        FullResearchAlgorithm.LAGRANGIAN: TrainingRunConfig.from_json(
            args.lagrangian_config
        ),
        FullResearchAlgorithm.DISCOUNTED: TrainingRunConfig.from_json(
            args.discounted_config
        ),
    }
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
        instrument_artifact_root=args.instrument_artifact_root,
        dataset_artifact_root=args.dataset_artifact_root,
        fold_train_range=(
            (args.fold_train_start, args.fold_train_stop)
            if args.fold_train_start is not None and args.fold_train_stop is not None
            else None
        ),
        normalizer_digest=args.normalizer_digest,
        feature_schema_digest=args.feature_schema_digest,
    )
    raw_runtime_factory = load_universal_runtime_factory(args.runtime_factory)

    def runtime_factory(
        *,
        algorithm: FullResearchAlgorithm,
        run_config: TrainingRunConfig,
    ) -> UniversalTrainingRuntime:
        return raw_runtime_factory(
            algorithm=algorithm,
            run_config=run_config,
            context=context,
        )

    result = run_universal_full_research_training(
        selected_architecture=UniversalArchitectureName(args.selected_architecture),
        run_configs=authored_configs,
        runtime_factory=runtime_factory,
        fold_train_range=context.manifest.fold_train_range,
        normalizer_digest=context.manifest.statistics_digest,
        feature_schema_digest=context.manifest.feature_schema_digest,
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
