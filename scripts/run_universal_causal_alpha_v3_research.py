from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_runner import (
    CausalAlphaV3AdmissionRejected,
    CausalAlphaV3SignalRejected,
    prepare_causal_alpha_v3_research_data,
    run_universal_causal_alpha_v3_research,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
)
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
    load_universal_runtime_factory,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the research-only artifact-bound Causal Alpha V3 path through "
            "signal gating, deterministic production replay, and untouched teacher "
            "admission. This command never claims production readiness."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument(
        "--runtime-factory",
        default="trade_rl.workflows.binance_universal_runtime:build_runtime",
        help=(
            "Import target module:function. The callable receives algorithm, "
            "run_config, and context."
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
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    research_config = CausalAlphaV3ResearchConfig.from_json(args.config)
    run_config = TrainingRunConfig.from_json(args.run_config)
    if (args.fold_train_start is None) != (args.fold_train_stop is None):
        raise ValueError(
            "fold-train-start and fold-train-stop must be provided together"
        )
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
        instrument_artifact_root=args.instrument_artifact_root,
        dataset_artifact_root=args.dataset_artifact_root,
        fold_train_range=(
            (args.fold_train_start, args.fold_train_stop)
            if args.fold_train_start is not None
            else None
        ),
        normalizer_digest=args.normalizer_digest,
        feature_schema_digest=args.feature_schema_digest,
    )
    runtime_factory = load_universal_runtime_factory(args.runtime_factory)
    runtime = runtime_factory(
        algorithm=FullResearchAlgorithm.PPO,
        run_config=run_config,
        context=context,
    )
    prepared = prepare_causal_alpha_v3_research_data(
        runtime=runtime,
        fold_train_range=context.manifest.fold_train_range,
    )
    try:
        package = run_universal_causal_alpha_v3_research(
            config=research_config,
            prepared=prepared,
            output_root=args.output_root,
        )
    except CausalAlphaV3SignalRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "signal_rejected",
            }
        )
        return 2
    except CausalAlphaV3SelectionRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "selection_rejected",
            }
        )
        return 3
    except CausalAlphaV3AdmissionRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "admission_rejected",
            }
        )
        return 4

    _emit(
        {
            "package_digest": package.digest,
            "promotion_eligible": package.promotion_eligible,
            "research_only": package.research_only,
            "selected_candidate_digest": package.selected_candidate_digest,
            "status": "admitted",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
