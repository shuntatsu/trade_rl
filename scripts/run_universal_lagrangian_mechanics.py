"""Run a real-data short-episode smoke that must exercise Lagrangian updates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.binance_universal_runtime import build_runtime
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
)
from trade_rl.workflows.universal_lagrangian_mechanics import (
    build_lagrangian_mechanics_config,
    verify_lagrangian_mechanics_model,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_training_runner import (
    assemble_universal_sb3_training_backend,
    train_universal_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--frozen-metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-hours", type=float, default=8.0)
    parser.add_argument("--timesteps", type=int, default=1_024)
    args = parser.parse_args()

    base = TrainingRunConfig.from_json(args.config)
    config = build_lagrangian_mechanics_config(
        base,
        episode_hours=args.episode_hours,
        timesteps=args.timesteps,
    )
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
    )
    runtime = build_runtime(
        algorithm=FullResearchAlgorithm.LAGRANGIAN,
        run_config=config,
        context=context,
    )
    backend, bundle = assemble_universal_sb3_training_backend(
        routed_environment_factory=runtime.routed_environment_factory,
        training=config.training,
        fold_train_range=context.manifest.fold_train_range,
        normalizer_digest=context.manifest.statistics_digest,
        feature_schema_digest=context.manifest.feature_schema_digest,
    )
    bound_runtime = runtime.with_pretraining_artifact(
        bundle.teacher_artifact.artifact_digest
    )
    training_manifest = train_universal_seeds(
        runtime=bound_runtime,
        training=config.training,
        backend=backend,
        output_root=args.output_root,
        architecture_name="u_medium_direct",
    )
    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

    model = LagrangianPPO.load(
        str(args.output_root / f"seed-{config.training.seeds[0]}" / "policy.zip"),
        device="cpu",
    )
    mechanics_evidence = verify_lagrangian_mechanics_model(model)
    payload: dict[str, object] = {
        "schema_version": "universal_lagrangian_mechanics_v1",
        "episode_hours": config.environment.episode_hours,
        "timesteps": config.training.timesteps,
        "reward_config_digest": content_digest(asdict(config.reward)),
        "training_config_digest": content_digest(config.training.digest_payload()),
        "runtime_manifest_digest": context.manifest.manifest_digest,
        "mechanics_evidence": mechanics_evidence,
        "training_manifest": training_manifest,
    }
    result = {**payload, "artifact_digest": content_digest(payload)}
    atomic_write_bytes(
        args.output_root / "lagrangian-mechanics.json",
        canonical_json_bytes(result) + b"\n",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
