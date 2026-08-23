"""Artifact-bound concrete entry for the research-only Causal Alpha V4 stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v4_pipeline import (
    CausalAlphaV4ResearchPackage,
    run_universal_causal_alpha_v4_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_execution import (
    run_causal_alpha_v4_admission_stage,
    run_causal_alpha_v4_selection_stage,
    run_causal_alpha_v4_signal_stage,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    prepare_causal_alpha_v4_stage_data,
    slice_causal_alpha_v4_forecast,
)

_V4_RUNTIME_FACTORY: Final = "trade_rl.workflows.binance_universal_runtime:build_runtime"


def _artifact(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "artifact_digest": content_digest(payload)}


def _run_manifest_payload(prepared: Any) -> dict[str, object]:
    return _artifact(
        {
            "schema_version": "causal_alpha_v4_run_manifest_v1",
            "run_manifest_digest": prepared.run_manifest_digest,
            "base_runtime_manifest_digest": prepared.base_runtime_manifest_digest,
            "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
            "config_digest": prepared.config_digest,
            "execution_identity_digest": prepared.execution_identity_digest,
            "nested_partition_digest": prepared.nested_partition_digest,
            "generator_code_digest": prepared.generator_code_digest,
            "train_symbols": prepared.train_symbols,
            "promotion_eligible": False,
            "research_only": True,
        }
    )


def _authored_config_payload(
    *,
    config: Any,
    config_path: Path,
    prepared: Any,
) -> dict[str, object]:
    try:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V4 authored config cannot be re-read") from error
    return _artifact(
        {
            "schema_version": "causal_alpha_v4_authored_config_record_v1",
            "run_manifest_digest": prepared.run_manifest_digest,
            "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
            "config_digest": config.digest,
            "generator_code_digest": prepared.generator_code_digest,
            "source_config": source,
            "promotion_eligible": False,
            "research_only": True,
        }
    )


def run_causal_alpha_v4_stage_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV4ResearchPackage:
    """Resolve immutable runtime/context identity and execute V4 gates in order."""

    # Keep the import-only CLI surface usable without the heavyweight training
    # extras. These dependencies are required only when the concrete research run
    # is actually executed.
    from trade_rl.rl.training_run_config import TrainingRunConfig
    from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
        prepare_causal_alpha_v3_research_data,
    )
    from trade_rl.workflows.universal_causal_alpha_v4_runner import (
        CausalAlphaV4ResearchConfig,
    )
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
        load_universal_runtime_factory,
    )
    from trade_rl.workflows.universal_research import FullResearchAlgorithm

    config_path = Path(config_path)
    config = CausalAlphaV4ResearchConfig.from_json(config_path)
    run_config = TrainingRunConfig.from_json(Path(run_config_path))
    runtime_context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=Path(runtime_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
        v4_context_manifest_path=Path(v4_context_manifest_path),
    )
    runtime_factory = load_universal_runtime_factory(_V4_RUNTIME_FACTORY)
    runtime = runtime_factory(
        algorithm=FullResearchAlgorithm.PPO,
        run_config=run_config,
        context=runtime_context,
    )
    prepared_v3 = prepare_causal_alpha_v3_research_data(
        runtime=runtime,
        fold_train_range=runtime_context.manifest.fold_train_range,
    )
    generator_code_digest = content_digest(
        {
            "schema_version": "causal_alpha_v4_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config.digest,
        generator_code_digest=generator_code_digest,
        runtime_context=runtime_context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    store = CausalAlphaV4ArtifactStore(
        Path(output_root),
        run_manifest_digest=prepared.run_manifest_digest,
        v4_context_manifest_digest=prepared.v4_context_manifest_digest,
        config_digest=prepared.config_digest,
        generator_code_digest=prepared.generator_code_digest,
    )
    store.write_leaf("run-manifest.json", _run_manifest_payload(prepared))
    store.write_leaf(
        "authored-config.json",
        _authored_config_payload(
            config=config,
            config_path=config_path,
            prepared=prepared,
        ),
    )
    return run_universal_causal_alpha_v4_research_pipeline(
        store=store,
        prepare_stage=lambda: prepared,
        signal_stage=lambda value: run_causal_alpha_v4_signal_stage(
            value,
            config=config,
            store=store,
            slice_forecast=slice_causal_alpha_v4_forecast,
        ),
        selection_stage=lambda value, signal: run_causal_alpha_v4_selection_stage(
            value,
            signal,
            config=config,
            store=store,
            slice_forecast=slice_causal_alpha_v4_forecast,
        ),
        admission_stage=lambda value, signal, selection: run_causal_alpha_v4_admission_stage(
            value,
            signal,
            selection,
            config=config,
            store=store,
            slice_forecast=slice_causal_alpha_v4_forecast,
        ),
    )


__all__ = ["run_causal_alpha_v4_stage_entry"]
