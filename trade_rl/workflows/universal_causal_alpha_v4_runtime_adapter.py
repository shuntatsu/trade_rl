"""Reusable artifact-bound V4 runtime preparation without running a learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
    prepare_causal_alpha_v3_research_data,
)
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
    load_universal_runtime_factory,
)

_RUNTIME_FACTORY: Final = "trade_rl.workflows.binance_universal_runtime:build_runtime"


def prepare_causal_alpha_v4_runtime_adapter(
    *,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
) -> tuple[UniversalRuntimeFactoryContext, Any, Any]:
    """Load V4 data/context/environment adapters; no BC or policy training runs."""

    run_config = TrainingRunConfig.from_json(Path(run_config_path))
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=Path(runtime_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
        v4_context_manifest_path=Path(v4_context_manifest_path),
    )
    runtime = load_universal_runtime_factory(_RUNTIME_FACTORY)(
        algorithm="ppo",
        run_config=run_config,
        context=context,
    )
    prepared_v3 = prepare_causal_alpha_v3_research_data(
        runtime=runtime,
        fold_train_range=context.manifest.fold_train_range,
    )
    return context, runtime, prepared_v3


__all__ = ["prepare_causal_alpha_v4_runtime_adapter"]
