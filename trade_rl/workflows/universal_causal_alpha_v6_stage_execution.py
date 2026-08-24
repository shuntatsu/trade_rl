"""Dependency-injected concrete stage boundary for Causal Alpha V6."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v6_pipeline import (
    CausalAlphaV6ResearchPackage,
)


def execute_causal_alpha_v6_stage_callbacks(
    *,
    prepare_v4: Callable[[], object],
    build_signal: Callable[[object], object],
    replay_and_select: Callable[[object, object], object],
    untouched_admission: Callable[[object, object, object], object],
) -> tuple[object, object, object | None, object | None]:
    """Expose the authored stage order without importing the learning stack."""

    prepared = prepare_v4()
    signal = build_signal(prepared)
    if not bool(getattr(signal, "passed", False)):
        return prepared, signal, None, None
    selection = replay_and_select(prepared, signal)
    if not bool(getattr(selection, "passed", False)):
        return prepared, signal, selection, None
    admission = untouched_admission(prepared, signal, selection)
    return prepared, signal, selection, admission


def run_causal_alpha_v6_stage_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV6ResearchPackage:
    """Load the artifact-bound concrete assembler only for an actual run."""

    from trade_rl.workflows.universal_causal_alpha_v6_stage_entry import (
        run_causal_alpha_v6_concrete_entry,
    )

    return run_causal_alpha_v6_concrete_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


__all__ = [
    "execute_causal_alpha_v6_stage_callbacks",
    "run_causal_alpha_v6_stage_entry",
]
