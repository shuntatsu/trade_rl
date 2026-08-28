"""Dependency-injected concrete stage boundary for Causal Alpha V5."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
)


def execute_causal_alpha_v5_stage_callbacks(
    *,
    prepare_v4: Callable[[], object],
    fit_calibration: Callable[[object], object],
    build_selective_signal: Callable[[object, object], object],
    replay_and_select: Callable[[object, object, object], object],
    untouched_admission: Callable[[object, object, object, object], object],
) -> tuple[object, object, object, object, object]:
    """Expose the authored order without importing any learning or serving stack."""

    prepared = prepare_v4()
    calibration = fit_calibration(prepared)
    signal = build_selective_signal(prepared, calibration)
    if not bool(getattr(signal, "passed", False)):
        return prepared, calibration, signal, None, None
    selection = replay_and_select(prepared, calibration, signal)
    if not bool(getattr(selection, "passed", False)):
        return prepared, calibration, signal, selection, None
    admission = untouched_admission(prepared, calibration, signal, selection)
    return prepared, calibration, signal, selection, admission


def run_causal_alpha_v5_stage_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV5ResearchPackage:
    """Load the artifact-bound concrete assembler only for an actual run."""

    from trade_rl.workflows.universal_causal_alpha_v5_stage_entry import (
        run_causal_alpha_v5_concrete_entry,
    )

    return run_causal_alpha_v5_concrete_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


__all__ = ["execute_causal_alpha_v5_stage_callbacks", "run_causal_alpha_v5_stage_entry"]
