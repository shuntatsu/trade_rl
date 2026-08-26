"""Lazy concrete stage boundary for Causal Alpha V7."""

from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v7_pipeline import (
    CausalAlphaV7ResearchPackage,
)


def run_causal_alpha_v7_stage_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV7ResearchPackage:
    from trade_rl.workflows.universal_causal_alpha_v7_stage_entry import (
        run_causal_alpha_v7_concrete_entry,
    )

    return run_causal_alpha_v7_concrete_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


__all__ = ["run_causal_alpha_v7_stage_entry"]
