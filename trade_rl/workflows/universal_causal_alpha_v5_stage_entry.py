"""Artifact-bound entry seam for the Causal Alpha V5 concrete runtime."""

from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
)


def run_causal_alpha_v5_concrete_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV5ResearchPackage:
    """Concrete entry implemented after architecture gates close the import surface."""

    del config_path, run_config_path, runtime_manifest_path
    del v4_context_manifest_path, frozen_metadata_root, output_root
    raise RuntimeError("Causal Alpha V5 concrete stage assembly is not yet available")


__all__ = ["run_causal_alpha_v5_concrete_entry"]
