"""Closed training-mode identities used by configuration and runtime evidence."""

from __future__ import annotations

from enum import StrEnum


class ObservationEncoder(StrEnum):
    """Maintained observation encoders."""

    FLAT_MLP = "flat_mlp"
    ASSET_SET = "asset_set"
    HIERARCHICAL_SEQUENCE_V2 = "hierarchical_sequence_v2"


class CudaRuntimeMode(StrEnum):
    """Explicit CUDA speed/reproducibility contract."""

    DETERMINISTIC = "deterministic"
    PERFORMANCE = "performance"


__all__ = ["CudaRuntimeMode", "ObservationEncoder"]
