"""Immutable atomic evidence storage for Causal Alpha V5."""

from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)

CAUSAL_ALPHA_V5_STORE_SCHEMA = "causal_alpha_v5_artifact_store_v1"


class CausalAlphaV5RunLock(CausalAlphaV4RunLock):
    """V5-named single-writer lock with the proven V4 atomic semantics."""


class CausalAlphaV5ArtifactStore(CausalAlphaV4ArtifactStore):
    """V5-named store reusing the immutable V4 leaf implementation."""


__all__ = [
    "CAUSAL_ALPHA_V5_STORE_SCHEMA",
    "CausalAlphaV5ArtifactStore",
    "CausalAlphaV5RunLock",
]
