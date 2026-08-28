"""Strict immutable atomic evidence storage for Causal Alpha V6."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)

CAUSAL_ALPHA_V6_STORE_SCHEMA: Final = "causal_alpha_v6_artifact_store_v1"


def _relative_path(value: str | Path) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("V6 relative artifact path is invalid")
    return path


class CausalAlphaV6RunLock(CausalAlphaV4RunLock):
    """V6-named single-writer lock with the proven atomic ownership model."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v6.lock"


class CausalAlphaV6ArtifactStore(CausalAlphaV4ArtifactStore):
    """V6 store that forbids reuse of every previously published leaf."""

    def write_leaf(
        self, relative_path: str | Path, payload: Mapping[str, object]
    ) -> Path:
        relative = _relative_path(relative_path)
        output = self.root / relative
        if output.exists():
            raise FileExistsError(f"V6 artifact already exists: {output}")
        return super().write_leaf(relative, payload)


__all__ = [
    "CAUSAL_ALPHA_V6_STORE_SCHEMA",
    "CausalAlphaV6ArtifactStore",
    "CausalAlphaV6RunLock",
]
