"""Strict immutable atomic evidence storage for Causal Alpha V7."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)


def _relative_path(value: str | Path) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("V7 relative artifact path is invalid")
    return path


class CausalAlphaV7RunLock(CausalAlphaV4RunLock):
    """V7 single-writer lock using the proven atomic ownership model."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v7.lock"


class CausalAlphaV7ArtifactStore(CausalAlphaV4ArtifactStore):
    """V7 store that forbids reuse of every published artifact leaf."""

    def write_leaf(
        self,
        relative_path: str | Path,
        payload: Mapping[str, object],
    ) -> Path:
        relative = _relative_path(relative_path)
        output = self.root / relative
        if output.exists():
            raise FileExistsError(f"V7 artifact already exists: {output}")
        return super().write_leaf(relative, payload)


__all__ = ["CausalAlphaV7ArtifactStore", "CausalAlphaV7RunLock"]
