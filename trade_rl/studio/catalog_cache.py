"""Small in-process cache for fully validated immutable artifacts."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class CatalogCache(Generic[T]):
    def __init__(self) -> None:
        self._entries: dict[tuple[str, Path], tuple[Hashable, T]] = {}

    def get(
        self,
        namespace: str,
        path: Path,
        fingerprint: Hashable,
        loader: Callable[[], T],
    ) -> T:
        key = (namespace, path.resolve())
        cached = self._entries.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        value = loader()
        self._entries[key] = (fingerprint, value)
        return value

    def clear(self) -> None:
        self._entries.clear()
