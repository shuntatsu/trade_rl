"""Validated dataset discovery and collision-free Studio identities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.data import load_market_dataset_artifact
from trade_rl.studio.catalog_cache import CatalogCache
from trade_rl.studio.catalog_common import fingerprint_identity, mtime, stat_fingerprint
from trade_rl.studio.contracts import DatasetSummary
from trade_rl.studio.errors import ArtifactInvalid, ResourceNotFound
from trade_rl.studio.resource_ids import require_resource_id, resource_id
from trade_rl.studio.settings import StudioSettings


def _iso_timestamp(value: np.datetime64) -> str:
    nanoseconds = int(value.astype("datetime64[ns]").astype(np.int64))
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, UTC).isoformat()


def _timeframe(bar_hours: float) -> str:
    if bar_hours < 1.0:
        return f"{int(round(bar_hours * 60.0))}m"
    if math.isclose(bar_hours % 24.0, 0.0):
        return f"{int(round(bar_hours / 24.0))}d"
    return f"{bar_hours:g}h"


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    path: Path
    summary: DatasetSummary


class DatasetCatalog:
    def __init__(self, settings: StudioSettings, cache: CatalogCache[object]) -> None:
        self.settings = settings
        self.cache = cache

    def _directories(self) -> tuple[Path, ...]:
        directories: set[Path] = set()
        for root in self.settings.dataset_roots:
            if (root / "manifest.json").is_file() and (root / "arrays.npz").is_file():
                directories.add(root)
            if root.is_dir():
                for manifest in root.rglob("manifest.json"):
                    if (manifest.parent / "arrays.npz").is_file():
                        directories.add(manifest.parent)
        return tuple(sorted(directories, key=lambda item: item.as_posix()))

    def _fingerprint(self, path: Path) -> tuple[object, ...]:
        return (
            stat_fingerprint(path / "manifest.json"),
            stat_fingerprint(path / "arrays.npz"),
        )

    def _load(self, path: Path) -> DatasetSummary:
        relative = self.settings.relative_path(path)
        fingerprint = self._fingerprint(path)

        def build() -> DatasetSummary:
            try:
                dataset = load_market_dataset_artifact(path)
                start = _iso_timestamp(dataset.timestamps[0])
                end = _iso_timestamp(dataset.timestamps[-1])
                return DatasetSummary(
                    id=resource_id("dataset", relative, dataset.dataset_id),
                    dataset_id=dataset.dataset_id,
                    name=path.name,
                    relative_path=relative,
                    market=dataset.calendar_kind,
                    symbols=dataset.symbols,
                    timeframes=(_timeframe(dataset.bar_hours),),
                    range=f"{start} — {end}",
                    status="VALID",
                    feature_count=dataset.n_features,
                    bar_count=dataset.n_bars,
                    symbol_count=dataset.n_symbols,
                    updated=mtime(path / "manifest.json"),
                )
            except (OSError, ValueError, TypeError) as error:
                identity = fingerprint_identity((relative, fingerprint))
                return DatasetSummary(
                    id=resource_id("dataset", relative, identity),
                    dataset_id="",
                    name=path.name,
                    relative_path=relative,
                    market="unknown",
                    symbols=(),
                    timeframes=(),
                    range="—",
                    status="INVALID",
                    feature_count=0,
                    bar_count=0,
                    symbol_count=0,
                    updated=mtime(path / "manifest.json"),
                    validation_error=str(error),
                )

        return cast(DatasetSummary, self.cache.get("dataset", path, fingerprint, build))

    def list(self) -> tuple[DatasetSummary, ...]:
        records = tuple(self._load(path) for path in self._directories())
        return tuple(sorted(records, key=lambda item: item.updated, reverse=True))

    def resolve(self, value: str) -> ResolvedDataset:
        try:
            require_resource_id(value, kind="dataset")
        except ValueError as error:
            raise ResourceNotFound(
                f"unknown Studio dataset resource: {value}"
            ) from error
        for summary in self.list():
            if summary.id != value:
                continue
            if summary.status != "VALID":
                raise ArtifactInvalid(
                    summary.validation_error or "dataset artifact is invalid"
                )
            return ResolvedDataset(
                path=self.settings.project_root / summary.relative_path,
                summary=summary,
            )
        raise ResourceNotFound(f"unknown Studio dataset resource: {value}")


__all__ = ["DatasetCatalog", "ResolvedDataset"]
