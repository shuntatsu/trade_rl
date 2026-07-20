"""Facade over focused, validated Studio catalogs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trade_rl.studio.catalog_cache import CatalogCache
from trade_rl.studio.config_catalog import ConfigCatalog, ResolvedConfig
from trade_rl.studio.contracts import (
    ConfigSummary,
    DatasetSummary,
    JobSummary,
    RunSummary,
    StudioOverview,
)
from trade_rl.studio.dataset_catalog import DatasetCatalog, ResolvedDataset
from trade_rl.studio.overview import OverviewService
from trade_rl.studio.run_catalog import ResolvedRun, RunCatalog
from trade_rl.studio.settings import StudioSettings
from trade_rl.studio.system_probe import SystemProbe


class StudioCatalog:
    """Compatibility facade whose collaborators each own one catalog concern."""

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings
        cache: CatalogCache[object] = CatalogCache()
        self.datasets = DatasetCatalog(settings, cache)
        self.configs = ConfigCatalog(settings, cache)
        self.runs = RunCatalog(settings, cache)
        self.overviews = OverviewService(
            self.datasets,
            self.runs,
            SystemProbe(settings.project_root),
        )

    def list_datasets(self) -> tuple[DatasetSummary, ...]:
        return self.datasets.list()

    def resolve_dataset(self, resource_id: str) -> ResolvedDataset:
        return self.datasets.resolve(resource_id)

    def list_configs(self) -> tuple[ConfigSummary, ...]:
        return self.configs.list()

    def resolve_config(self, resource_id: str) -> ResolvedConfig:
        return self.configs.resolve(resource_id)

    def list_runs(self) -> tuple[RunSummary, ...]:
        return self.runs.list()

    def resolve_run(self, resource_id: str) -> ResolvedRun:
        return self.runs.resolve(resource_id)

    def resolve_run_for_evidence(self, resource_id: str) -> Path:
        return self.runs.resolve_for_evidence(resource_id)

    def overview(self, jobs: Sequence[JobSummary]) -> StudioOverview:
        return self.overviews.build(jobs)


__all__ = ["StudioCatalog"]
