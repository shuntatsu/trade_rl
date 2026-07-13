"""Canonical market-dataset artifacts and range-scoped immutable views."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifact_codec import (
    DATASET_ARRAYS_NAME,
    DATASET_ARTIFACT_SCHEMA,
    DATASET_MANIFEST_NAME,
    load_dataset_files,
    write_dataset_files,
)
from trade_rl.data.market import MarketDataset

DATASET_VIEW_SCHEMA: Final = "market_dataset_view_v1"


def write_market_dataset_artifact(root: Path, dataset: MarketDataset) -> Path:
    """Write the canonical deterministic artifact into ``root``."""

    return write_dataset_files(Path(root), dataset)[0]


def load_market_dataset_artifact(root: Path) -> MarketDataset:
    return load_dataset_files(Path(root))


@dataclass(frozen=True, slots=True)
class MarketDatasetView:
    """A half-open, absolute-index view that cannot escape its assigned range."""

    dataset: MarketDataset
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or not 0 <= self.start < self.stop <= self.dataset.n_bars
        ):
            raise ValueError("dataset view range is outside the dataset")

    @property
    def identity(self) -> str:
        return content_digest(
            {
                "dataset_id": self.dataset.dataset_id,
                "schema_version": DATASET_VIEW_SCHEMA,
                "start": self.start,
                "stop": self.stop,
            }
        )

    def subview(self, start: int, stop: int) -> MarketDatasetView:
        if not self.start <= start < stop <= self.stop:
            raise ValueError("requested subview is outside the parent range")
        return MarketDatasetView(self.dataset, start, stop)

    def materialize(self) -> MarketDataset:
        if self.stop - self.start < 3:
            raise ValueError("materialized dataset view requires at least three bars")
        kwargs: dict[str, Any] = {}
        for item in fields(MarketDataset):
            if not item.init or item.name.startswith("_"):
                continue
            value = getattr(self.dataset, item.name)
            if item.name == "dataset_id":
                kwargs[item.name] = self.identity
            elif item.name == "identity_payload_json":
                # A range view has a different identity from the source artifact.
                kwargs[item.name] = None
            elif isinstance(value, np.ndarray) and value.shape[:1] == (
                self.dataset.n_bars,
            ):
                kwargs[item.name] = value[self.start : self.stop]
            else:
                kwargs[item.name] = value
        return MarketDataset(**kwargs)


__all__ = [
    "DATASET_ARRAYS_NAME",
    "DATASET_ARTIFACT_SCHEMA",
    "DATASET_MANIFEST_NAME",
    "DATASET_VIEW_SCHEMA",
    "MarketDatasetView",
    "load_market_dataset_artifact",
    "write_market_dataset_artifact",
]
