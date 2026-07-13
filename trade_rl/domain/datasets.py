"""Dataset identity and schema records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
    require_unique_non_empty,
)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable identity for one fully resolved training/evaluation dataset."""

    dataset_id: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    base_timeframe: str
    created_at: datetime
    schema_version: str = "dataset_manifest_v1"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_unique_non_empty(self.symbols, field="symbols")
        require_unique_non_empty(self.feature_names, field="feature_names")
        require_non_empty(self.base_timeframe, field="base_timeframe")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")
