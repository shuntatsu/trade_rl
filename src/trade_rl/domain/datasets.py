"""Dataset identity and schema records."""

from __future__ import annotations

import math
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
    bar_hours: float = 1.0
    schema_version: str = "dataset_manifest_v2"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_unique_non_empty(self.symbols, field="symbols")
        require_unique_non_empty(self.feature_names, field="feature_names")
        require_non_empty(self.base_timeframe, field="base_timeframe")
        require_aware_datetime(self.created_at, field="created_at")
        if not math.isfinite(self.bar_hours) or self.bar_hours <= 0.0:
            raise ValueError("bar_hours must be finite and positive")
        require_non_empty(self.schema_version, field="schema_version")
