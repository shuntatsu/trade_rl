"""PostgreSQL orchestration for universal-instrument artifact publication."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from trade_rl.integrations.postgres_indicator_artifacts import (
    INDICATOR_CACHE_ID,
    IndicatorArtifactConnection,
)
from trade_rl.integrations.postgres_indicator_inventory import (
    load_postgres_indicator_source_inventory,
)
from trade_rl.integrations.postgres_market_tables import (
    LEGACY_MARKET_TABLES,
    PostgresMarketTableSet,
)
from trade_rl.workflows.universal_instrument_artifacts import (
    UniversalInstrumentArtifactPaths,
    materialize_universal_instrument_artifacts,
)


def materialize_postgres_universal_instrument_artifacts(
    connection: IndicatorArtifactConnection,
    *,
    output_dir: str | Path,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    seed: int,
    cache_id: str = INDICATOR_CACHE_ID,
    tables: PostgresMarketTableSet = LEGACY_MARKET_TABLES,
) -> UniversalInstrumentArtifactPaths:
    """Load metadata-only PostgreSQL evidence and publish one bound bundle."""

    source = load_postgres_indicator_source_inventory(
        connection,
        cache_id=cache_id,
        tables=tables,
    )
    return materialize_universal_instrument_artifacts(
        output_dir,
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
        seed=seed,
    )


__all__ = ["materialize_postgres_universal_instrument_artifacts"]
