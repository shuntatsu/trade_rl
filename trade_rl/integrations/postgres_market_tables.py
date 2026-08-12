"""Validated PostgreSQL table identities for immutable market-data generations."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Final

_TABLE_IDENTIFIER: Final = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PostgresMarketTableSet:
    """One validated, immutable set of schema-qualified market tables."""

    kline: str
    funding: str
    indicator_manifest: str
    indicator_artifact: str

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if _TABLE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{item.name} table identifier is invalid")


LEGACY_MARKET_TABLES: Final = PostgresMarketTableSet(
    kline="market_raw.binance_usds_m_klines_202101_202606",
    funding="market_raw.binance_usds_m_funding_202101_202606",
    indicator_manifest=("market_raw.binance_usds_m_indicator_manifests_202101_202606"),
    indicator_artifact=("market_raw.binance_usds_m_indicator_artifacts_202101_202606"),
)
UNIVERSAL_202411_202607_TABLES: Final = PostgresMarketTableSet(
    kline="market_raw.binance_usds_m_klines_202411_202607",
    funding="market_raw.binance_usds_m_funding_202411_202607",
    indicator_manifest=("market_raw.binance_usds_m_indicator_manifests_202411_202607"),
    indicator_artifact=("market_raw.binance_usds_m_indicator_artifacts_202411_202607"),
)
UNIVERSAL_202411_202607_CACHE_ID: Final = (
    "binance-usds-m-native-indicators-15x-20241113-20260705-v1"
)

__all__ = [
    "LEGACY_MARKET_TABLES",
    "UNIVERSAL_202411_202607_CACHE_ID",
    "UNIVERSAL_202411_202607_TABLES",
    "PostgresMarketTableSet",
]
