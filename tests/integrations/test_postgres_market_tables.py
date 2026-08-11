from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
    PostgresMarketTableSet,
)


def test_period_correct_table_set_is_exact_and_immutable() -> None:
    tables = UNIVERSAL_202411_202607_TABLES

    assert tables.kline == "market_raw.binance_usds_m_klines_202411_202607"
    assert tables.funding == "market_raw.binance_usds_m_funding_202411_202607"
    assert tables.indicator_manifest == (
        "market_raw.binance_usds_m_indicator_manifests_202411_202607"
    )
    assert tables.indicator_artifact == (
        "market_raw.binance_usds_m_indicator_artifacts_202411_202607"
    )
    assert UNIVERSAL_202411_202607_CACHE_ID == (
        "binance-usds-m-native-indicators-15x-20241113-20260705-v1"
    )
    with pytest.raises(FrozenInstanceError):
        tables.kline = "market_raw.replacement"  # type: ignore[misc]


def test_table_set_rejects_sql_identifier_injection() -> None:
    with pytest.raises(ValueError, match="table identifier"):
        PostgresMarketTableSet(
            kline="market_raw.safe; DROP TABLE public.rl_klines",
            funding="market_raw.funding",
            indicator_manifest="market_raw.manifest",
            indicator_artifact="market_raw.artifact",
        )
