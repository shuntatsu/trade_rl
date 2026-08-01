from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import (
    NATIVE_TIMEFRAMES,
    build_postgres_market_dataset,
)

pytestmark = pytest.mark.postgres

_STEP_MS = 15 * 60 * 1000


def _database_url() -> str:
    value = os.environ.get("TRADE_RL_TEST_DATABASE_URL")
    if not value:
        pytest.skip("TRADE_RL_TEST_DATABASE_URL is not configured")
    pytest.importorskip("psycopg")
    return value


def _indicator_bundle(
    symbols: tuple[str, ...], start_ms: int
) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m",
        feature_timeframes=("1h", "4h", "1d"),
    )
    event_time_ms = start_ms + np.arange(1, 5, dtype=np.int64) * _STEP_MS
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol_index, symbol in enumerate(symbols):
        for timeframe in NATIVE_TIMEFRAMES:
            feature_names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.full(
                (4, len(feature_names)),
                float(symbol_index + 1),
                dtype=np.float32,
            )
            available = np.ones(values.shape, dtype=np.bool_)
            artifacts.append(
                NativeIndicatorArtifact(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_names=feature_names,
                    event_time_ms=event_time_ms.copy(),
                    values=values,
                    available=available,
                    payload_schema=f"npz_native_indicator_v1:{'1' * 64}",
                    payload_sha256=f"{symbol_index + 1:064x}",
                )
            )
    return NativeIndicatorArtifactBundle(
        cache_id="live-adapter-test-cache",
        market="usds-m",
        symbols=symbols,
        timeframes=NATIVE_TIMEFRAMES,
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 7, 1, tzinfo=UTC),
        feature_config_digest="2" * 64,
        artifacts=tuple(artifacts),
    )


def _metadata(
    symbols: tuple[str, ...], start: datetime
) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "listed_at": start.isoformat(),
            "tick_size": str(Decimal("0.01") + Decimal(index) / Decimal("1000")),
            "lot_size": str(Decimal("0.001") + Decimal(index) / Decimal("10000")),
            "minimum_notional": str(Decimal("5") + Decimal(index)),
        }
        for index, symbol in enumerate(symbols)
    }


def test_build_postgres_market_dataset_against_live_postgres() -> None:
    psycopg = pytest.importorskip("psycopg")
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    start_ms = int(start.timestamp() * 1000)

    connection = psycopg.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS market_raw")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS
                    market_raw.binance_usds_m_klines_202101_202606 (
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        open_time_ms BIGINT NOT NULL,
                        open NUMERIC NOT NULL,
                        high NUMERIC NOT NULL,
                        low NUMERIC NOT NULL,
                        close NUMERIC NOT NULL,
                        quote_volume NUMERIC NOT NULL,
                        PRIMARY KEY (symbol, interval, open_time_ms)
                    )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS
                    market_raw.binance_usds_m_funding_202101_202606 (
                        symbol TEXT NOT NULL,
                        calc_time_ms BIGINT NOT NULL,
                        last_funding_rate NUMERIC NOT NULL,
                        PRIMARY KEY (symbol, calc_time_ms)
                    )
                """
            )
            cursor.execute("TRUNCATE market_raw.binance_usds_m_klines_202101_202606")
            cursor.execute("TRUNCATE market_raw.binance_usds_m_funding_202101_202606")

            for symbol_index, symbol in enumerate(symbols):
                base = Decimal("10") + Decimal(symbol_index)
                for row in range(4):
                    offset = Decimal(row)
                    cursor.execute(
                        """
                        INSERT INTO market_raw.binance_usds_m_klines_202101_202606 (
                            symbol, interval, open_time_ms,
                            open, high, low, close, quote_volume
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            symbol,
                            "15m",
                            start_ms + row * _STEP_MS,
                            base + offset,
                            base + offset + Decimal("1"),
                            base + offset - Decimal("1"),
                            base + offset + Decimal("0.5"),
                            Decimal("1000") + offset,
                        ),
                    )
                cursor.execute(
                    """
                    INSERT INTO market_raw.binance_usds_m_funding_202101_202606 (
                        symbol, calc_time_ms, last_funding_rate
                    ) VALUES (%s, %s, %s)
                    """,
                    (symbol, start_ms + 2 * _STEP_MS, Decimal("0.0001")),
                )

        dataset = build_postgres_market_dataset(
            connection,
            symbols=symbols,
            symbol_vocabulary=("BTCUSDT", *symbols, "XRPUSDT"),
            start_time=start,
            end_time=end,
            metadata=_metadata(symbols, start),
            metadata_evidence_digest="3" * 64,
            indicator_bundle=_indicator_bundle(symbols, start_ms),
        )

        assert dataset.symbols == symbols
        assert dataset.n_bars == 4
        assert dataset.n_features == 226
        np.testing.assert_array_equal(
            dataset.timestamps,
            np.datetime64("2024-01-01T00:15:00", "ns")
            + np.arange(4) * np.timedelta64(15, "m"),
        )
        np.testing.assert_allclose(dataset.open[:, 0], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(dataset.close[:, 2], [12.5, 13.5, 14.5, 15.5])
        np.testing.assert_allclose(
            dataset.volume[:, 1], [1000.0, 1001.0, 1002.0, 1003.0]
        )
        np.testing.assert_array_equal(dataset.funding_event_count[:, 0], [0, 1, 0, 0])
        np.testing.assert_allclose(dataset.funding_rate[:, 0], [0.0, 0.0001, 0.0, 0.0])
        assert np.isfinite(dataset.open).all()
        assert dataset.identity_verified
    finally:
        connection.rollback()
        connection.close()
