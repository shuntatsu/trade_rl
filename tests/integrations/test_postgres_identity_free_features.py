from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import (
    NATIVE_TIMEFRAMES,
    _align_indicators,
)


def _bundle(symbols: tuple[str, ...], timestamps_ms: np.ndarray) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol_index, symbol in enumerate(symbols):
        for timeframe in NATIVE_TIMEFRAMES:
            names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.full(
                (len(timestamps_ms), len(names)),
                float(symbol_index + 1),
                dtype=np.float32,
            )
            artifacts.append(
                NativeIndicatorArtifact(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_names=names,
                    event_time_ms=timestamps_ms.copy(),
                    values=values,
                    available=np.ones(values.shape, dtype=np.bool_),
                    payload_schema=f"npz_native_indicator_v1:{'1' * 64}",
                    payload_sha256=f"{symbol_index + 1:064x}",
                )
            )
    return NativeIndicatorArtifactBundle(
        cache_id="cache",
        market="usds-m",
        symbols=symbols,
        timeframes=NATIVE_TIMEFRAMES,
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 7, 1, tzinfo=UTC),
        feature_config_digest="2" * 64,
        artifacts=tuple(artifacts),
    )


def test_postgres_policy_features_do_not_encode_symbol_vocabulary() -> None:
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    timestamps_ms = 1_704_067_200_000 + np.arange(4, dtype=np.int64) * 900_000
    bundle = _bundle(symbols, timestamps_ms)
    first_vocabulary = (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
    )
    second_vocabulary = tuple(reversed(first_vocabulary))

    first = _align_indicators(
        bundle,
        timestamps_ms=timestamps_ms,
        symbol_vocabulary=first_vocabulary,
    )
    second = _align_indicators(
        bundle,
        timestamps_ms=timestamps_ms,
        symbol_vocabulary=second_vocabulary,
    )

    first_values, first_available, first_age, first_staleness, names, digest = first
    second_values, second_available, second_age, second_staleness, other_names, other_digest = second

    assert len(names) == 226
    assert names == other_names
    assert not any("symbol_id" in name for name in names)
    assert digest == other_digest
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(first_available, second_available)
    np.testing.assert_array_equal(first_age, second_age)
    np.testing.assert_array_equal(first_staleness, second_staleness)
