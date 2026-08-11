from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.binance_universal import binance_universal_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import (
    NATIVE_TIMEFRAMES,
    _align_indicators,
)
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer


def _bundle(
    symbols: tuple[str, ...], timestamps_ms: np.ndarray
) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m",
        feature_timeframes=("1h", "4h", "1d"),
    )
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol_index, symbol in enumerate(symbols):
        for timeframe in NATIVE_TIMEFRAMES:
            names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.arange(
                len(timestamps_ms) * len(names), dtype=np.float32
            ).reshape(len(timestamps_ms), len(names))
            values += float(symbol_index * 10_000)
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


def test_postgres_alignment_accepts_universal_206_feature_profile() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    timestamps_ms = 1_704_067_200_000 + np.arange(8, dtype=np.int64) * 900_000
    bundle = _bundle(symbols, timestamps_ms)
    specs = binance_universal_feature_specs(
        base_timeframe="15m",
        feature_timeframes=("1h", "4h", "1d"),
    )

    values, available, age, staleness, names, digest = _align_indicators(
        bundle,
        timestamps_ms=timestamps_ms,
        symbol_vocabulary=symbols,
        feature_specs=specs,
    )

    assert len(names) == 206
    assert names == tuple(spec.name for spec in specs)
    assert values.shape == (8, 2, 206)
    assert available.shape == values.shape
    assert age.shape == values.shape
    assert staleness.shape == values.shape
    assert len(digest) == 64
    assert not any("relative_return_to_btc" in name for name in names)
    assert not any("rolling_beta_to_btc" in name for name in names)


def test_postgres_alignment_default_profile_remains_legacy_226() -> None:
    symbols = ("BTCUSDT",)
    timestamps_ms = 1_704_067_200_000 + np.arange(4, dtype=np.int64) * 900_000
    bundle = _bundle(symbols, timestamps_ms)

    _, _, _, _, names, _ = _align_indicators(
        bundle,
        timestamps_ms=timestamps_ms,
        symbol_vocabulary=symbols,
    )

    assert len(names) == 226


def test_symbol_balanced_normalizer_excludes_unavailable_values() -> None:
    features = {
        "A": np.asarray([[0.0], [2.0], [50_000.0]], dtype=np.float64),
        "B": np.asarray([[100.0], [102.0], [-50_000.0]], dtype=np.float64),
    }
    available = {
        "A": np.asarray([[True], [True], [False]], dtype=np.bool_),
        "B": np.asarray([[True], [True], [False]], dtype=np.bool_),
    }

    normalizer = SymbolBalancedStandardNormalizer.fit(
        features,
        symbol_available=available,
        train_symbols=("A", "B"),
        feature_schema_digest="features-v1",
        catalog_digest="catalog-v1",
        split_manifest_digest="split-v1",
        fold_train_range=(0, 3),
        max_samples_per_symbol=3,
    )

    assert normalizer.mean[0] == 51.0
    transformed = normalizer.transform(
        np.asarray([[51.0], [999.0]], dtype=np.float64),
        available=np.asarray([[True], [False]], dtype=np.bool_),
    )
    assert transformed[0, 0] == 0.0
    assert transformed[1, 0] == 0.0
