"""Regression oracles: lower native cross-asset rolls must not be sampled first."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.data.test_multitimeframe_builder import (
    MemoryMultiTimeframeSource,
    _series,
)
from trade_rl.artifacts import content_digest
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)


def _fixture(
    *,
    delayed_eth_at: int | None = None,
    change_future: bool = False,
) -> tuple[MemoryMultiTimeframeSource, tuple[InstrumentContract, ...]]:
    deltas = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.04, -0.01])
    btc_close = 100.0 * np.exp(np.r_[0.0, np.cumsum(deltas)])
    eth_close = 200.0 * np.exp(np.r_[0.0, np.cumsum(2.0 * deltas)])
    if change_future:
        eth_close[5:] *= 1.11
    qh = [
        f"2026-01-01T{(index * 15) // 60:02d}:{(index * 15) % 60:02d}:00"
        for index in range(9)
    ]
    hourly = [qh[index] for index in (0, 4, 8)]
    avail = qh.copy()
    if delayed_eth_at is not None:
        avail[delayed_eth_at] = qh[delayed_eth_at + 2]
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _series(hourly, btc_close[[0, 4, 8]].tolist()),
            ("ETHUSDT", "1h"): _series(hourly, eth_close[[0, 4, 8]].tolist()),
            ("BTCUSDT", "15m"): _series(qh, btc_close.tolist()),
            ("ETHUSDT", "15m"): _series(
                qh, eth_close.tolist(), available_at=avail
            ),
        }
    )
    instruments = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for symbol in ("BTCUSDT", "ETHUSDT")
    )
    return source, instruments


def _config() -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        cross_asset_reference_symbol="BTCUSDT",
        features=(
            FeatureSpec(
                name="15m__log_return_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe="15m",
                lookback=1,
                max_staleness_hours=0.25,
            ),
            FeatureSpec(
                name="15m__relative_return",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
                lookback=1,
                min_periods=1,
                max_staleness_hours=0.25,
            ),
            FeatureSpec(
                name="15m__beta_4",
                kind=FeatureKind.ROLLING_BETA_TO_BTC,
                timeframe="15m",
                lookback=4,
                min_periods=4,
                max_staleness_hours=0.25,
            ),
            FeatureSpec(
                name="15m__corr_4",
                kind=FeatureKind.ROLLING_CORRELATION_TO_BTC,
                timeframe="15m",
                lookback=4,
                min_periods=4,
                max_staleness_hours=0.25,
            ),
            FeatureSpec(
                name="15m__rank_4",
                kind=FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
                timeframe="15m",
                lookback=4,
                min_periods=4,
                max_staleness_hours=0.25,
            ),
            FeatureSpec(
                name="15m__dispersion",
                kind=FeatureKind.CROSS_ASSET_DISPERSION,
                timeframe="15m",
                lookback=1,
                min_periods=1,
                max_staleness_hours=0.25,
            ),
        ),
    )


def test_native_fifteen_minute_rolling_beta_uses_four_intra_hour_events() -> None:
    source, instruments = _fixture()
    config = _config()
    dataset = MarketDatasetBuilder(config).build(source, instruments)
    assert dataset.symbols == ("BTCUSDT", "ETHUSDT")
    beta_idx = dataset.feature_names.index("15m__beta_4")
    corr_idx = dataset.feature_names.index("15m__corr_4")
    relative_idx = dataset.feature_names.index("15m__relative_return")
    rank_idx = dataset.feature_names.index("15m__rank_4")
    dispersion_idx = dataset.feature_names.index("15m__dispersion")
    ret_idx = dataset.feature_names.index("15m__log_return_1bar")

    assert not dataset.feature_available[0, :, beta_idx].any()
    assert dataset.feature_available[1, :, beta_idx].all()
    np.testing.assert_allclose(dataset.features[1, 1, beta_idx], 2.0, atol=1e-6)
    np.testing.assert_allclose(dataset.features[1, 1, corr_idx], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        dataset.features[1, 1, relative_idx], -0.02, atol=1e-6
    )
    np.testing.assert_allclose(dataset.features[1, :, rank_idx], [-1.0, 1.0])
    np.testing.assert_allclose(
        dataset.features[1, :, dispersion_idx], 0.01, atol=1e-6
    )
    np.testing.assert_allclose(dataset.features[1, 0, ret_idx], -0.02, atol=1e-6)
    assert dataset.feature_staleness_hours[1, 1, beta_idx] == 0.0
    assert dataset.identity_verified

    metadata = json.loads(dataset.identity_payload_json)
    assert metadata["config"]["native_cross_asset_alignment"] == (
        "native_before_base_sync_v1"
    )
    assert dataset.feature_config_digest != content_digest(config.canonical_payload())


def test_delayed_source_excludes_two_contaminated_native_pair_events() -> None:
    source, contracts = _fixture(delayed_eth_at=2)
    dataset = MarketDatasetBuilder(_config()).build(source, contracts)
    idx = dataset.feature_names.index("15m__beta_4")
    assert not dataset.feature_available[1, 1, idx]
    assert dataset.feature_available[2, 1, idx]


def test_native_cross_asset_history_is_future_mutation_invariant() -> None:
    source, contracts = _fixture()
    future, _ = _fixture(change_future=True)
    first = MarketDatasetBuilder(_config()).build(source, contracts)
    changed = MarketDatasetBuilder(_config()).build(future, contracts)
    np.testing.assert_array_equal(
        first.feature_available[:2], changed.feature_available[:2]
    )
    np.testing.assert_array_equal(first.features[:2], changed.features[:2])
    assert first.dataset_id != changed.dataset_id


@pytest.mark.parametrize(
    "kind",
    (
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
    ),
)
def test_native_cross_asset_reference_order_does_not_change_pair_statistics(
    kind: FeatureKind,
) -> None:
    source, contracts = _fixture()
    cfg = MarketBuildConfig(
        base_timeframe="1h",
        cross_asset_reference_symbol="BTCUSDT",
        features=(
            _config().features[0],
            FeatureSpec(
                name="pair",
                kind=kind,
                timeframe="15m",
                lookback=4,
                min_periods=4,
                max_staleness_hours=0.25,
            ),
        ),
    )
    normal = MarketDatasetBuilder(cfg).build(source, contracts)
    swapped = MarketDatasetBuilder(cfg).build(source, contracts[::-1])
    idx = normal.feature_names.index("pair")
    a = normal.symbols.index("ETHUSDT")
    b = swapped.symbols.index("ETHUSDT")
    assert normal.feature_available[1, a, idx]
    assert swapped.feature_available[1, b, idx]
    np.testing.assert_allclose(
        normal.features[1, a, idx],
        swapped.features[1, b, idx],
        atol=1e-6,
    )
