"""Independent native event oracles: sub-hour cross asset windows must not subsample."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.identity import DATASET_ID_ARRAY_FIELDS
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries


class _Source:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self.values = values

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        return self.values[symbol, timeframe]


def _bars(start: str, minutes: int, closes: np.ndarray) -> RawMarketSeries:
    times = np.datetime64(start, "ns") + np.arange(len(closes)) * np.timedelta64(
        minutes, "m"
    )
    previous = np.concatenate((closes[:1], closes[:-1]))
    return RawMarketSeries(
        timestamps=times,
        available_at=times,
        open=previous,
        high=np.maximum(previous, closes),
        low=np.minimum(previous, closes),
        close=closes,
        volume=np.ones(len(closes), dtype=np.float64) * 100_000.0,
        funding_rate=np.zeros(len(closes)),
        funding_available=np.zeros(len(closes), dtype=np.bool_),
        tradable=np.ones(len(closes), dtype=np.bool_),
    )


def _source() -> _Source:
    steps = np.arange(32, dtype=np.float64)
    btc_returns = 0.022 * np.sin(steps * 0.69) + 0.008 * np.cos(steps * 0.31)
    eth_returns = 0.017 * np.cos(steps * 0.53) - 0.007 * np.sin(steps * 0.41)
    btc_close = 100.0 * np.exp(np.r_[0.0, np.cumsum(btc_returns)])
    eth_close = 90.0 * np.exp(np.r_[0.0, np.cumsum(eth_returns)])
    return _Source(
        {
            ("BTCUSDT", "15m"): _bars("2026-01-01T00:00", 15, btc_close),
            ("ETHUSDT", "15m"): _bars("2026-01-01T00:00", 15, eth_close),
            ("BTCUSDT", "1h"): _bars("2026-01-01T00:00", 60, btc_close[::4]),
            ("ETHUSDT", "1h"): _bars("2026-01-01T00:00", 60, eth_close[::4]),
        }
    )


def _dataset(source: _Source) -> MarketDataset:
    specs = (
        FeatureSpec(
            name="15m__log_return_1bar",
            kind=FeatureKind.LOG_RETURN,
            timeframe="15m",
            lookback=1,
        ),
        FeatureSpec(
            name="15m__rolling_correlation_to_btc_4bar",
            kind=FeatureKind.ROLLING_CORRELATION_TO_BTC,
            timeframe="15m",
            lookback=4,
            min_periods=4,
        ),
        FeatureSpec(
            name="15m__rolling_beta_to_btc_4bar",
            kind=FeatureKind.ROLLING_BETA_TO_BTC,
            timeframe="15m",
            lookback=4,
            min_periods=4,
        ),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=specs,
            cross_asset_reference_symbol="BTCUSDT",
        )
    ).build(
        source,
        (
            InstrumentContract("BTCUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
            InstrumentContract("ETHUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ),
    )


def test_15m_rolling_correlation_and_beta_use_four_consecutive_native_bars() -> None:
    dataset = _dataset(_source())
    btc = _source().values["BTCUSDT", "15m"].close
    eth = _source().values["ETHUSDT", "15m"].close
    br = np.diff(np.log(btc))[-4:]
    er = np.diff(np.log(eth))[-4:]
    expected_corr = np.corrcoef(br, er)[0, 1]
    expected_beta = np.cov(br, er, ddof=0)[0, 1] / np.var(br)
    actual_corr = dataset.features[8, 1, 1]
    actual_beta = dataset.features[8, 1, 2]
    assert dataset.feature_available[8, 1, 1:3].all()
    # MarketDataset features are stored as float32: permit serialization rounding.
    assert actual_corr == pytest.approx(expected_corr, abs=1e-6)
    assert actual_beta == pytest.approx(expected_beta, abs=1e-6)
    # An hourly sample of one 15m return per hour must NOT equal a native
    # four-consecutive-bar rolling statistic.
    hourly_sample = np.corrcoef(
        np.diff(np.log(btc))[3::4][-4:],
        np.diff(np.log(eth))[3::4][-4:],
    )[0, 1]
    assert abs(expected_corr - hourly_sample) > 0.01


def test_native_cross_asset_future_modification_preserves_earlier_decisions() -> None:
    original_source = _source()
    original = _dataset(original_source)
    raw = original_source.values["BTCUSDT", "15m"]
    altered_close = raw.close.copy()
    altered_close[22:] *= 1.21
    mutated = dict(original_source.values)
    mutated["BTCUSDT", "15m"] = replace(
        raw,
        close=altered_close,
        high=np.maximum(raw.high, altered_close),
        low=np.minimum(raw.low, altered_close),
    )
    changed = _dataset(_Source(mutated))
    np.testing.assert_array_equal(original.features[:5], changed.features[:5])
    np.testing.assert_array_equal(
        original.feature_available[:5], changed.feature_available[:5]
    )
    for field in (
        "feature_staleness_hours",
        "feature_staleness",
        "feature_missing_reason",
        "global_features",
        "global_feature_available",
        "global_feature_staleness_hours",
        "global_feature_missing_reason",
    ):
        np.testing.assert_array_equal(
            getattr(original, field)[:5], getattr(changed, field)[:5]
        )


def test_delayed_native_publication_cannot_backdate_cross_asset_features() -> None:
    source = _source()
    raw = source.values["BTCUSDT", "15m"]
    later = raw.available_at.copy()
    later[16] = np.datetime64("2026-01-01T05:00", "ns")
    modified = dict(source.values)
    modified["BTCUSDT", "15m"] = replace(raw, available_at=later)
    normal = _dataset(source)
    delayed = _dataset(_Source(modified))
    np.testing.assert_array_equal(normal.features[:4], delayed.features[:4])
    # At hour four the native event ending at hour four has NOT been observed.
    assert delayed.feature_staleness_hours[4, 1, 1] > 0.0
    assert normal.feature_staleness_hours[4, 1, 1] == pytest.approx(0.0)


def _controlled_source(timeframe: str, *, proportional: bool = False) -> _Source:
    reference = np.array([0.02, -0.01, 0.03, -0.02])
    asset = 2.0 * reference if proportional else np.array([0.03, 0.04, -0.01, 0.05])
    minutes = {"15m": 15, "1h": 60, "4h": 240}[timeframe]
    values = {}
    for symbol, returns in (("BTCUSDT", reference), ("ETHUSDT", asset)):
        values[symbol, "1h"] = _bars("2026-01-01T00:00", 60, np.full(21, 100.0))
        values[symbol, timeframe] = _bars(
            "2026-01-01T00:00", minutes, 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
        )
    return _Source(values)


def _controlled_config(timeframe: str, *, lookback: int = 4) -> MarketBuildConfig:
    native = None if timeframe == "1h" else timeframe
    kinds = (
        FeatureKind.LOG_RETURN,
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_ASSET_DISPERSION,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
    )
    rolling = {
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
    }
    return MarketBuildConfig(
        base_timeframe="1h",
        cross_asset_reference_symbol="BTCUSDT",
        features=tuple(
            FeatureSpec(
                name=f"{timeframe}__{kind.value}",
                kind=kind,
                timeframe=native,
                lookback=lookback if kind in rolling else 1,
                min_periods=lookback if kind in rolling else 1,
                max_staleness_hours=6.0,
            )
            for kind in kinds
        ),
    )


def _controlled_dataset(
    source: _Source,
    config: MarketBuildConfig,
    *,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> MarketDataset:
    return MarketDatasetBuilder(config).build(
        source,
        tuple(
            InstrumentContract(symbol, listed_at=datetime(2025, 1, 1, tzinfo=UTC))
            for symbol in symbols
        ),
    )


def test_finer_native_all_channels_have_literal_first_hour_oracle() -> None:
    dataset = _controlled_dataset(
        _controlled_source("15m", proportional=True), _controlled_config("15m")
    )
    assert dataset.identity_verified
    assert dataset.feature_available[1, :, 1:].all()
    # Four actual native returns, rather than a single hourly sample, are present.
    assert dataset.features[1, 1, 1:5] == pytest.approx([-0.02, 1.0, 2.0, 0.01])
    assert dataset.features[1, :, 5] == pytest.approx([-1.0, 1.0])
    assert not dataset.feature_available[0, :, 2:4].any()


@pytest.mark.parametrize("invalid_row", ["delayed", "untradable"])
def test_finer_invalid_row_excludes_both_touching_returns_without_reset(
    invalid_row: str,
) -> None:
    source = _controlled_source("15m")
    raw = source.values["BTCUSDT", "15m"]
    if invalid_row == "delayed":
        arrivals = raw.available_at.copy()
        arrivals[2] = np.datetime64("2026-01-01T02:00", "ns")
        source.values["BTCUSDT", "15m"] = replace(raw, available_at=arrivals)
    else:
        tradable = raw.tradable.copy()
        tradable[2] = False
        source.values["BTCUSDT", "15m"] = replace(raw, tradable=tradable)
    dataset = _controlled_dataset(source, _controlled_config("15m", lookback=2))
    assert dataset.feature_available[1, 1, 3]
    # Eligible pair events e1/e4 survive the gap: (.02,.03), (-.02,.05).
    assert dataset.features[1, 1, 3] == pytest.approx(-0.5, abs=1e-6)
    assert dataset.features[1, 1, 2] == pytest.approx(-1.0, abs=1e-6)
    assert dataset.feature_staleness_hours[1, 1, 3] == 0.0


def test_coarser_delayed_publication_preserves_observed_pair_history() -> None:
    source = _controlled_source("4h")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        raw = source.values[symbol, "4h"]
        arrivals = raw.available_at.copy()
        arrivals[2] = np.datetime64("2026-01-01T13:00", "ns")
        source.values[symbol, "4h"] = replace(raw, available_at=arrivals)
    dataset = _controlled_dataset(source, _controlled_config("4h", lookback=2))
    assert not dataset.feature_available[:13, 1, 3].any()
    assert dataset.features[13:16, 1, 3] == pytest.approx([-4.0] * 3, abs=1e-6)
    assert dataset.feature_staleness_hours[13:16, 1, 3] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert dataset.feature_staleness[13:16, 1, 3] == pytest.approx(
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]
    )
    assert dataset.features[16, 1, 3] == pytest.approx(-1.2, abs=1e-6)
    assert dataset.feature_staleness_hours[16, 1, 3] == 0.0


@pytest.mark.parametrize(
    "timeframe,cross_asset", [("15m", True), ("15m", False), ("1h", True), ("4h", True)]
)
def test_only_finer_cross_asset_config_binds_native_alignment_identity(
    timeframe: str,
    cross_asset: bool,
) -> None:
    config = _controlled_config(timeframe, lookback=2)
    if not cross_asset:
        config = replace(config, features=config.features[:1])
    dataset = _controlled_dataset(_controlled_source(timeframe), config)
    expected = config.canonical_payload()
    if timeframe == "15m" and cross_asset:
        expected["native_cross_asset_alignment"] = "native_before_base_sync_v1"
        expected["native_cross_asset_history"] = "last_n_eligible_pair_events_v1"
    assert dataset.feature_config_digest == content_digest(expected)
    assert dataset.identity_payload_json is not None
    payload = json.loads(dataset.identity_payload_json)
    assert payload["config"] == json.loads(json.dumps(expected))
    assert payload["feature_config_digest"] == dataset.feature_config_digest
    assert dataset.identity_verified


def test_equal_clock_cross_asset_values_and_ages_retain_literal_oracle() -> None:
    dataset = _controlled_dataset(
        _controlled_source("1h"), _controlled_config("1h", lookback=2)
    )
    assert dataset.features[2:5, 1, 3] == pytest.approx(
        [-1.0 / 3.0, -1.25, -1.2], abs=1e-6
    )
    assert dataset.feature_available[2:5, 1, 3].all()
    assert dataset.feature_staleness_hours[2:5, 1, 3] == pytest.approx([0.0] * 3)


def test_finer_route_preserves_symbol_order_and_financial_inputs() -> None:
    source = _controlled_source("15m")
    config = _controlled_config("15m", lookback=2)
    dataset = _controlled_dataset(source, config)
    reordered = _controlled_dataset(source, config, symbols=("ETHUSDT", "BTCUSDT"))
    for field in (
        "features",
        "feature_available",
        "feature_staleness_hours",
        "feature_staleness",
        "feature_missing_reason",
    ):
        np.testing.assert_array_equal(
            getattr(dataset, field), getattr(reordered, field)[:, ::-1]
        )
    baseline = _controlled_dataset(
        source, replace(config, features=config.features[:1])
    )
    local = {
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
    }
    for field in DATASET_ID_ARRAY_FIELDS:
        if field not in local:
            np.testing.assert_array_equal(
                getattr(dataset, field), getattr(baseline, field)
            )
    assert dataset.nominal_bar_hours == baseline.nominal_bar_hours == 1.0
    assert dataset.periods_per_year == baseline.periods_per_year


def test_finer_native_feature_expiry_keeps_age_and_masks_zero_values() -> None:
    dataset = _controlled_dataset(
        _controlled_source("15m", proportional=True), _controlled_config("15m")
    )
    assert dataset.feature_available[7, :, 2:4].all()
    assert not dataset.feature_available[8, :, 2:4].any()
    assert not dataset.features[8, :, 2:4].any()
    assert dataset.feature_staleness_hours[8, :, 2:4] == pytest.approx(
        np.full((2, 2), 7.0)
    )
    assert dataset.feature_staleness[8, :, 2:4] == pytest.approx(np.ones((2, 2)))


def test_marked_finer_dataset_artifact_retains_identity_and_dependency_config(
    tmp_path: Path,
) -> None:
    dataset = _controlled_dataset(_controlled_source("15m"), _controlled_config("15m"))
    write_market_dataset_files(tmp_path, dataset)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.identity_verified
    assert restored.dataset_id == dataset.dataset_id
    assert restored.feature_config_digest == dataset.feature_config_digest
    assert restored.identity_payload_json == dataset.identity_payload_json
    assert restored.identity_payload_json is not None
    assert (
        json.loads(restored.identity_payload_json)["config"][
            "native_cross_asset_alignment"
        ]
        == "native_before_base_sync_v1"
    )
    assert (
        json.loads(restored.identity_payload_json)["config"].get(
            "native_cross_asset_history"
        )
        == "last_n_eligible_pair_events_v1"
    )
