from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
)
from trade_rl.data.source import (
    InMemoryMarketDataSource,
    RawIndexPriceSeries,
    RawMarketSeries,
)


def _timestamps(n: int) -> np.ndarray:
    return np.datetime64("2022-01-01T01:00:00", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )


def _trade_series(
    n: int = 30,
    *,
    close: float = 110.0,
    unavailable_at: int | None = None,
) -> RawMarketSeries:
    timestamps = _timestamps(n)
    closes = np.full(n, close, dtype=np.float64)
    available_at = timestamps.copy()
    if unavailable_at is not None:
        available_at[unavailable_at] = timestamps[unavailable_at] + np.timedelta64(1, "h")
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=closes.copy(),
        high=closes.copy(),
        low=closes.copy(),
        close=closes,
        volume=np.full(n, 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros(n, dtype=np.float64),
        tradable=np.ones(n, dtype=np.bool_),
    )


def _index_series(
    n: int = 30,
    *,
    close: float = 100.0,
    omit: int | None = None,
    delay: int | None = None,
) -> RawIndexPriceSeries:
    timestamps = _timestamps(n)
    closes = np.full(n, close, dtype=np.float64)
    if omit is not None:
        keep = np.ones(n, dtype=np.bool_)
        keep[omit] = False
        timestamps = timestamps[keep]
        closes = closes[keep]
    available_at = timestamps.copy()
    if delay is not None:
        delayed_timestamp = _timestamps(n)[delay]
        matches = np.flatnonzero(timestamps == delayed_timestamp)
        assert matches.size == 1
        available_at[matches[0]] = delayed_timestamp + np.timedelta64(1, "h")
    return RawIndexPriceSeries(
        timestamps=timestamps,
        available_at=available_at,
        close=closes,
    )


class _IndexCapableSource:
    def __init__(
        self,
        trade: RawMarketSeries,
        index: RawIndexPriceSeries,
        *,
        provenance_digest: str = "a" * 64,
    ) -> None:
        self._trade = trade
        self._index = index
        self._provenance_digest = provenance_digest

    def load(self, symbol: str) -> RawMarketSeries:
        assert symbol == "BTCUSDT"
        return self._trade

    def load_index_price(self, symbol: str, timeframe: str) -> RawIndexPriceSeries:
        assert symbol == "BTCUSDT"
        assert timeframe == "1h"
        return self._index

    @property
    def index_price_provenance(self) -> Mapping[str, object]:
        return {
            "schema_version": "synthetic_index_source_v1",
            "source_family": "indexPriceKlines",
            "manifest_digest": self._provenance_digest,
        }


def _instrument() -> tuple[InstrumentContract, ...]:
    return (
        InstrumentContract(
            symbol="BTCUSDT",
            listed_at=datetime(2021, 1, 1, tzinfo=UTC),
        ),
    )


def _basis_spec(**changes: object) -> FeatureSpec:
    values: dict[str, object] = {
        "name": "1h__perp_index_log_basis_bps",
        "kind": FeatureKind.PERP_INDEX_LOG_BASIS_BPS,
        "lookback": 1,
        "normalization": NormalizationMode.NONE,
        "normalization_window": 1,
        "min_periods": 1,
    }
    values.update(changes)
    return FeatureSpec(**values)  # type: ignore[arg-type]


def _basis_config(**spec_changes: object) -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(_basis_spec(**spec_changes),),
    )


def test_basis_is_exact_log_ratio_bps_not_simple_percentage() -> None:
    dataset = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(_trade_series(), _index_series()),
        _instrument(),
    )

    expected = 10_000.0 * np.log(1.1)
    simple_percentage = 10_000.0 * (1.1 - 1.0)
    np.testing.assert_allclose(dataset.features[:, 0, 0], expected, rtol=0.0, atol=1e-5)
    assert not np.isclose(expected, simple_percentage)
    assert np.all(dataset.feature_available[:, 0, 0])


def test_sparse_index_row_is_unavailable_and_never_stale_carried() -> None:
    dataset = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(_trade_series(), _index_series(omit=10)),
        _instrument(),
    )

    assert dataset.feature_available[9, 0, 0]
    assert not dataset.feature_available[10, 0, 0]
    assert dataset.features[10, 0, 0] == 0.0
    assert dataset.feature_available[11, 0, 0]
    np.testing.assert_allclose(dataset.features[11, 0, 0], 10_000.0 * np.log(1.1))
    assert dataset.feature_staleness[10, 0, 0] == 1.0
    assert dataset.feature_age_hours[10, 0, 0] == _basis_spec().max_staleness_hours


def test_index_delay_and_perpetual_information_availability_fail_closed() -> None:
    index_delayed = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(_trade_series(), _index_series(delay=10)),
        _instrument(),
    )
    assert not index_delayed.feature_available[10, 0, 0]
    assert index_delayed.features[10, 0, 0] == 0.0
    assert index_delayed.feature_available[11, 0, 0]

    perp_delayed = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(_trade_series(unavailable_at=10), _index_series()),
        _instrument(),
    )
    assert not perp_delayed.feature_available[10, 0, 0]
    assert perp_delayed.features[10, 0, 0] == 0.0
    assert perp_delayed.feature_available[11, 0, 0]


def test_future_index_mutation_cannot_change_prior_basis_prefix() -> None:
    trade = _trade_series(40)
    base_index = _index_series(40)
    mutated_close = np.asarray(base_index.close).copy()
    mutated_close[30:] *= 1.5
    mutated_index = RawIndexPriceSeries(
        timestamps=base_index.timestamps,
        available_at=base_index.available_at,
        close=mutated_close,
    )

    baseline = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(trade, base_index),
        _instrument(),
    )
    mutated = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(trade, mutated_index),
        _instrument(),
    )

    np.testing.assert_array_equal(baseline.features[:30], mutated.features[:30])
    np.testing.assert_array_equal(
        baseline.feature_available[:30], mutated.feature_available[:30]
    )


def test_missing_index_source_capability_is_rejected_only_when_basis_requested() -> None:
    plain = InMemoryMarketDataSource({"BTCUSDT": _trade_series()})

    with pytest.raises(ValueError, match="index.*source|capability|basis"):
        MarketDatasetBuilder(_basis_config()).build(plain, _instrument())

    legacy_config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="ret_1",
                kind=FeatureKind.LOG_RETURN,
                lookback=1,
            ),
        ),
    )
    dataset = MarketDatasetBuilder(legacy_config).build(plain, _instrument())
    assert dataset.feature_names == ("ret_1",)


def test_index_capability_is_identity_inert_when_basis_feature_is_omitted() -> None:
    trade = _trade_series()
    legacy_config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret_1", kind=FeatureKind.LOG_RETURN, lookback=1),
        ),
    )
    plain = MarketDatasetBuilder(legacy_config).build(
        InMemoryMarketDataSource({"BTCUSDT": trade}),
        _instrument(),
    )
    enriched = MarketDatasetBuilder(legacy_config).build(
        _IndexCapableSource(trade, _index_series(), provenance_digest="b" * 64),
        _instrument(),
    )

    assert enriched.dataset_id == plain.dataset_id
    np.testing.assert_array_equal(enriched.features, plain.features)
    np.testing.assert_array_equal(enriched.feature_available, plain.feature_available)
    assert enriched.identity_arrays().keys() == plain.identity_arrays().keys()
    for name in plain.identity_arrays():
        np.testing.assert_array_equal(
            enriched.identity_arrays()[name], plain.identity_arrays()[name]
        )


def test_basis_source_provenance_changes_identity_but_not_accounting_index_price() -> None:
    trade = _trade_series()
    first = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(trade, _index_series(), provenance_digest="c" * 64),
        _instrument(),
    )
    second = MarketDatasetBuilder(_basis_config()).build(
        _IndexCapableSource(trade, _index_series(), provenance_digest="d" * 64),
        _instrument(),
    )

    assert first.dataset_id != second.dataset_id
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.index_price, first.close)
    np.testing.assert_array_equal(second.index_price, second.close)


def test_index_raw_contract_rejects_nonpositive_duplicate_and_early_availability() -> None:
    timestamps = _timestamps(3)
    with pytest.raises(ValueError, match="positive|finite"):
        RawIndexPriceSeries(
            timestamps=timestamps,
            close=np.array([100.0, 0.0, 101.0]),
        )
    with pytest.raises(ValueError, match="increasing|unique"):
        RawIndexPriceSeries(
            timestamps=np.array([timestamps[0], timestamps[0], timestamps[2]]),
            close=np.array([100.0, 100.0, 101.0]),
        )
    with pytest.raises(ValueError, match="available_at|availability"):
        RawIndexPriceSeries(
            timestamps=timestamps,
            available_at=timestamps - np.timedelta64(1, "h"),
            close=np.array([100.0, 100.0, 101.0]),
        )


@pytest.mark.parametrize(
    "changes, pattern",
    [
        ({"name": "wrong_basis"}, "canonical|name|basis"),
        ({"lookback": 2}, "lookback|basis"),
        ({"normalization": NormalizationMode.ROLLING_ZSCORE}, "normalization|basis"),
        ({"normalization_window": 2}, "normalization|window|basis"),
        ({"min_periods": 2}, "min_periods|basis"),
        ({"timeframe": "4h"}, "1h|timeframe|basis"),
    ],
)
def test_basis_feature_config_rejects_semantic_drift(
    changes: dict[str, object], pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _basis_config(**changes)

    if "timeframe" not in changes:
        with pytest.raises(ValueError, match="1h|base.*clock|basis"):
            MarketBuildConfig(
                base_timeframe="15m",
                features=(_basis_spec(**changes),),
            )
