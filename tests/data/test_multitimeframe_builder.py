from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import MultiTimeframeMarketDataSource, RawMarketSeries


def _series(
    timestamps: list[str],
    closes: list[float],
    *,
    available_at: list[str] | None = None,
) -> RawMarketSeries:
    timestamp_array = np.asarray(timestamps, dtype="datetime64[ns]")
    close = np.asarray(closes, dtype=np.float64)
    open_price = np.concatenate((close[:1], close[:-1]))
    return RawMarketSeries(
        timestamps=timestamp_array,
        available_at=(
            timestamp_array
            if available_at is None
            else np.asarray(available_at, dtype="datetime64[ns]")
        ),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.arange(1, len(close) + 1, dtype=np.float64) * 100.0,
        funding_rate=np.zeros(len(close), dtype=np.float64),
        funding_available=np.zeros(len(close), dtype=np.bool_),
        tradable=np.ones(len(close), dtype=np.bool_),
    )


class MemoryMultiTimeframeSource:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self.values = values

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        return self.values[(symbol, timeframe)]


def _base() -> RawMarketSeries:
    timestamps = [f"2026-01-01T{hour:02d}:00:00" for hour in range(9)]
    return _series(timestamps, [100.0 + hour for hour in range(9)])


def _contract() -> tuple[InstrumentContract, ...]:
    return (
        InstrumentContract(
            symbol="BTCUSDT",
            listed_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


def _config(*, max_staleness_hours: float = 24.0) -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="4h__ret_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe="4h",
                lookback=1,
                max_staleness_hours=max_staleness_hours,
            ),
        ),
    )


def test_feature_spec_binds_native_timeframe_to_identity() -> None:
    hourly = FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    four_hour = FeatureSpec(
        name="ret",
        kind=FeatureKind.LOG_RETURN,
        timeframe="4h",
    )

    assert hourly.resolved_timeframe("1h") == "1h"
    assert four_hour.resolved_timeframe("1h") == "4h"
    assert four_hour.canonical_payload()["timeframe"] == "4h"
    assert hourly.canonical_payload() != four_hour.canonical_payload()

    with pytest.raises(ValueError, match="timeframe"):
        FeatureSpec(name="bad", kind=FeatureKind.LOG_RETURN, timeframe="3h")


def test_multitimeframe_protocol_is_runtime_checkable() -> None:
    source = MemoryMultiTimeframeSource({("BTCUSDT", "1h"): _base()})
    assert isinstance(source, MultiTimeframeMarketDataSource)


def test_four_hour_feature_is_not_visible_before_native_close() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T04:00:00",
                    "2026-01-01T08:00:00",
                ],
                [100.0, 104.0, 112.0],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config()).build(source, _contract())

    assert not dataset.feature_available[3, 0, 0]
    assert dataset.feature_available[4, 0, 0]
    np.testing.assert_allclose(dataset.features[4, 0, 0], np.log(104.0 / 100.0))
    np.testing.assert_allclose(dataset.features[7, 0, 0], np.log(104.0 / 100.0))
    np.testing.assert_allclose(dataset.features[8, 0, 0], np.log(112.0 / 104.0))


def test_native_availability_delay_prevents_same_close_leakage() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T04:00:00",
                    "2026-01-01T08:00:00",
                ],
                [100.0, 104.0, 112.0],
                available_at=[
                    "2026-01-01T00:00:00",
                    "2026-01-01T05:00:00",
                    "2026-01-01T08:00:00",
                ],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config()).build(source, _contract())

    assert not dataset.feature_available[4, 0, 0]
    assert dataset.feature_available[5, 0, 0]
    np.testing.assert_allclose(dataset.features[5, 0, 0], np.log(104.0 / 100.0))
    assert dataset.feature_staleness_hours[5, 0, 0] == pytest.approx(1.0)


def test_multitimeframe_staleness_expires_on_base_clock() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                ["2026-01-01T00:00:00", "2026-01-01T04:00:00"],
                [100.0, 104.0],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config(max_staleness_hours=1.0)).build(
        source,
        _contract(),
    )

    assert dataset.feature_available[4, 0, 0]
    assert dataset.feature_available[5, 0, 0]
    assert not dataset.feature_available[6, 0, 0]
    assert dataset.feature_staleness[6, 0, 0] == pytest.approx(1.0)


def test_latest_closed_fifteen_minute_feature_aligns_to_hourly_decision() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T02:00:00",
                ],
                [100.0, 110.0, 120.0],
            ),
            ("BTCUSDT", "15m"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:15:00",
                    "2026-01-01T00:30:00",
                    "2026-01-01T00:45:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T01:15:00",
                    "2026-01-01T01:30:00",
                    "2026-01-01T01:45:00",
                    "2026-01-01T02:00:00",
                ],
                [100.0, 101.0, 103.0, 106.0, 110.0, 111.0, 113.0, 116.0, 120.0],
            ),
        }
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="15m__ret_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe="15m",
                lookback=1,
            ),
        ),
    )

    dataset = MarketDatasetBuilder(config).build(source, _contract())

    np.testing.assert_allclose(dataset.features[1, 0, 0], np.log(110.0 / 106.0))
    np.testing.assert_allclose(dataset.features[2, 0, 0], np.log(120.0 / 116.0))


def _cross_subbase_source() -> MemoryMultiTimeframeSource:
    hourly = [f"2026-01-01T{hour:02d}:00:00" for hour in range(3)]
    quarter = [
        f"2026-01-01T{index // 4:02d}:{(index % 4) * 15:02d}:00" for index in range(9)
    ]
    return MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _series(hourly, [100.0, 105.0, 111.0]),
            ("ETHUSDT", "1h"): _series(hourly, [200.0, 207.0, 220.0]),
            ("BTCUSDT", "15m"): _series(quarter, [100.0 + i for i in range(9)]),
            ("ETHUSDT", "15m"): _series(
                quarter, [200.0 + 3 * i + 0.2 * i * i for i in range(9)]
            ),
        }
    )


def _cross_subbase_contracts() -> tuple[InstrumentContract, ...]:
    return (
        InstrumentContract(
            symbol="BTCUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)
        ),
        InstrumentContract(
            symbol="ETHUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)
        ),
    )


@pytest.mark.parametrize(
    "kind",
    (
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    ),
)
def test_subbase_cross_asset_works_without_an_explicit_one_bar_source_channel(
    kind: FeatureKind,
) -> None:
    source = _cross_subbase_source()
    cross = FeatureSpec(
        name=f"15m_{kind.value}",
        kind=kind,
        timeframe="15m",
        lookback=2,
        min_periods=2,
        max_staleness_hours=2.0,
    )
    implicit = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(cross,),
            cross_asset_reference_symbol="BTCUSDT",
        )
    ).build(source, _cross_subbase_contracts())
    explicit = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="15m_ret_1",
                    kind=FeatureKind.LOG_RETURN,
                    timeframe="15m",
                    lookback=1,
                    max_staleness_hours=2.0,
                ),
                cross,
            ),
            cross_asset_reference_symbol="BTCUSDT",
        )
    ).build(source, _cross_subbase_contracts())

    np.testing.assert_array_equal(
        implicit.feature_available[:, :, 0], explicit.feature_available[:, :, 1]
    )
    np.testing.assert_allclose(
        implicit.features[:, :, 0], explicit.features[:, :, 1], rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        implicit.feature_staleness_hours[:, :, 0],
        explicit.feature_staleness_hours[:, :, 1],
    )
    assert implicit.feature_names == (cross.name,)
    assert implicit.feature_available[2, 1, 0]
    assert implicit.features[2, 1, 0] != 0.0


def test_subbase_cross_asset_requires_valid_native_source_not_base_price_fallback() -> (
    None
):
    source = _cross_subbase_source()
    source.values[("BTCUSDT", "15m")] = _series(
        [
            "2026-01-01T00:00:00",
            "2026-01-01T00:15:00",
            "2026-01-01T00:45:00",
            "2026-01-01T01:00:00",
        ],
        [100.0, 101.0, 104.0, 105.0],
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="relative",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
            ),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    with pytest.raises(ValueError, match="complete and exactly regular"):
        MarketDatasetBuilder(config).build(source, _cross_subbase_contracts())


def test_base_cross_asset_derives_a_private_return_without_a_public_source() -> None:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="relative", kind=FeatureKind.RELATIVE_RETURN_TO_BTC),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    result = MarketDatasetBuilder(config).build(
        _cross_subbase_source(), _cross_subbase_contracts()
    )

    assert result.feature_names == ("relative",)
    assert result.feature_available is not None
    np.testing.assert_array_equal(
        result.feature_available[:, 1, 0], [False, True, True]
    )
    np.testing.assert_allclose(
        result.features[1:, 1, 0],
        [
            np.log(207.0 / 200.0) - np.log(105.0 / 100.0),
            np.log(220.0 / 207.0) - np.log(111.0 / 105.0),
        ],
        rtol=1e-6,
    )


def test_subbase_cross_asset_refuses_missing_native_source_protocol() -> None:
    source = _cross_subbase_source()
    from trade_rl.data.source import InMemoryMarketDataSource

    base_only = InMemoryMarketDataSource(
        {symbol: source.values[(symbol, "1h")] for symbol in ("BTCUSDT", "ETHUSDT")}
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="relative",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
            ),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    with pytest.raises(ValueError, match="MultiTimeframeMarketDataSource"):
        MarketDatasetBuilder(config).build(base_only, _cross_subbase_contracts())


def test_subbase_cross_asset_obeys_delayed_native_source_available_at() -> None:
    source = _cross_subbase_source()
    original = source.values[("ETHUSDT", "15m")]
    native_times = [str(time.astype("datetime64[s]")) for time in original.timestamps]
    late_times = [
        str((time + np.timedelta64(3, "h")).astype("datetime64[s]"))
        for time in original.timestamps
    ]
    source.values[("ETHUSDT", "15m")] = _series(
        native_times,
        [float(price) for price in original.close],
        available_at=late_times,
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="relative",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
            ),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    result = MarketDatasetBuilder(config).build(source, _cross_subbase_contracts())
    np.testing.assert_array_equal(
        result.feature_available[:, 1, 0], [False, False, False]
    )
    assert result.feature_available[1, 0, 0]


def test_subbase_cross_asset_ignores_duplicate_public_return_channels() -> None:
    relative = FeatureSpec(
        name="relative",
        kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
        timeframe="15m",
    )
    duplicated = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret_a", kind=FeatureKind.LOG_RETURN, timeframe="15m"),
            FeatureSpec(name="ret_b", kind=FeatureKind.LOG_RETURN, timeframe="15m"),
            relative,
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    alone = MarketBuildConfig(
        base_timeframe="1h",
        features=(relative,),
        cross_asset_reference_symbol="BTCUSDT",
    )
    with_duplicates = MarketDatasetBuilder(duplicated).build(
        _cross_subbase_source(), _cross_subbase_contracts()
    )
    without = MarketDatasetBuilder(alone).build(
        _cross_subbase_source(), _cross_subbase_contracts()
    )

    np.testing.assert_array_equal(
        with_duplicates.features[..., 2], without.features[..., 0]
    )
    assert with_duplicates.feature_available is not None
    assert without.feature_available is not None
    np.testing.assert_array_equal(
        with_duplicates.feature_available[..., 2], without.feature_available[..., 0]
    )


def test_subbase_cross_asset_missing_reference_is_rejected_before_native_load() -> None:
    source = _cross_subbase_source()
    original = source.load_timeframe

    def guarded(symbol: str, timeframe: str) -> RawMarketSeries:
        if timeframe == "15m":
            pytest.fail("invalid cross-asset config must not load extra native source")
        return original(symbol, timeframe)

    source.load_timeframe = guarded  # type: ignore[method-assign]
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="relative",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
            ),
        ),
    )
    with pytest.raises(ValueError, match="cross_asset_reference_symbol"):
        MarketDatasetBuilder(config).build(source, _cross_subbase_contracts())


def test_implicit_subbase_cross_asset_is_prefix_causal_under_future_mutation() -> None:
    original = _cross_subbase_source()
    changed = _cross_subbase_source()
    original_native = changed.values[("ETHUSDT", "15m")]
    changed_prices = original_native.close.copy()
    changed_prices[-1] *= 1.4
    native_times = [
        str(time.astype("datetime64[s]")) for time in original_native.timestamps
    ]
    changed.values[("ETHUSDT", "15m")] = _series(
        native_times,
        [float(price) for price in changed_prices],
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="relative",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="15m",
                max_staleness_hours=2.0,
            ),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    before = MarketDatasetBuilder(config).build(original, _cross_subbase_contracts())
    after = MarketDatasetBuilder(config).build(changed, _cross_subbase_contracts())
    np.testing.assert_array_equal(
        before.feature_available[:2], after.feature_available[:2]
    )
    np.testing.assert_array_equal(before.features[:2], after.features[:2])
    assert before.features[2, 1, 0] != after.features[2, 1, 0]


@pytest.mark.parametrize(("timeframe", "steps"), [("1h", 1), ("15m", 4)])
def test_private_return_detects_new_close_at_same_elapsed_age(
    timeframe: str, steps: int
) -> None:
    # At decisions 1h, 2h, 3h and 4h the latest ETH close is 1h, 1h,
    # 2h and 4h respectively. The new 2h close must be consumed at 3h
    # even though its elapsed age is the same as at the previous decision.
    indices = np.arange(8 * steps + 1)
    times = np.datetime64("2026-01-01T00:00:00", "ns") + indices * np.timedelta64(
        60 // steps, "m"
    )
    receipts = times.copy()
    receipts[(indices > steps) & (indices <= 2 * steps)] = times[3 * steps]
    receipts[(indices > 2 * steps) & (indices <= 3 * steps)] = times[4 * steps]
    timestamps = np.datetime_as_string(times, unit="s").tolist()
    published = np.datetime_as_string(receipts, unit="s").tolist()
    eth_closes = 100.0 + indices.astype(np.float64) ** 2
    native = {
        ("BTCUSDT", timeframe): _series(timestamps, [100.0] * len(indices)),
        ("ETHUSDT", timeframe): _series(
            timestamps, eth_closes.tolist(), available_at=published
        ),
    }
    source = MemoryMultiTimeframeSource(
        {("BTCUSDT", "1h"): _base(), ("ETHUSDT", "1h"): _base(), **native}
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                "rel",
                FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe=None if timeframe == "1h" else timeframe,
                max_staleness_hours=4.0,
            ),
        ),
        cross_asset_reference_symbol="BTCUSDT",
    )
    contracts = tuple(
        InstrumentContract(symbol, listed_at=datetime(2025, 1, 1, tzinfo=UTC))
        for symbol in ("BTCUSDT", "ETHUSDT")
    )
    dataset = MarketDatasetBuilder(config).build(source, contracts)
    assert dataset.feature_available is not None
    assert dataset.feature_staleness_hours is not None
    assert dataset.feature_available[3, 1, 0]
    expected = np.log(eth_closes[2 * steps] / eth_closes[2 * steps - 1])
    assert dataset.features[3, 1, 0] == pytest.approx(expected, abs=1e-7)
    assert dataset.feature_staleness_hours[3, 1, 0] == 1.0
