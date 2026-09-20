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
from trade_rl.data.source import RawMarketSeries
from trade_rl.strategies.dataset_scope import validated_training_scope
from trade_rl.strategies.rl.ppo import PPOTradingEnv

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_LOG_STEPS = {
    "BTCUSDT": {"1h": 0.01, "4h": 0.10, "1d": 0.40},
    "ETHUSDT": {"1h": 0.03, "4h": 0.30, "1d": 0.90},
}


class _MemoryMultiTimeframeSource:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self._values = values

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        return self._values[(symbol, timeframe)]


def _series(hours: tuple[int, ...], *, log_step: float) -> RawMarketSeries:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.asarray(
        hours
    ) * np.timedelta64(1, "h")
    close = np.exp(log_step * np.arange(len(hours), dtype=np.float64))
    open_price = np.concatenate((close[:1], close[:-1]))
    return RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full(len(hours), 100.0),
        funding_rate=np.zeros(len(hours)),
        funding_available=np.zeros(len(hours), dtype=np.bool_),
        tradable=np.ones(len(hours), dtype=np.bool_),
    )


def _source(*, missing_btc_hours: tuple[int, ...] = ()) -> _MemoryMultiTimeframeSource:
    values: dict[tuple[str, str], RawMarketSeries] = {}
    for symbol in _SYMBOLS:
        base_hours = tuple(
            hour
            for hour in range(29)
            if not (
                (symbol == "ETHUSDT" and hour == 25)
                or (symbol == "BTCUSDT" and hour in missing_btc_hours)
            )
        )
        values[(symbol, "1h")] = _series(
            base_hours,
            log_step=_LOG_STEPS[symbol]["1h"],
        )
        values[(symbol, "4h")] = _series(
            tuple(range(0, 29, 4)),
            log_step=_LOG_STEPS[symbol]["4h"],
        )
        values[(symbol, "1d")] = _series(
            (0, 24),
            log_step=_LOG_STEPS[symbol]["1d"],
        )
    return _MemoryMultiTimeframeSource(values)


def _contracts(symbols: tuple[str, ...]) -> tuple[InstrumentContract, ...]:
    return tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for symbol in symbols
    )


def _dataset(
    symbols: tuple[str, ...] = _SYMBOLS,
    *,
    source: _MemoryMultiTimeframeSource | None = None,
):
    return MarketDatasetBuilder(_config()).build(
        source or _source(),
        _contracts(symbols),
    )


def _with_future_prices_changed(
    source: _MemoryMultiTimeframeSource,
    *,
    cutoff_hour: int,
) -> _MemoryMultiTimeframeSource:
    epoch = np.datetime64("2026-01-01T00:00:00", "ns")
    changed: dict[tuple[str, str], RawMarketSeries] = {}
    for key, raw in source._values.items():
        hours = ((raw.timestamps - epoch) / np.timedelta64(1, "h")).astype(np.int64)
        future = hours >= cutoff_hour
        open_price = raw.open.copy()
        close = raw.close.copy()
        open_price[future] *= 7.0
        close[future] *= 7.0
        changed[key] = RawMarketSeries(
            timestamps=raw.timestamps,
            open=open_price,
            high=np.maximum(open_price, close),
            low=np.minimum(open_price, close),
            close=close,
            volume=raw.volume,
            funding_rate=raw.funding_rate,
            funding_available=raw.funding_available,
            funding_event_count=raw.funding_event_count,
            tradable=raw.tradable,
            available_at=raw.available_at,
        )
    return _MemoryMultiTimeframeSource(changed)


def _config() -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        cross_asset_reference_symbol="BTCUSDT",
        features=(
            FeatureSpec(
                name="ret_1h",
                kind=FeatureKind.LOG_RETURN,
                lookback=1,
                max_staleness_hours=1.0,
            ),
            FeatureSpec(
                name="rel_1h",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                lookback=1,
                min_periods=1,
                max_staleness_hours=1.0,
            ),
            FeatureSpec(
                name="ret_4h",
                kind=FeatureKind.LOG_RETURN,
                timeframe="4h",
                lookback=1,
                max_staleness_hours=4.0,
            ),
            FeatureSpec(
                name="rel_4h",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="4h",
                lookback=1,
                min_periods=1,
                max_staleness_hours=4.0,
            ),
            FeatureSpec(
                name="ret_1d",
                kind=FeatureKind.LOG_RETURN,
                timeframe="1d",
                lookback=1,
                max_staleness_hours=24.0,
            ),
            FeatureSpec(
                name="rel_1d",
                kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe="1d",
                lookback=1,
                min_periods=1,
                max_staleness_hours=24.0,
            ),
        ),
    )


def test_btc_relative_multitimeframe_observation_matches_analytical_oracle() -> None:
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for symbol in _SYMBOLS
    )
    dataset = MarketDatasetBuilder(_config()).build(_source(), contracts)
    relative_indices = (1, 3, 5)
    rows = np.asarray((23, 24, 25, 26, 27))

    native_relative = np.asarray(
        [
            _LOG_STEPS["ETHUSDT"][timeframe] - _LOG_STEPS["BTCUSDT"][timeframe]
            for timeframe in ("1h", "4h", "1d")
        ],
        dtype=np.float32,
    )
    expected_values = np.asarray(
        [
            [native_relative[0], native_relative[1], 0.0],
            native_relative,
            native_relative,
            [0.0, native_relative[1], native_relative[2]],
            native_relative,
        ],
        dtype=np.float32,
    )
    expected_available = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
            [False, True, True],
            [True, True, True],
        ],
        dtype=np.bool_,
    )
    expected_age_hours = np.asarray(
        [
            [0.0, 3.0, 24.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    expected_staleness = np.asarray(
        [
            [0.0, 0.75, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.25, 1.0 / 24.0],
            [1.0, 0.50, 2.0 / 24.0],
            [0.0, 0.75, 3.0 / 24.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        dataset.features[rows, 1][:, relative_indices],
        expected_values,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        dataset.feature_available[rows, 1][:, relative_indices],
        expected_available,
    )
    assert dataset.feature_staleness_hours is not None
    assert dataset.feature_staleness is not None
    np.testing.assert_allclose(
        dataset.feature_staleness_hours[rows, 1][:, relative_indices],
        expected_age_hours,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        dataset.feature_staleness[rows, 1][:, relative_indices],
        expected_staleness,
        rtol=0.0,
        atol=0.0,
    )

    assert validated_training_scope(
        dataset,
        feature_indices=relative_indices,
        fit_symbol_indices=(0, 1),
    ) == (relative_indices, (0, 1))
    with pytest.raises(
        ValueError,
        match="reference symbol must belong to fit symbol scope",
    ):
        validated_training_scope(
            dataset,
            feature_indices=relative_indices,
            fit_symbol_indices=(1,),
        )

    environment = PPOTradingEnv(
        dataset,
        feature_indices=relative_indices,
        symbol_indices=(1,),
        information_symbol_indices=(0, 1),
        start_index=25,
        stop_index=27,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    observation, info = environment.reset(seed=7)
    expected_observation = np.concatenate(
        (
            native_relative,
            np.ones(3, dtype=np.float32),
            expected_staleness[2],
            np.zeros(2, dtype=np.float32),
        )
    )

    np.testing.assert_allclose(
        observation,
        expected_observation,
        rtol=0.0,
        atol=1e-7,
    )
    assert observation.shape == (11,)
    assert info == {"symbol_index": 1, "symbol": "ETHUSDT"}


def test_btc_relative_features_are_invariant_to_symbol_order() -> None:
    forward = _dataset(_SYMBOLS)
    reversed_dataset = _dataset(tuple(reversed(_SYMBOLS)))
    relative_indices = (1, 3, 5)

    for symbol in _SYMBOLS:
        forward_index = forward.symbols.index(symbol)
        reversed_index = reversed_dataset.symbols.index(symbol)
        np.testing.assert_allclose(
            forward.features[:, forward_index][:, relative_indices],
            reversed_dataset.features[:, reversed_index][:, relative_indices],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            forward.feature_available[:, forward_index][:, relative_indices],
            reversed_dataset.feature_available[:, reversed_index][:, relative_indices],
        )
        assert forward.feature_staleness is not None
        assert reversed_dataset.feature_staleness is not None
        np.testing.assert_allclose(
            forward.feature_staleness[:, forward_index][:, relative_indices],
            reversed_dataset.feature_staleness[:, reversed_index][:, relative_indices],
            rtol=0.0,
            atol=0.0,
        )


def test_missing_btc_event_expires_relative_feature_and_v2_masks_it() -> None:
    dataset = _dataset(source=_source(missing_btc_hours=(25, 26)))
    eth_index = dataset.symbols.index("ETHUSDT")
    row = 26

    assert not dataset.feature_available[row, eth_index, 1]
    assert dataset.features[row, eth_index, 1] == 0.0

    environment = PPOTradingEnv(
        dataset,
        feature_indices=(1, 3, 5),
        symbol_indices=(eth_index,),
        information_symbol_indices=(0, eth_index),
        start_index=row,
        stop_index=row + 1,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    observation, _ = environment.reset(seed=7)

    # PPO Observation v2 is values, availability masks, staleness, then state.
    assert observation[0] == 0.0
    assert observation[3] == 0.0
    assert observation[6] == 1.0
    assert np.isfinite(observation).all()


def test_future_source_price_changes_do_not_change_past_relative_features() -> None:
    cutoff_hour = 12
    original = _dataset()
    changed = _dataset(
        source=_with_future_prices_changed(_source(), cutoff_hour=cutoff_hour)
    )
    before_cutoff = original.timestamps < np.datetime64(
        "2026-01-01T00:00:00", "ns"
    ) + np.timedelta64(cutoff_hour, "h")
    relative_indices = (1, 3, 5)

    np.testing.assert_allclose(
        original.features[before_cutoff][:, :, relative_indices],
        changed.features[before_cutoff][:, :, relative_indices],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        original.feature_available[before_cutoff][:, :, relative_indices],
        changed.feature_available[before_cutoff][:, :, relative_indices],
    )
    assert original.feature_staleness is not None
    assert changed.feature_staleness is not None
    np.testing.assert_allclose(
        original.feature_staleness[before_cutoff][:, :, relative_indices],
        changed.feature_staleness[before_cutoff][:, :, relative_indices],
        rtol=0.0,
        atol=0.0,
    )
