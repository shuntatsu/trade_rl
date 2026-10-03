from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.features.price_channels import CHANNEL_NAMES
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.bot import (
    BotConfig,
    create_strategy_instances,
    generate_demo_dataset,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def _named_channel_dataset() -> MarketDataset:
    prices = np.full((4, 1), 100.0)
    names = (
        "unrelated_momentum",
        CHANNEL_NAMES[3],
        CHANNEL_NAMES[0],
        CHANNEL_NAMES[1],
        CHANNEL_NAMES[2],
    )
    # The close lies inside both channels despite positive unrelated momentum.
    values = np.array([0.2, 0.1, -0.1, 0.1, -0.1], dtype=np.float32)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.broadcast_to(values, (4, 1, 5)).copy(),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full_like(prices, 10_000.0),
        funding_rate=np.zeros_like(prices),
        tradable=np.ones_like(prices, dtype=np.bool_),
        feature_available=np.ones((4, 1, 5), dtype=np.bool_),
        feature_names=names,
        global_feature_names=("global",),
        periods_per_year=8_760,
    )


@pytest.mark.parametrize(
    ("values", "expected_intent"),
    [
        ([0.2, 0.1, -0.1, 0.1, -0.1], PositionIntent.FLAT),
        ([0.2, 0.1, 0.02, 0.1, -0.1], PositionIntent.LONG),
        ([0.2, 0.1, -0.1, -0.02, -0.1], PositionIntent.SHORT),
    ],
)
def test_bot_breakout_resolves_named_channels_in_any_column_order(
    values: list[float], expected_intent: PositionIntent
) -> None:
    dataset = _named_channel_dataset()
    strategy = create_strategy_instances(
        dataset, BotConfig(strategy_name="channel_breakout")
    )[0]
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[0],
        features=np.asarray(values, dtype=np.float32),
        feature_available=dataset.feature_available[0, 0],
        global_features=dataset.global_features[0],
        global_feature_available=np.ones(1, dtype=np.bool_),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )

    assert strategy.decide(observation) is expected_intent


def test_bot_breakout_rejects_missing_named_channels_before_replay() -> None:
    dataset = _named_channel_dataset()
    missing = replace(
        dataset,
        feature_names=tuple(
            "unrelated" if name == CHANNEL_NAMES[0] else name
            for name in dataset.feature_names
        ),
    )
    with pytest.raises(ValueError, match="channel_entry_upper"):
        create_strategy_instances(missing, BotConfig(strategy_name="channel_breakout"))


def test_demo_channels_use_prior_candles_and_separate_entry_exit_windows() -> None:
    dataset = generate_demo_dataset(n_bars=32, n_symbols=2, seed=23)
    assert dataset.identity_payload_json is None
    assert dataset.feature_names == ("ema_crossover",) + CHANNEL_NAMES
    assert not dataset.feature_available[:20, :, 1:3].any()
    assert not dataset.feature_available[:10, :, 3:5].any()
    assert dataset.feature_available[20, :, 1:].all()

    for column, window, price_field in (
        (1, 20, dataset.high),
        (2, 20, dataset.low),
        (3, 10, dataset.high),
        (4, 10, dataset.low),
    ):
        previous = price_field[20 - window : 20]
        boundary = previous.max(axis=0) if column in (1, 3) else previous.min(axis=0)
        expected = (dataset.close[20] / boundary - 1.0).astype(np.float32)
        np.testing.assert_array_equal(dataset.features[20, :, column], expected)


def test_demo_short_fixture_keeps_valid_channel_windows() -> None:
    dataset = generate_demo_dataset(n_bars=3, n_symbols=1)
    assert dataset.feature_names == ("ema_crossover",) + CHANNEL_NAMES
    assert not dataset.feature_available[:2, 0, 1:3].any()
    assert dataset.feature_available[2, 0, 1:].all()


@pytest.mark.parametrize("n_bars", [0, 1, 2, True, 3.5])
def test_demo_rejects_invalid_bar_count(n_bars: object) -> None:
    with pytest.raises(ValueError, match="n_bars"):
        generate_demo_dataset(n_bars=n_bars)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_symbols", [0, 4, True, 1.5])
def test_demo_rejects_unsupported_symbol_count(n_symbols: object) -> None:
    with pytest.raises(ValueError, match="n_symbols"):
        generate_demo_dataset(n_symbols=n_symbols)  # type: ignore[arg-type]
