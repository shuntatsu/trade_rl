from __future__ import annotations

import pytest

from trade_rl.rl.episode import EpisodeRange, resolve_episode_range


def test_episode_range_reserves_reward_preroll() -> None:
    value = resolve_episode_range(
        requested_start=200,
        episode_bars=60,
        reward_preroll_bars=180,
        dataset_bars=500,
    )
    assert value == EpisodeRange(start=20, reward_start=200, stop=260)


def test_episode_range_rejects_missing_preroll() -> None:
    with pytest.raises(ValueError, match="pre-roll"):
        resolve_episode_range(
            requested_start=100,
            episode_bars=60,
            reward_preroll_bars=180,
            dataset_bars=500,
        )


def test_episode_range_rejects_episode_outside_dataset() -> None:
    with pytest.raises(ValueError, match="fit"):
        resolve_episode_range(
            requested_start=470,
            episode_bars=60,
            reward_preroll_bars=180,
            dataset_bars=500,
        )


def _hourly_dataset(n_bars: int):
    import numpy as np

    from trade_rl.data.market import MarketDataset

    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.ones((n_bars, 1), dtype=np.float64)
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("BTC",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones((n_bars, 1), dtype=np.float64),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_minimum_reward_start_reserves_signal_history_and_full_window() -> None:
    from trade_rl.rl.episode import minimum_reward_start_index

    assert (
        minimum_reward_start_index(
            _hourly_dataset(800), signal_minimum=10, window_hours=720.0
        )
        == 730
    )


def test_minimum_reward_start_rejects_dataset_without_full_window() -> None:
    from trade_rl.rl.episode import minimum_reward_start_index

    with pytest.raises(ValueError, match="complete reward pre-roll"):
        minimum_reward_start_index(
            _hourly_dataset(700), signal_minimum=10, window_hours=720.0
        )
