from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import (
    OracleEpisodeSamplingConfig,
    build_episode_oracle_batch,
    sample_oracle_episode_contracts,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig


def _market(n_bars: int = 24) -> MarketDataset:
    close = np.linspace(100.0, 110.0, n_bars, dtype=np.float64)[:, None]
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def test_episode_contract_sampling_respects_explicit_maximum_stop_index() -> None:
    market = _market()
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=4,
        episode_count=100,
        initial_state_modes=("cash",),
        seed=17,
    )

    contracts = sample_oracle_episode_contracts(
        market,
        minimum_start_index=3,
        maximum_stop_index=12,
        config=sampling,
    )

    assert contracts
    assert min(contract.start for contract in contracts) >= 3
    assert max(contract.stop for contract in contracts) <= 12


def test_episode_oracle_batch_respects_explicit_train_stop() -> None:
    market = _market()
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=3,
        episode_count=32,
        initial_state_modes=("cash",),
        seed=23,
    )

    batch = build_episode_oracle_batch(
        market,
        minimum_start_index=4,
        maximum_stop_index=14,
        sampling_config=sampling,
        teacher_config=OracleTeacherConfig(),
    )

    assert min(contract.start for contract in batch.contracts) >= 4
    assert max(contract.stop for contract in batch.contracts) <= 14
