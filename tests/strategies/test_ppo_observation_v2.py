from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.rl import ppo
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _market(
    *,
    symbol: str = "BTCUSDT",
    global_names: tuple[str, ...] = (
        "market_return_dispersion",
        "active_fraction",
        "market_return_mean",
        "tradable_fraction",
    ),
    future_shift: float = 0.0,
) -> MarketDataset:
    close = np.asarray([[100.0], [101.0], [102.0], [103.0]])
    features = np.asarray(
        [
            [[10.0, 20.0]],
            [[11.0, 21.0]],
            [[12.0 + future_shift, 22.0 + future_shift]],
            [[13.0 + future_shift, 23.0 + future_shift]],
        ],
        dtype=np.float32,
    )
    feature_available = np.ones_like(features, dtype=np.bool_)
    feature_available[0, 0, 1] = False
    feature_staleness = np.asarray(
        [
            [[0.25, 1.0]],
            [[0.0, 0.5]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    base_global_by_name = {
        "active_fraction": 0.8,
        "tradable_fraction": 0.7,
        "market_return_mean": 0.1,
        "market_return_dispersion": 0.4,
    }
    global_features = np.asarray(
        [
            [base_global_by_name[name] for name in global_names],
            [base_global_by_name[name] + 0.01 for name in global_names],
            [base_global_by_name[name] + future_shift for name in global_names],
            [base_global_by_name[name] + future_shift for name in global_names],
        ],
        dtype=np.float32,
    )
    global_available = np.ones_like(global_features, dtype=np.bool_)
    if "market_return_mean" in global_names:
        global_available[0, global_names.index("market_return_mean")] = False

    return MarketDataset(
        dataset_id="9" * 64,
        symbols=(symbol,),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=features,
        global_features=global_features,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 1), 1_000_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("fast", "slow"),
        global_feature_names=global_names,
        periods_per_year=8_760,
        feature_staleness=feature_staleness,
        global_feature_available=global_available,
    )


def _env(dataset: MarketDataset) -> PPOTradingEnv:
    return PPOTradingEnv(
        dataset,
        feature_indices=(1, 0),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )


def test_observation_v2_contract_is_fixed_before_m2() -> None:
    assert ppo.PPO_OBSERVATION_SCHEMA == "ppo_observation_v2"
    assert ppo.PPO_GLOBAL_FEATURE_NAMES == (
        "active_fraction",
        "tradable_fraction",
        "market_return_mean",
        "market_return_dispersion",
    )


def test_observation_v2_encodes_exact_order_masks_and_staleness() -> None:
    observation, _ = _env(_market()).reset(seed=7)

    expected = np.asarray(
        [
            # selected local values: slow is unavailable, fast is usable
            0.0,
            10.0,
            # local availability / finite mask
            0.0,
            1.0,
            # selected normalized staleness in the same order
            1.0,
            0.25,
            # fixed global contract order, not dataset storage order
            0.8,
            0.7,
            0.0,
            0.4,
            # global availability
            1.0,
            1.0,
            0.0,
            1.0,
            # portfolio state
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(observation, expected)
    assert observation.shape == (16,)
    assert observation.flags.writeable is True  # Gym receives its own mutable copy.


def test_observation_v2_fails_closed_when_global_contract_is_missing() -> None:
    dataset = _market(
        global_names=(
            "active_fraction",
            "tradable_fraction",
            "market_return_mean",
        )
    )

    with pytest.raises(ValueError, match="missing PPO global feature"):
        _env(dataset)


def test_observation_v2_is_symbol_name_invariant() -> None:
    first, _ = _env(_market(symbol="BTCUSDT")).reset(seed=3)
    renamed, _ = _env(_market(symbol="RENAMED")).reset(seed=3)

    np.testing.assert_array_equal(first, renamed)


def test_observation_v2_does_not_read_future_rows() -> None:
    baseline, _ = _env(_market(future_shift=0.0)).reset(seed=5)
    mutated_future, _ = _env(_market(future_shift=999.0)).reset(seed=5)

    np.testing.assert_array_equal(baseline, mutated_future)
