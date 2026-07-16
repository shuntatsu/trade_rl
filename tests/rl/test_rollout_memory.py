from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.rl.rollout_memory import (
    estimate_index_backed_ppo_rollout_buffer_bytes,
    estimate_ppo_rollout_buffer_bytes,
)
from trade_rl.rl.training import ResidualTrainingConfig


def test_dict_rollout_memory_matches_sb3_float32_storage_contract() -> None:
    observation_space = spaces.Dict(
        {
            "a": spaces.Box(-1, 1, shape=(2, 3), dtype=np.float16),
            "b": spaces.Box(0, 1, shape=(5,), dtype=np.uint8),
        }
    )
    # SB3 DictRolloutBuffer coerces every observation array to float32.
    elements_per_step = 6 + 5 + 3 + 6  # observations + actions + six scalar arrays
    assert (
        estimate_ppo_rollout_buffer_bytes(
            observation_space,
            n_steps=4,
            n_envs=2,
            action_dim=3,
        )
        == elements_per_step * 4 * 2 * 4
    )


def test_training_config_rejects_invalid_rollout_memory_cap() -> None:
    with pytest.raises(ValueError, match="max_rollout_buffer_bytes"):
        ResidualTrainingConfig(
            timesteps=8,
            gamma=0.99,
            seeds=(0,),
            n_steps=8,
            batch_size=8,
            max_rollout_buffer_bytes=0,
        )


def test_production_sequence_rollout_fits_configured_memory_cap() -> None:
    counts = {"15m": 59, "1h": 59, "4h": 55, "1d": 53}
    lengths = {"15m": 96, "1h": 168, "4h": 120, "1d": 60}
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -np.inf, np.inf, shape=(3, 4 * 226), dtype=np.float32
        ),
        "asset_state": spaces.Box(-np.inf, np.inf, shape=(3, 18), dtype=np.float32),
        "global_state": spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
    }
    for timeframe, count in counts.items():
        shape = (3, lengths[timeframe], count)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -np.inf, np.inf, shape=shape, dtype=np.float16
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 1.0, shape=shape, dtype=np.float16
        )
    space = spaces.Dict(observation_spaces)
    default_estimated = estimate_ppo_rollout_buffer_bytes(
        space, n_steps=128, n_envs=4, action_dim=3
    )
    space.spaces["decision_index"] = spaces.Box(0, 100_000, shape=(1,), dtype=np.int64)
    compact_estimated = estimate_index_backed_ppo_rollout_buffer_bytes(
        space, n_steps=128, n_envs=4, action_dim=3
    )
    assert default_estimated == 473_122_816
    assert compact_estimated == 5_765_120
    assert compact_estimated < default_estimated / 50
    assert compact_estimated < 805_306_368


def test_index_backed_rollout_buffer_does_not_allocate_sequence_arrays() -> None:
    from trade_rl.integrations.compact_rollout_buffer import (
        IndexBackedDictRolloutBuffer,
        SequenceRolloutReconstructor,
    )
    from trade_rl.rl.sequence_observations import (
        SequenceObservationBuilder,
        SequenceWindowSpec,
    )

    dataset = _sequence_dataset()
    builder = SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 4),
            SequenceWindowSpec("1h", 3),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 1),
        )
    )
    observation_space = _sequence_observation_space()
    reconstructor = SequenceRolloutReconstructor(
        dataset=dataset,
        builder=builder,
        normalizer=None,
        expected_dataset_id=dataset.dataset_id,
        expected_layout_digest=builder.layout_digest(dataset),
    )
    buffer = IndexBackedDictRolloutBuffer(
        4,
        observation_space,
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=2,
        sequence_reconstructor=reconstructor,
    )

    assert "decision_index" in buffer.observations
    assert buffer.observations["decision_index"].dtype == np.int64
    assert not any(key.startswith("sequence_") for key in buffer.observations)
    assert buffer.observations["current_snapshot"].dtype == np.float32


def test_sequence_rollout_reconstructor_matches_direct_causal_builder() -> None:
    from trade_rl.integrations.compact_rollout_buffer import (
        SequenceRolloutReconstructor,
    )
    from trade_rl.rl.sequence_observations import (
        SequenceObservationBuilder,
        SequenceWindowSpec,
        sequence_policy_values,
    )

    dataset = _sequence_dataset()
    builder = SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 4),
            SequenceWindowSpec("1h", 3),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 1),
        )
    )
    reconstructor = SequenceRolloutReconstructor(
        dataset=dataset,
        builder=builder,
        normalizer=None,
        expected_dataset_id=dataset.dataset_id,
        expected_layout_digest=builder.layout_digest(dataset),
    )

    indices = np.array([64, 68], dtype=np.int64)
    batch = reconstructor.reconstruct(indices)
    for batch_index, decision_index in enumerate(indices):
        direct = builder.build(dataset, index=int(decision_index))
        for timeframe in ("15m", "1h", "4h", "1d"):
            expected = sequence_policy_values(
                timeframe=timeframe,
                values=direct.values[timeframe],
                available=direct.available[timeframe],
                feature_names=direct.feature_names[timeframe],
            )
            np.testing.assert_array_equal(
                batch[f"sequence_{timeframe}_values"][batch_index], expected
            )
            np.testing.assert_array_equal(
                batch[f"sequence_{timeframe}_available"][batch_index],
                direct.available[timeframe].astype(np.uint8),
            )


def test_sequence_rollout_reconstructor_fails_closed_on_identity_mismatch() -> None:
    from trade_rl.integrations.compact_rollout_buffer import (
        SequenceRolloutReconstructor,
    )
    from trade_rl.rl.sequence_observations import (
        SequenceObservationBuilder,
        SequenceWindowSpec,
    )

    dataset = _sequence_dataset()
    builder = SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 4),
            SequenceWindowSpec("1h", 3),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 1),
        )
    )
    with pytest.raises(ValueError, match="dataset identity"):
        SequenceRolloutReconstructor(
            dataset=dataset,
            builder=builder,
            normalizer=None,
            expected_dataset_id="f" * 64,
            expected_layout_digest=builder.layout_digest(dataset),
        )


def _sequence_dataset():
    from trade_rl.data.market import MarketDataset

    n = 96
    names = ("15m__ret", "1h__ret", "4h__ret", "1d__ret")
    timestamps = np.datetime64("2026-01-01T00:15", "ns") + np.arange(
        n
    ) * np.timedelta64(15, "m")
    features = np.zeros((n, 2, len(names)), dtype=np.float32)
    for index in range(n):
        features[index] = index + np.arange(len(names), dtype=np.float32)
    close = 100.0 + np.arange(n, dtype=np.float64)[:, None] + np.arange(2)[None, :]
    open_price = np.vstack((close[:1], close[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_staleness_hours=np.zeros_like(features, dtype=np.float32),
        feature_names=names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _sequence_observation_space() -> spaces.Dict:
    result: dict[str, spaces.Space] = {
        "decision_index": spaces.Box(0, 100_000, shape=(1,), dtype=np.int64),
        "current_snapshot": spaces.Box(
            -np.inf, np.inf, shape=(2, 16), dtype=np.float32
        ),
        "asset_state": spaces.Box(-np.inf, np.inf, shape=(2, 4), dtype=np.float32),
        "global_state": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
    }
    lengths = {"15m": 4, "1h": 3, "4h": 2, "1d": 1}
    for timeframe, length in lengths.items():
        shape = (2, length, 1)
        result[f"sequence_{timeframe}_values"] = spaces.Box(
            -np.inf, np.inf, shape=shape, dtype=np.float16
        )
        result[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        result[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0, np.inf, shape=shape, dtype=np.float16
        )
    return spaces.Dict(result)


def test_index_backed_buffer_requires_runtime_reconstructor_only_when_sampling() -> (
    None
):
    from trade_rl.integrations.compact_rollout_buffer import (
        IndexBackedDictRolloutBuffer,
    )

    buffer = IndexBackedDictRolloutBuffer(
        2,
        _sequence_observation_space(),
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=1,
    )
    assert buffer.sequence_reconstructor is None
    with pytest.raises(RuntimeError, match="reconstructor"):
        buffer._get_samples(np.array([0], dtype=np.int64))
