from __future__ import annotations

from pathlib import Path

import numpy as np

from trade_rl.learning.teacher_artifact import (
    SupervisedPolicyDataset,
    load_teacher_artifact,
    write_teacher_artifact,
)


def test_structured_teacher_artifact_round_trip_binds_key_order_and_shapes(
    tmp_path: Path,
) -> None:
    observations = {
        "active": np.ones((4, 3), dtype=np.float32),
        "sequence_15m_values": np.arange(4 * 3 * 2 * 5, dtype=np.float32).reshape(
            4, 3, 2, 5
        ),
    }
    dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=np.zeros((4, 3), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=5,
        train_stop=10,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )

    write_teacher_artifact(tmp_path, dataset)
    manifest, loaded = load_teacher_artifact(tmp_path)

    assert manifest.observation_keys == ("active", "sequence_15m_values")
    assert manifest.observation_shapes == {
        "active": (4, 3),
        "sequence_15m_values": (4, 3, 2, 5),
    }
    assert isinstance(loaded.observations, dict)
    for key in observations:
        np.testing.assert_array_equal(loaded.observations[key], observations[key])


def _structured_environment():
    from trade_rl.data.market import MarketDataset
    from trade_rl.rl.actions import ActionSpec
    from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
    from trade_rl.strategies.trend import TrendConfig, TrendStrategy

    n_bars = 160
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(phase * 0.0001))[:, None]
    feature_names = (
        "15m__return",
        "1h__return",
        "4h__return",
        "1d__return",
    )
    dataset = MarketDataset(
        dataset_id="e" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:15:00", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.stack(
            tuple(np.sin(phase / divisor) for divisor in (3.0, 5.0, 7.0, 11.0)),
            axis=1,
        )[:, None, :].astype(np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=feature_names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=1, base_lookback=2, slow_lookback=3)
        ),
        action_spec=ActionSpec(
            mode="target_weight",
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=4,
            decision_every=1,
            structured_sequence_observation=True,
            sequence_windows=(("15m", 4), ("1h", 3), ("4h", 2), ("1d", 2)),
        ),
    )


def test_structured_rollout_stores_compact_state_and_reconstructs_exact_sequences() -> (
    None
):
    from trade_rl.learning.teacher_artifact import (
        StructuredTeacherObservationProvider,
        collect_teacher_rollout,
    )

    targets = np.zeros((4, 1), dtype=np.float32)
    environment = _structured_environment()
    start = environment.minimum_start_index
    supervised = collect_teacher_rollout(
        environment,
        targets,
        dataset_id=environment.dataset.dataset_id,
        train_range=(start, start + 5),
        teacher_config_digest="d" * 64,
    )
    assert isinstance(supervised.observations, dict)
    assert not any(key.startswith("sequence_") for key in supervised.observations)
    assert set(supervised.observations) == {
        "active",
        "asset_state",
        "current_snapshot",
        "decision_index",
        "global_state",
    }

    replay = _structured_environment()
    observation, _ = replay.reset(
        options={"start_idx": start, "episode_bars": 4, "initial_state_mode": "cash"}
    )
    direct: dict[str, list[np.ndarray]] = {}
    for target in targets:
        assert isinstance(observation, dict)
        for key, value in observation.items():
            direct.setdefault(key, []).append(np.asarray(value).copy())
        observation, *_ = replay.step(target)

    assert replay.sequence_observation_builder is not None
    provider = StructuredTeacherObservationProvider(
        dataset=replay.dataset,
        sequence_builder=replay.sequence_observation_builder,
        observations=supervised.observations,
    )
    rebuilt = provider.get(np.arange(4))
    assert "decision_index" not in rebuilt
    for key, values in direct.items():
        if key == "decision_index":
            continue
        np.testing.assert_array_equal(rebuilt[key], np.stack(values, axis=0))
    assert provider.maximum_requested_batch == 4


def test_structured_provider_uses_precomputed_policy_plane_without_builder_loop() -> (
    None
):
    from trade_rl.data.market import MarketDataset
    from trade_rl.learning.teacher_artifact import StructuredTeacherObservationProvider
    from trade_rl.rl.sequence_observations import (
        SequenceObservationBuilder,
        SequenceWindowSpec,
        build_sequence_policy_plane,
    )

    n_bars = 140
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 + phase)[:, None]
    dataset = MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:15:00", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.stack(
            tuple(np.sin(phase / divisor) for divisor in (3.0, 5.0, 7.0, 11.0)),
            axis=1,
        )[:, None, :].astype(np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack((close[:1], close[:-1])),
        high=np.maximum(np.vstack((close[:1], close[:-1])), close) + 0.1,
        low=np.minimum(np.vstack((close[:1], close[:-1])), close) - 0.1,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=(
            "15m__return",
            "1h__return",
            "4h__return",
            "1d__return",
        ),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )
    windows = (
        SequenceWindowSpec("15m", 4),
        SequenceWindowSpec("1h", 3),
        SequenceWindowSpec("4h", 2),
        SequenceWindowSpec("1d", 2),
    )
    builder = SequenceObservationBuilder(windows=windows)
    policy_plane = build_sequence_policy_plane(dataset, builder, None)

    class ForbiddenBuilder(SequenceObservationBuilder):
        def build(self, dataset: MarketDataset, *, index: int):  # type: ignore[no-untyped-def]
            raise AssertionError("per-sample sequence builder must not be called")

    observations = {
        "active": np.ones((2, 1), dtype=np.float32),
        "asset_state": np.zeros((2, 1, 2), dtype=np.float32),
        "current_snapshot": np.zeros((2, 1, 16), dtype=np.float32),
        "global_state": np.zeros((2, 3), dtype=np.float32),
        "decision_index": np.asarray([[128], [129]], dtype=np.int64),
    }
    provider = StructuredTeacherObservationProvider(
        dataset=dataset,
        sequence_builder=ForbiddenBuilder(windows=windows),
        observations=observations,
        policy_plane=policy_plane,
    )

    actual = provider.get(np.asarray([0, 1], dtype=np.int64))
    expected_sequences = policy_plane.batch_components(
        np.asarray([128, 129], dtype=np.int64)
    )

    for key, expected in observations.items():
        if key != "decision_index":
            np.testing.assert_array_equal(actual[key], expected)
    for key, expected in expected_sequences.items():
        np.testing.assert_array_equal(actual[key], expected)
