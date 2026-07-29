from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.integrations import sb3_training
from trade_rl.integrations.sb3_training import (
    StableBaselines3Backend,
    _oracle_episode_sampling_config,
)
from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
    episode_oracle_target_path,
)
from trade_rl.learning.episode_teacher_artifact import (
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _market(n_bars: int = 512) -> MarketDataset:
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(phase * 0.001))[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
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


def _environment() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _market(),
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
            initial_state_modes=("cash", "baseline"),
            execution_cost=ExecutionCostConfig.zero(),
            structured_sequence_observation=True,
            sequence_windows=(("15m", 4), ("1h", 3), ("4h", 2), ("1d", 2)),
        ),
    )


def _episode_batch(environment: ResidualMarketEnv) -> EpisodeOracleBatch:
    teacher = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        reference_portfolio_value=environment.initial_capital,
        signal_delay_decisions=environment.config.signal_delay_decisions,
    )
    start = environment.minimum_start_index
    baseline_start = start + 8
    cash = np.zeros(environment.dataset.n_symbols, dtype=np.float64)
    baseline = environment.initial_weights_for_reset("baseline", baseline_start)
    contracts = (
        OracleEpisodeContract(
            dataset_id=environment.dataset.dataset_id,
            episode_index=0,
            start=start,
            stop=start + 5,
            initial_state_mode="cash",
            initial_weights=cash,
        ),
        OracleEpisodeContract(
            dataset_id=environment.dataset.dataset_id,
            episode_index=1,
            start=baseline_start,
            stop=baseline_start + 5,
            initial_state_mode="baseline",
            initial_weights=baseline,
        ),
    )
    targets = tuple(
        episode_oracle_target_path(
            environment.dataset,
            (contract.start, contract.stop),
            teacher,
            initial_weights=contract.initial_weights,
        )
        for contract in contracts
    )
    return EpisodeOracleBatch(
        dataset_id=environment.dataset.dataset_id,
        teacher_config_digest=teacher.digest,
        sampling_config_digest="b" * 64,
        contracts=contracts,
        targets=targets,
    )


def test_episode_teacher_rollout_round_trip_preserves_boundaries(
    tmp_path: Path,
) -> None:
    environment = _environment()
    batch = _episode_batch(environment)

    supervised = collect_episode_teacher_rollout(
        environment,
        batch,
        teacher_config_digest=batch.teacher_config_digest,
    )

    assert supervised.sample_count == 8
    assert supervised.episode_count == 2
    np.testing.assert_array_equal(
        supervised.episode_ids,
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        supervised.decision_indices,
        np.concatenate(
            [
                np.arange(contract.start, contract.stop - 1, dtype=np.int64)
                for contract in batch.contracts
            ]
        ),
    )
    assert isinstance(supervised.observations, dict)
    np.testing.assert_allclose(
        supervised.observations["current_weights"][0],
        batch.contracts[0].initial_weights,
    )
    np.testing.assert_allclose(
        supervised.observations["current_weights"][4],
        batch.contracts[1].initial_weights,
    )

    write_episode_teacher_artifact(tmp_path, supervised)
    manifest, loaded = load_episode_teacher_artifact(tmp_path)

    assert manifest.episode_count == 2
    np.testing.assert_array_equal(loaded.episode_ids, supervised.episode_ids)
    np.testing.assert_array_equal(loaded.decision_indices, supervised.decision_indices)
    assert loaded.episode_ids.flags.writeable is False
    assert loaded.decision_indices.flags.writeable is False


def _supervised_with_episode_ids() -> EpisodeSupervisedPolicyDataset:
    sample_count = 12
    return EpisodeSupervisedPolicyDataset(
        observations=np.zeros((sample_count, 2), dtype=np.float32),
        actions=np.zeros((sample_count, 1), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=10,
        train_stop=32,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
        decision_indices=np.asarray(
            [10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 30],
            dtype=np.int64,
        ),
        episode_ids=np.repeat(np.arange(3, dtype=np.int64), 4),
    )


def test_behavior_cloning_validation_holds_out_complete_episodes() -> None:
    split = behavior_cloning_split(
        _supervised_with_episode_ids(),
        validation_fraction=0.34,
    )

    np.testing.assert_array_equal(split.train_episode_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(split.validation_episode_ids, np.asarray([2]))
    np.testing.assert_array_equal(split.train_indices, np.arange(8, dtype=np.int64))
    np.testing.assert_array_equal(
        split.validation_indices,
        np.arange(8, 12, dtype=np.int64),
    )


def test_behavior_cloning_split_preserves_chronological_tail_for_one_episode() -> None:
    dataset = SupervisedPolicyDataset(
        observations=np.zeros((10, 2), dtype=np.float32),
        actions=np.zeros((10, 1), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=4,
        train_stop=15,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )

    split = behavior_cloning_split(dataset, validation_fraction=0.2)

    np.testing.assert_array_equal(split.train_indices, np.arange(8, dtype=np.int64))
    np.testing.assert_array_equal(
        split.validation_indices,
        np.arange(8, 10, dtype=np.int64),
    )


def test_oracle_episode_sampling_is_derived_from_environment_contract() -> None:
    environment = _environment()
    train_range = (environment.minimum_start_index, environment.dataset.n_bars)

    sampling = _oracle_episode_sampling_config(
        environment,
        train_range=train_range,
        seed=17,
    )

    train_decisions = train_range[1] - train_range[0] - 1
    assert sampling.episode_bars == environment.episode_bars
    assert sampling.episode_count == math.ceil(
        train_decisions / environment.episode_bars
    )
    assert sampling.initial_state_modes == environment.config.initial_state_modes
    assert sampling.seed == 17


def test_backend_caches_episode_oracle_batch_by_sampling_identity(
    monkeypatch: Any,
) -> None:
    environment = _environment()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    train_range = (environment.minimum_start_index, environment.dataset.n_bars)
    sampling = _oracle_episode_sampling_config(
        environment,
        train_range=train_range,
        seed=3,
    )
    calls = 0
    expected = SimpleNamespace(digest="e" * 64)

    def build(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(sb3_training, "build_episode_oracle_batch", build)
    backend = StableBaselines3Backend(lambda: environment)

    first = backend._oracle_episode_batch(
        environment,
        train_range,
        teacher,
        sampling,
    )
    second = backend._oracle_episode_batch(
        environment,
        train_range,
        teacher,
        sampling,
    )

    assert calls == 1
    assert first is expected
    assert second is expected
