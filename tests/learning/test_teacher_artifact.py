from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.teacher_artifact import (
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    load_teacher_artifact,
    write_teacher_artifact,
)
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _supervised() -> SupervisedPolicyDataset:
    return SupervisedPolicyDataset(
        observations=np.arange(18, dtype=np.float32).reshape(6, 3),
        actions=np.array(
            [[0.5, -0.5], [0.5, 0.0], [0.0, 0.0]] * 2,
            dtype=np.float32,
        ),
        dataset_id="a" * 64,
        train_start=2,
        train_stop=9,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )


def test_teacher_artifact_round_trip_is_content_addressed(tmp_path: Path) -> None:
    supervised = _supervised()

    digest = write_teacher_artifact(tmp_path, supervised)
    manifest, loaded = load_teacher_artifact(
        tmp_path,
        expected_dataset_id=supervised.dataset_id,
        expected_environment_digest=supervised.environment_digest,
        expected_action_spec_digest=supervised.action_spec_digest,
        expected_train_range=(supervised.train_start, supervised.train_stop),
    )

    assert digest == manifest.artifact_digest
    assert manifest.observation_digest == supervised.observation_digest
    assert manifest.action_digest == supervised.action_digest
    assert manifest.sample_count == 6
    np.testing.assert_array_equal(loaded.observations, supervised.observations)
    np.testing.assert_array_equal(loaded.actions, supervised.actions)


def test_teacher_artifact_detects_array_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    arrays_root = tmp_path / "arrays"
    write_teacher_artifact(arrays_root, _supervised())
    arrays_path = arrays_root / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_teacher_artifact(arrays_root)

    manifest_root = tmp_path / "manifest"
    write_teacher_artifact(manifest_root, _supervised())
    manifest_path = manifest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_stop"] = 10
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_teacher_artifact(manifest_root)


def test_supervised_dataset_rejects_range_and_identity_mismatches(
    tmp_path: Path,
) -> None:
    supervised = _supervised()
    write_teacher_artifact(tmp_path, supervised)

    with pytest.raises(ValueError, match="training range identity mismatch"):
        load_teacher_artifact(tmp_path, expected_train_range=(2, 10))
    with pytest.raises(ValueError, match="environment identity mismatch"):
        load_teacher_artifact(tmp_path, expected_environment_digest="e" * 64)

    with pytest.raises(ValueError, match="sample count"):
        SupervisedPolicyDataset(
            observations=np.ones((2, 3), dtype=np.float32),
            actions=np.ones((2, 2), dtype=np.float32),
            dataset_id="a" * 64,
            train_start=2,
            train_stop=9,
            environment_digest="b" * 64,
            action_spec_digest="c" * 64,
            teacher_config_digest="d" * 64,
        )


class _RolloutEnvironment:
    environment_digest = "b" * 64
    action_spec_digest = "c" * 64

    def __init__(self) -> None:
        self.current_index = 0
        self.actions: list[np.ndarray] = []

    def reset(
        self,
        *,
        options: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, object]]:
        assert options["initial_state_mode"] == "cash"
        start = options["start_idx"]
        bars = options["episode_bars"]
        assert isinstance(start, int)
        assert isinstance(bars, int)
        self.current_index = start
        self.stop = self.current_index + bars
        return np.array([self.current_index, 1.0], dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.actions.append(action.copy())
        self.current_index += 1
        terminated = self.current_index == self.stop
        observation = np.array([self.current_index, 1.0], dtype=np.float32)
        return observation, 0.0, terminated, False, {}


def test_collect_teacher_rollout_covers_exact_training_decisions() -> None:
    environment = _RolloutEnvironment()
    targets = np.array([[0.5, -0.5], [0.0, 0.0], [-0.5, 0.5]], dtype=np.float64)

    supervised = collect_teacher_rollout(
        environment,
        targets,
        dataset_id="a" * 64,
        train_range=(4, 8),
        teacher_config_digest="d" * 64,
    )

    np.testing.assert_array_equal(
        supervised.observations[:, 0], np.array([4.0, 5.0, 6.0])
    )
    np.testing.assert_array_equal(supervised.actions, targets.astype(np.float32))
    assert environment.current_index == 7


def _real_environment() -> ResidualMarketEnv:
    n_bars = 40
    close = np.column_stack(
        [np.linspace(100.0, 120.0, n_bars), np.linspace(100.0, 90.0, n_bars)]
    )
    market = MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    return ResidualMarketEnv(
        market,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=2,
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=3,
            decision_every=1,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_collect_teacher_rollout_uses_real_environment_episode_contract() -> None:
    environment = _real_environment()
    targets = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 0.0]], dtype=np.float32)

    supervised = collect_teacher_rollout(
        environment,
        targets,
        dataset_id=environment.dataset.dataset_id,
        train_range=(10, 14),
        teacher_config_digest="d" * 64,
    )

    assert supervised.observations.shape[0] == 3
    assert environment.start_index == 10
    assert environment.end_index == 13
    assert environment.current_index == 13
