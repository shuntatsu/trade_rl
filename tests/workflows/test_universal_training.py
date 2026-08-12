from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.rl.observations import observation_layout
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer


def _digest(label: str) -> str:
    return content_digest(label)


def _dataset(symbol: str) -> SimpleNamespace:
    feature_names = (
        "15m__constant",
        "1h__trend",
        "4h__momentum",
        "1d__volatility",
    )
    return SimpleNamespace(
        dataset_id=_digest(f"dataset:{symbol}"),
        symbols=(symbol,),
        n_symbols=1,
        feature_names=feature_names,
        n_features=len(feature_names),
        global_feature_names=("global_a", "global_b"),
        bar_hours=0.25,
        nominal_bar_hours=0.25,
    )


def _shared_normalizer() -> SymbolBalancedStandardNormalizer:
    return SymbolBalancedStandardNormalizer.fit(
        {
            "AAAUSDT": np.asarray(
                [
                    [1.0, 0.0, 10.0, 100.0],
                    [1.0, 2.0, 12.0, 102.0],
                    [1.0, 4.0, 14.0, 104.0],
                    [1.0, 6.0, 16.0, 106.0],
                ],
                dtype=np.float64,
            ),
            "BBBUSDT": np.asarray(
                [
                    [1.0, 100.0, 20.0, 200.0],
                    [1.0, 102.0, 22.0, 202.0],
                    [1.0, 104.0, 24.0, 204.0],
                    [1.0, 106.0, 26.0, 206.0],
                ],
                dtype=np.float64,
            ),
        },
        train_symbols=("AAAUSDT", "BBBUSDT"),
        feature_schema_digest=_digest("feature-schema"),
        catalog_digest=_digest("catalog"),
        split_manifest_digest=_digest("partition"),
        fold_train_range=(0, 4),
        max_samples_per_symbol=4,
    )


def test_shared_statistics_bind_to_each_dataset_without_changing_statistics_digest() -> (
    None
):
    from trade_rl.workflows.universal_training import bind_universal_normalizers

    shared = _shared_normalizer()
    first = _dataset("AAAUSDT")
    second = _dataset("BBBUSDT")
    first_flat, first_sequence = bind_universal_normalizers(
        first,
        shared=shared,
        action_spec_digest=_digest("generic-action"),
        finite_horizon=True,
    )
    second_flat, second_sequence = bind_universal_normalizers(
        second,
        shared=shared,
        action_spec_digest=_digest("generic-action"),
        finite_horizon=True,
    )

    assert first_flat.statistics_digest == shared.statistics_digest
    assert second_flat.statistics_digest == shared.statistics_digest
    assert first_sequence.statistics_digest == shared.statistics_digest
    assert second_sequence.statistics_digest == shared.statistics_digest
    assert first_flat.dataset_id != second_flat.dataset_id
    assert first_flat.digest != second_flat.digest
    assert first_sequence.dataset_id != second_sequence.dataset_id
    assert first_sequence.digest != second_sequence.digest

    layout = observation_layout(first, action_size=1, finite_horizon=True)
    raw = np.zeros(layout.size, dtype=np.float32)
    raw[: first.n_features] = np.asarray([999.0, 51.0, 18.0, 151.0])
    raw[first.n_features : 2 * first.n_features] = 1.0
    raw[-1] = 0.75
    normalized = first_flat.transform(raw)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(0.75)

    sequence = first_sequence.transform(
        "15m",
        np.asarray([[[999.0]]], dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.bool_),
        feature_names=("15m__constant",),
    )
    assert sequence.item() == pytest.approx(0.0)


def test_universal_dataset_scope_rejects_validation_or_test_inputs() -> None:
    from trade_rl.workflows.universal_training import validate_universal_dataset_scope

    with pytest.raises(ValueError, match="exactly match train_symbols"):
        validate_universal_dataset_scope(
            {
                "AAAUSDT": object(),
                "VALIDATION": object(),
            },
            train_symbols=("AAAUSDT", "BBBUSDT"),
        )


class _FullDictTeacherEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.environment_digest = _digest("universal-teacher-env")
        self.action_spec_digest = _digest("generic-action")
        self.current_index = 0
        self._end = 0
        component_spaces: dict[str, gym.Space[Any]] = {
            "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
            "current_snapshot": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float32),
            "asset_state": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float32),
            "global_state": spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32),
            "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "instrument_context": spaces.Box(
                -10.0, 10.0, shape=(1, 9), dtype=np.float32
            ),
        }
        for timeframe in ("15m", "1h", "4h", "1d"):
            component_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
                -10.0, 10.0, shape=(1, 2, 1), dtype=np.float32
            )
            component_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
                0, 1, shape=(1, 2, 1), dtype=np.uint8
            )
            component_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
                0.0, 10.0, shape=(1, 2, 1), dtype=np.float32
            )
        self.observation_space = spaces.Dict(component_spaces)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "decision_index": np.asarray([self.current_index], dtype=np.int64),
            "current_snapshot": np.asarray(
                [[float(self.current_index)]], dtype=np.float32
            ),
            "asset_state": np.zeros((1, 1), dtype=np.float32),
            "global_state": np.zeros((1,), dtype=np.float32),
            "active": np.ones((1,), dtype=np.float32),
            "current_weights": np.zeros((1,), dtype=np.float32),
            "instrument_context": np.ones((1, 9), dtype=np.float32),
        }
        for timeframe in ("15m", "1h", "4h", "1d"):
            result[f"sequence_{timeframe}_values"] = np.full(
                (1, 2, 1), float(self.current_index), dtype=np.float32
            )
            result[f"sequence_{timeframe}_available"] = np.ones(
                (1, 2, 1), dtype=np.uint8
            )
            result[f"sequence_{timeframe}_staleness"] = np.zeros(
                (1, 2, 1), dtype=np.float32
            )
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        super().reset(seed=seed)
        if options is None:
            raise ValueError("teacher reset requires explicit options")
        self.current_index = int(options["start_idx"])
        self._end = self.current_index + int(options["episode_bars"])
        return self._observation(), {
            "start_index": self.current_index,
            "end_index": self._end,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        reward = float(action[0])
        self.current_index += 1
        terminated = self.current_index >= self._end
        return self._observation(), reward, terminated, False, {}


def test_universal_teacher_collector_keeps_full_dict_and_aligned_return_targets() -> (
    None
):
    from trade_rl.workflows.universal_training import collect_universal_episode_teacher

    dataset_id = _digest("teacher-dataset")
    teacher_digest = _digest("teacher-config")
    batch = EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=teacher_digest,
        sampling_config_digest=_digest("sampling"),
        contracts=(
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=0,
                start=10,
                stop=13,
                initial_state_mode="cash",
                initial_weights=np.zeros(1, dtype=np.float64),
            ),
        ),
        targets=(np.asarray([[1.0], [2.0]], dtype=np.float32),),
        solver_provenance=None,
    )

    collected = collect_universal_episode_teacher(
        _FullDictTeacherEnv(),
        batch,
        teacher_config_digest=teacher_digest,
        gamma=1.0,
    )

    assert collected.dataset.sample_count == 2
    assert isinstance(collected.dataset.observations, dict)
    assert "instrument_context" in collected.dataset.observations
    assert "sequence_15m_values" in collected.dataset.observations
    assert collected.dataset.observations["sequence_15m_values"].shape == (
        2,
        1,
        2,
        1,
    )
    assert collected.critic_targets.tolist() == pytest.approx([3.0, 2.0])


def test_universal_teacher_collector_causally_subsamples_observations() -> None:
    from trade_rl.workflows.universal_training import collect_universal_episode_teacher

    dataset_id = _digest("teacher-dataset-strided")
    teacher_digest = _digest("teacher-config")
    batch = EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=teacher_digest,
        sampling_config_digest=_digest("sampling-strided"),
        contracts=(
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=0,
                start=10,
                stop=15,
                initial_state_mode="cash",
                initial_weights=np.zeros(1, dtype=np.float64),
            ),
        ),
        targets=(np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),),
        solver_provenance=None,
    )

    collected = collect_universal_episode_teacher(
        _FullDictTeacherEnv(),
        batch,
        teacher_config_digest=teacher_digest,
        gamma=1.0,
        sample_stride=2,
    )

    assert collected.dataset.sample_count == 2
    np.testing.assert_array_equal(
        collected.dataset.decision_indices,
        np.asarray([10, 12], dtype=np.int64),
    )
    assert collected.critic_targets.tolist() == pytest.approx([10.0, 7.0])
    assert collected.dataset.environment_digest != _digest("environment")
