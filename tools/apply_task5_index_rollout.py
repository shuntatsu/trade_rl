from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 5 anchor in {path}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/rl/test_rollout_memory.py",
        "    estimate_compact_ppo_rollout_buffer_bytes,\n    estimate_ppo_rollout_buffer_bytes,\n",
        "    estimate_index_backed_ppo_rollout_buffer_bytes,\n    estimate_ppo_rollout_buffer_bytes,\n",
    )
    replace_once(
        "tests/rl/test_rollout_memory.py",
        '''    compact_estimated = estimate_compact_ppo_rollout_buffer_bytes(
        space, n_steps=128, n_envs=4, action_dim=3
    )
    assert default_estimated == 473_122_816
    assert compact_estimated == 200_495_104
    assert compact_estimated < default_estimated / 2
    assert compact_estimated < 805_306_368
''',
        '''    space.spaces["decision_index"] = spaces.Box(
        0, 100_000, shape=(1,), dtype=np.int64
    )
    compact_estimated = estimate_index_backed_ppo_rollout_buffer_bytes(
        space, n_steps=128, n_envs=4, action_dim=3
    )
    assert default_estimated == 473_122_816
    assert compact_estimated == 5_765_120
    assert compact_estimated < default_estimated / 50
    assert compact_estimated < 805_306_368
''',
    )
    replace_once(
        "tests/rl/test_rollout_memory.py",
        '''def test_compact_dict_rollout_buffer_preserves_declared_component_dtypes() -> None:
    from trade_rl.integrations.compact_rollout_buffer import CompactDictRolloutBuffer

    observation_space = spaces.Dict(
        {
            "values": spaces.Box(-10, 10, shape=(2, 3), dtype=np.float16),
            "available": spaces.Box(0, 1, shape=(2, 3), dtype=np.uint8),
            "current": spaces.Box(-10, 10, shape=(4,), dtype=np.float32),
        }
    )
    buffer = CompactDictRolloutBuffer(
        4,
        observation_space,
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=2,
    )

    assert buffer.observations["values"].dtype == np.float16
    assert buffer.observations["available"].dtype == np.uint8
    assert buffer.observations["current"].dtype == np.float32
''',
        '''def test_index_backed_rollout_buffer_does_not_allocate_sequence_arrays() -> None:
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
    from trade_rl.integrations.compact_rollout_buffer import SequenceRolloutReconstructor
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
    from trade_rl.integrations.compact_rollout_buffer import SequenceRolloutReconstructor
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
''',
    )
    append_once(
        "tests/rl/test_rollout_memory.py",
        "def _sequence_dataset()",
        '''

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
''',
    )
    append_once(
        "tests/rl/test_sequence_observations.py",
        "test_sequence_observation_schema_is_index_backed",
        '''

def test_sequence_observation_schema_is_index_backed() -> None:
    from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA

    assert SEQUENCE_OBSERVATION_SCHEMA == "native_timeframe_sequence_observation_v2"
''',
    )


def add_implementation() -> None:
    compact = ROOT / "trade_rl/integrations/compact_rollout_buffer.py"
    compact.write_text(
        '''"""Index-backed SB3 rollout storage for structured sequence policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceNormalizerProtocol,
    SequenceObservationBuilder,
    sequence_policy_values,
)

_SEQUENCE_PREFIX = "sequence_"
_DECISION_INDEX_KEY = "decision_index"
_FLOAT16_MAX = float(np.finfo(np.float16).max)


@dataclass(frozen=True, slots=True)
class SequenceRolloutReconstructor:
    """Rebuild overlapping native histories only for sampled PPO minibatches."""

    dataset: MarketDataset
    builder: SequenceObservationBuilder
    normalizer: SequenceNormalizerProtocol | None
    expected_dataset_id: str
    expected_layout_digest: str

    def __post_init__(self) -> None:
        if self.dataset.dataset_id != self.expected_dataset_id:
            raise ValueError("rollout reconstruction dataset identity mismatch")
        if self.builder.layout_digest(self.dataset) != self.expected_layout_digest:
            raise ValueError("rollout reconstruction sequence layout mismatch")

    def reconstruct(self, decision_indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("rollout reconstruction indices must not be empty")
        if np.any(indices < self.builder.minimum_index(self.dataset)) or np.any(
            indices >= self.dataset.n_bars
        ):
            raise ValueError("rollout reconstruction index is outside causal history")
        cache: dict[int, dict[str, np.ndarray]] = {}
        for raw_index in np.unique(indices):
            index = int(raw_index)
            sequence = self.builder.build(self.dataset, index=index)
            components: dict[str, np.ndarray] = {}
            for timeframe in sequence.values:
                components[f"sequence_{timeframe}_values"] = sequence_policy_values(
                    timeframe=timeframe,
                    values=sequence.values[timeframe],
                    available=sequence.available[timeframe],
                    feature_names=sequence.feature_names[timeframe],
                    sequence_normalizer=self.normalizer,
                )
                components[f"sequence_{timeframe}_available"] = np.asarray(
                    sequence.available[timeframe], dtype=np.uint8
                )
                components[f"sequence_{timeframe}_staleness"] = np.asarray(
                    np.clip(sequence.staleness[timeframe], 0.0, _FLOAT16_MAX),
                    dtype=np.float16,
                )
            cache[index] = components
        keys = tuple(cache[int(indices[0])])
        return {
            key: np.stack([cache[int(index)][key] for index in indices], axis=0)
            for key in keys
        }


class IndexBackedDictRolloutBuffer(DictRolloutBuffer):
    """Store current state and indices; reconstruct native histories on sampling."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Any = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        sequence_reconstructor: SequenceRolloutReconstructor,
    ) -> None:
        if _DECISION_INDEX_KEY not in observation_space.spaces:
            raise ValueError("index-backed rollout requires decision_index observation")
        self._sequence_keys = tuple(
            key for key in observation_space.spaces if key.startswith(_SEQUENCE_PREFIX)
        )
        if not self._sequence_keys:
            raise ValueError("index-backed rollout requires sequence components")
        self._compact_keys = tuple(
            key for key in observation_space.spaces if key not in self._sequence_keys
        )
        self.sequence_reconstructor = sequence_reconstructor
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape[key]),
                dtype=np.dtype(self.observation_space.spaces[key].dtype),
            )
            for key in self._compact_keys
        }
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray | dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: Any,
        log_prob: Any,
    ) -> None:
        if not isinstance(obs, dict):
            raise TypeError("index-backed Dict rollout requires mapping observations")
        missing = set(self.observation_space.spaces).difference(obs)
        if missing:
            raise ValueError(f"rollout observation is missing components: {sorted(missing)}")
        decision_index = np.asarray(obs[_DECISION_INDEX_KEY])
        if not np.issubdtype(decision_index.dtype, np.integer):
            raise ValueError("decision_index observation must be integral")
        super().add(obs, action, reward, episode_start, value, log_prob)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
        if env is not None:
            raise ValueError("index-backed sequence rollout does not support VecNormalize")
        raw_indices = self.observations[_DECISION_INDEX_KEY][batch_inds]
        decision_indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        observations = {
            key: values[batch_inds]
            for key, values in self.observations.items()
            if key != _DECISION_INDEX_KEY
        }
        observations.update(self.sequence_reconstructor.reconstruct(decision_indices))
        return DictRolloutBufferSamples(
            observations={
                key: self.to_torch(value) for key, value in observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )


CompactDictRolloutBuffer = IndexBackedDictRolloutBuffer

__all__ = [
    "CompactDictRolloutBuffer",
    "IndexBackedDictRolloutBuffer",
    "SequenceRolloutReconstructor",
]
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/rl/sequence_observations.py",
        'SEQUENCE_OBSERVATION_SCHEMA = "native_timeframe_sequence_observation_v1"',
        'SEQUENCE_OBSERVATION_SCHEMA = "native_timeframe_sequence_observation_v2"',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''            sequence_spaces: dict[str, spaces.Space[np.ndarray]] = {
                "current_snapshot": spaces.Box(
''',
        '''            sequence_spaces: dict[str, spaces.Space[np.ndarray]] = {
                "decision_index": spaces.Box(
                    low=0,
                    high=dataset.n_bars - 1,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "current_snapshot": spaces.Box(
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''        return build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
''',
        '''        structured = build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        structured["decision_index"] = np.asarray(
            [self.current_index], dtype=np.int64
        )
        return structured
''',
    )

    rollout_memory = ROOT / "trade_rl/rl/rollout_memory.py"
    text = rollout_memory.read_text(encoding="utf-8")
    text = text.replace(
        '''def estimate_compact_ppo_rollout_buffer_bytes(
    observation_space: spaces.Space,
    *,
    n_steps: int,
    n_envs: int,
    action_dim: int,
) -> int:
    """Estimate persistent bytes when Dict components keep declared dtypes."""
''',
        '''def estimate_index_backed_ppo_rollout_buffer_bytes(
    observation_space: spaces.Space,
    *,
    n_steps: int,
    n_envs: int,
    action_dim: int,
) -> int:
    """Estimate persistent bytes after overlapping sequence arrays are removed."""
''',
        1,
    )
    text = text.replace(
        '''    observation_bytes = 0
    for component in observation_space.spaces.values():
''',
        '''    if "decision_index" not in observation_space.spaces:
        raise ValueError("index-backed rollout requires decision_index observation")
    observation_bytes = 0
    for key, component in observation_space.spaces.items():
        if key.startswith("sequence_"):
            continue
''',
        1,
    )
    text = text.replace(
        '''__all__ = [
    "estimate_compact_ppo_rollout_buffer_bytes",
    "estimate_ppo_rollout_buffer_bytes",
]
''',
        '''estimate_compact_ppo_rollout_buffer_bytes = (
    estimate_index_backed_ppo_rollout_buffer_bytes
)

__all__ = [
    "estimate_compact_ppo_rollout_buffer_bytes",
    "estimate_index_backed_ppo_rollout_buffer_bytes",
    "estimate_ppo_rollout_buffer_bytes",
]
''',
        1,
    )
    rollout_memory.write_text(text, encoding="utf-8")

    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''    estimate_compact_ppo_rollout_buffer_bytes,
    estimate_ppo_rollout_buffer_bytes,
''',
        '''    estimate_index_backed_ppo_rollout_buffer_bytes,
    estimate_ppo_rollout_buffer_bytes,
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                    estimate_compact_ppo_rollout_buffer_bytes
                    if config.sequence_encoder
''',
        '''                    estimate_index_backed_ppo_rollout_buffer_bytes
                    if config.sequence_encoder
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''            policy_kwargs: dict[str, Any]
            sequence_metadata: dict[str, Any] | None = None
''',
        '''            policy_kwargs: dict[str, Any]
            sequence_metadata: dict[str, Any] | None = None
            sequence_reconstructor: Any | None = None
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                sequence_metadata = dict(metadata)
                policy_kwargs = {
''',
        '''                sequence_metadata = dict(metadata)
                from trade_rl.integrations.compact_rollout_buffer import (
                    SequenceRolloutReconstructor,
                )

                dataset = getattr(unwrapped, "dataset", None)
                sequence_builder = getattr(
                    unwrapped, "sequence_observation_builder", None
                )
                if dataset is None or sequence_builder is None:
                    raise ValueError(
                        "sequence training requires dataset-bound reconstruction metadata"
                    )
                sequence_reconstructor = SequenceRolloutReconstructor(
                    dataset=dataset,
                    builder=sequence_builder,
                    normalizer=getattr(unwrapped, "sequence_normalizer", None),
                    expected_dataset_id=dataset.dataset_id,
                    expected_layout_digest=sequence_builder.layout_digest(dataset),
                )
                policy_kwargs = {
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                    from trade_rl.integrations.compact_rollout_buffer import (
                        CompactDictRolloutBuffer,
                    )

                    rollout_kwargs["rollout_buffer_class"] = CompactDictRolloutBuffer
''',
        '''                    from trade_rl.integrations.compact_rollout_buffer import (
                        IndexBackedDictRolloutBuffer,
                    )

                    if sequence_reconstructor is None:
                        raise RuntimeError("sequence rollout reconstructor was not resolved")
                    rollout_kwargs["rollout_buffer_class"] = (
                        IndexBackedDictRolloutBuffer
                    )
                    rollout_kwargs["rollout_buffer_kwargs"] = {
                        "sequence_reconstructor": sequence_reconstructor
                    }
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                            "compact_dict" if config.sequence_encoder else "default"
''',
        '''                            "index_backed_dict"
                            if config.sequence_encoder
                            else "default"
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task5_index_rollout.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
