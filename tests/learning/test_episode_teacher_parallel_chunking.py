from __future__ import annotations

import multiprocessing
import threading
import warnings

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.learning.episode_teacher_artifact import (
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout_parallel,
)


class _FakeTeacherEnvironment:
    environment_digest = "e" * 64
    action_spec_digest = "f" * 64

    def __init__(self, close_counter: list[int], lock: threading.Lock) -> None:
        self.current_index = 0
        self._remaining = 0
        self._close_counter = close_counter
        self._lock = lock

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        raw_start = options["start_idx"]
        raw_episode_bars = options["episode_bars"]
        if isinstance(raw_start, bool) or not isinstance(raw_start, int):
            raise TypeError("start_idx must be an integer")
        if isinstance(raw_episode_bars, bool) or not isinstance(raw_episode_bars, int):
            raise TypeError("episode_bars must be an integer")
        self.current_index = raw_start
        self._remaining = raw_episode_bars
        return np.asarray([raw_start], dtype=np.float32), {"start_index": raw_start}

    def step(
        self, target: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del target
        self.current_index += 1
        self._remaining -= 1
        terminated = self._remaining == 0
        return (
            np.asarray([self.current_index], dtype=np.float32),
            0.0,
            terminated,
            False,
            {},
        )

    def close(self) -> None:
        with self._lock:
            self._close_counter[0] += 1


def _episode_batch(episode_count: int = 8) -> EpisodeOracleBatch:
    contracts: list[OracleEpisodeContract] = []
    targets: list[np.ndarray] = []
    for episode_index in range(episode_count):
        start = 4 + episode_index * 4
        contracts.append(
            OracleEpisodeContract(
                dataset_id="a" * 64,
                episode_index=episode_index,
                start=start,
                stop=start + 3,
                initial_state_mode="cash",
                initial_weights=np.zeros(1, dtype=np.float64),
            )
        )
        targets.append(
            np.asarray(
                [[episode_index / 10.0], [(episode_index + 1) / 10.0]],
                dtype=np.float32,
            )
        )
    return EpisodeOracleBatch(
        dataset_id="a" * 64,
        teacher_config_digest="d" * 64,
        sampling_config_digest="b" * 64,
        contracts=tuple(contracts),
        targets=tuple(targets),
    )


def _collect_threaded_teacher_batch(
    *,
    episode_count: int,
    max_workers: int,
) -> tuple[EpisodeSupervisedPolicyDataset, int, int]:
    factory_calls = [0]
    close_calls = [0]
    lock = threading.Lock()

    def environment_factory() -> _FakeTeacherEnvironment:
        with lock:
            factory_calls[0] += 1
        return _FakeTeacherEnvironment(close_calls, lock)

    batch = _episode_batch(episode_count)
    dataset = collect_episode_teacher_rollout_parallel(
        environment_factory,
        batch,
        teacher_config_digest=batch.teacher_config_digest,
        max_workers=max_workers,
    )
    return dataset, factory_calls[0], close_calls[0]


def test_parallel_teacher_rollout_reuses_one_environment_per_episode_chunk() -> None:
    dataset, factory_calls, close_calls = _collect_threaded_teacher_batch(
        episode_count=8,
        max_workers=2,
    )
    batch = _episode_batch()

    assert factory_calls == 2
    assert close_calls == 2
    np.testing.assert_array_equal(
        dataset.episode_ids,
        np.repeat(np.arange(batch.episode_count, dtype=np.int64), 2),
    )
    np.testing.assert_array_equal(
        dataset.decision_indices,
        np.concatenate(
            [
                np.arange(contract.start, contract.stop - 1, dtype=np.int64)
                for contract in batch.contracts
            ]
        ),
    )
    np.testing.assert_array_equal(
        dataset.actions, np.concatenate(batch.targets, axis=0)
    )


def test_parallel_teacher_rollout_uses_every_worker_when_episodes_are_available() -> (
    None
):
    dataset, factory_calls, close_calls = _collect_threaded_teacher_batch(
        episode_count=17,
        max_workers=16,
    )

    assert factory_calls == 16
    assert close_calls == 16
    np.testing.assert_array_equal(
        dataset.episode_ids,
        np.repeat(np.arange(17, dtype=np.int64), 2),
    )


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the Python 3.12 fork deprecation is POSIX-specific",
)
def test_parallel_teacher_rollout_does_not_fork_a_multithreaded_parent() -> None:
    factory_calls = [0]
    close_calls = [0]
    lock = threading.Lock()
    batch = _episode_batch(episode_count=2)
    started = threading.Event()
    stop = threading.Event()

    def environment_factory() -> _FakeTeacherEnvironment:
        with lock:
            factory_calls[0] += 1
        return _FakeTeacherEnvironment(close_calls, lock)

    def keep_thread_alive() -> None:
        started.set()
        stop.wait()

    thread = threading.Thread(target=keep_thread_alive, daemon=True)
    thread.start()
    assert started.wait(timeout=5.0)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            dataset = collect_episode_teacher_rollout_parallel(
                environment_factory,
                batch,
                teacher_config_digest=batch.teacher_config_digest,
                max_workers=2,
            )
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert dataset.episode_count == 2
    assert [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "fork" in str(warning.message).lower()
    ] == []
