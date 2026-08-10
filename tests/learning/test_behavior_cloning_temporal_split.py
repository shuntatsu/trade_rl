from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split


def _episode_dataset(
    *,
    episode_ids: list[int],
    decision_indices: list[int],
) -> SimpleNamespace:
    if len(episode_ids) != len(decision_indices):
        raise ValueError("test episode provenance must be sample aligned")
    return SimpleNamespace(
        sample_count=len(episode_ids),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
    )


def test_episode_split_orders_by_time_and_purges_future_support_overlap() -> None:
    dataset = _episode_dataset(
        episode_ids=[9, 9, 9, 2, 2, 2, 7, 7, 7, 1, 1, 1],
        decision_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )

    split = behavior_cloning_split(dataset, validation_fraction=0.26)

    np.testing.assert_array_equal(split.train_episode_ids, np.asarray([9, 2]))
    np.testing.assert_array_equal(split.validation_episode_ids, np.asarray([1]))
    np.testing.assert_array_equal(split.purged_episode_ids, np.asarray([7]))
    np.testing.assert_array_equal(split.train_indices, np.arange(0, 6))
    np.testing.assert_array_equal(split.purged_indices, np.arange(6, 9))
    np.testing.assert_array_equal(split.validation_indices, np.arange(9, 12))
    assert split.validation_sample_count == 3
    assert split.purged_sample_count == 3
    combined = np.concatenate(
        (split.train_indices, split.purged_indices, split.validation_indices)
    )
    np.testing.assert_array_equal(np.sort(combined), np.arange(dataset.sample_count))


def test_episode_split_fails_when_temporal_purge_removes_every_training_episode() -> (
    None
):
    dataset = _episode_dataset(
        episode_ids=[0, 0, 0, 1, 1, 2, 2, 2],
        decision_indices=[7, 8, 9, 8, 9, 10, 11, 12],
    )

    with pytest.raises(ValueError, match="purging leaves no training episodes"):
        behavior_cloning_split(dataset, validation_fraction=0.34)
