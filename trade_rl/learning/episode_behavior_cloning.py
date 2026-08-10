"""Episode-aware train/validation contracts for behavior cloning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, cast

import numpy as np

from trade_rl.learning.behavior_cloning import BehaviorCloningConfig


def _empty_int_vector() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


class BehaviorCloningDataset(Protocol):
    @property
    def sample_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class BehaviorCloningSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_episode_ids: np.ndarray
    validation_episode_ids: np.ndarray
    purged_indices: np.ndarray = field(default_factory=_empty_int_vector)
    purged_episode_ids: np.ndarray = field(default_factory=_empty_int_vector)

    def __post_init__(self) -> None:
        index_names = ("train_indices", "validation_indices", "purged_indices")
        episode_names = (
            "train_episode_ids",
            "validation_episode_ids",
            "purged_episode_ids",
        )
        resolved: dict[str, np.ndarray] = {}
        for name in (*index_names, *episode_names):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must be an integer vector")
            value = np.asarray(raw, dtype=np.int64).copy(order="C")
            if np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")
            if np.unique(value).size != value.size:
                raise ValueError(f"{name} must not contain duplicates")
            value.setflags(write=False)
            resolved[name] = value
        if resolved["train_indices"].size == 0:
            raise ValueError("behavior cloning split requires training samples")
        for names, label in ((index_names, "samples"), (episode_names, "episodes")):
            for left_position, left_name in enumerate(names):
                for right_name in names[left_position + 1 :]:
                    if np.intersect1d(resolved[left_name], resolved[right_name]).size:
                        raise ValueError(f"behavior cloning split overlaps {label}")
        for name, value in resolved.items():
            object.__setattr__(self, name, value)

    @property
    def validation_sample_count(self) -> int:
        return int(self.validation_indices.size)

    @property
    def purged_sample_count(self) -> int:
        return int(self.purged_indices.size)


def _validation_count(sample_count: int, validation_fraction: float) -> int:
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be finite and in [0, 0.5)")
    count = int(math.floor(sample_count * validation_fraction))
    if validation_fraction > 0.0:
        count = max(1, count)
    if count >= sample_count:
        raise ValueError("validation split leaves no training samples")
    return count


def _integer_vector(value: object, *, field_name: str, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != count or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{field_name} must be a sample-aligned integer vector")
    resolved = np.asarray(raw, dtype=np.int64)
    if np.any(resolved < 0):
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _temporal_episode_records(
    episode_ids: np.ndarray,
    decision_indices: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    records: list[tuple[int, int, int]] = []
    for raw_episode_id in np.unique(episode_ids):
        episode_id = int(raw_episode_id)
        decisions = np.sort(decision_indices[episode_ids == episode_id])
        if decisions.size == 0 or np.unique(decisions).size != decisions.size:
            raise ValueError("episode decisions must be non-empty and unique")
        expected = np.arange(
            int(decisions[0]),
            int(decisions[0]) + decisions.size,
            dtype=np.int64,
        )
        if not np.array_equal(decisions, expected):
            raise ValueError("episode decisions must be contiguous")
        records.append(
            (
                int(decisions[0]),
                int(decisions[-1]) + 2,
                episode_id,
            )
        )
    return tuple(sorted(records))


def _plain_tail_split(
    *,
    sample_count: int,
    validation_fraction: float,
    episode_ids: np.ndarray,
) -> BehaviorCloningSplit:
    validation_count = _validation_count(sample_count, validation_fraction)
    train_stop = sample_count - validation_count
    return BehaviorCloningSplit(
        train_indices=np.arange(train_stop, dtype=np.int64),
        validation_indices=np.arange(train_stop, sample_count, dtype=np.int64),
        train_episode_ids=np.unique(episode_ids),
        validation_episode_ids=_empty_int_vector(),
    )


def behavior_cloning_split(
    dataset: BehaviorCloningDataset,
    *,
    validation_fraction: float,
) -> BehaviorCloningSplit:
    """Hold out a chronological episode block and purge overlapping Oracle support."""

    sample_count = int(dataset.sample_count)
    if sample_count <= 0:
        raise ValueError("behavior cloning dataset must not be empty")
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be finite and in [0, 0.5)")
    episode_ids = _integer_vector(
        getattr(dataset, "episode_ids", np.zeros(sample_count, dtype=np.int64)),
        field_name="episode ids",
        count=sample_count,
    )
    unique_episode_ids = np.unique(episode_ids)
    if unique_episode_ids.size <= 1:
        return _plain_tail_split(
            sample_count=sample_count,
            validation_fraction=validation_fraction,
            episode_ids=episode_ids,
        )
    if not hasattr(dataset, "decision_indices"):
        raise TypeError("episode behavior cloning requires decision-index provenance")
    decision_indices = _integer_vector(
        getattr(dataset, "decision_indices"),
        field_name="decision indices",
        count=sample_count,
    )
    records = _temporal_episode_records(episode_ids, decision_indices)
    ordered_episode_ids = np.asarray(
        [episode_id for _, _, episode_id in records],
        dtype=np.int64,
    )
    if validation_fraction == 0.0:
        return BehaviorCloningSplit(
            train_indices=np.arange(sample_count, dtype=np.int64),
            validation_indices=_empty_int_vector(),
            train_episode_ids=ordered_episode_ids,
            validation_episode_ids=_empty_int_vector(),
        )
    validation_episode_count = max(
        1,
        int(math.floor(len(records) * validation_fraction)),
    )
    if validation_episode_count >= len(records):
        raise ValueError("episode validation split leaves no training episodes")
    validation_start = records[-validation_episode_count][0]
    validation_episode_ids = np.asarray(
        [episode_id for start, _, episode_id in records if start >= validation_start],
        dtype=np.int64,
    )
    earlier_records = tuple(
        record for record in records if record[0] < validation_start
    )
    train_episode_ids = np.asarray(
        [
            episode_id
            for _, support_stop, episode_id in earlier_records
            if support_stop <= validation_start
        ],
        dtype=np.int64,
    )
    purged_episode_ids = np.asarray(
        [
            episode_id
            for _, support_stop, episode_id in earlier_records
            if support_stop > validation_start
        ],
        dtype=np.int64,
    )
    if train_episode_ids.size == 0:
        raise ValueError("episode validation purging leaves no training episodes")
    train_mask = np.isin(episode_ids, train_episode_ids)
    validation_mask = np.isin(episode_ids, validation_episode_ids)
    purged_mask = np.isin(episode_ids, purged_episode_ids)
    if not np.all(train_mask | validation_mask | purged_mask):
        raise RuntimeError("episode behavior cloning split lost samples")
    return BehaviorCloningSplit(
        train_indices=np.flatnonzero(train_mask),
        validation_indices=np.flatnonzero(validation_mask),
        train_episode_ids=train_episode_ids,
        validation_episode_ids=validation_episode_ids,
        purged_indices=np.flatnonzero(purged_mask),
        purged_episode_ids=purged_episode_ids,
    )


def align_behavior_cloning_validation(
    config: BehaviorCloningConfig,
    dataset: object,
) -> tuple[BehaviorCloningConfig, BehaviorCloningSplit]:
    """Adjust the scalar tail fraction so the existing trainer cuts at an episode edge."""

    if not hasattr(dataset, "sample_count") or not hasattr(dataset, "episode_ids"):
        raise TypeError("episode behavior cloning requires episode-aware provenance")
    episode_dataset = cast(BehaviorCloningDataset, dataset)
    split = behavior_cloning_split(
        episode_dataset,
        validation_fraction=config.validation_fraction,
    )
    if split.validation_sample_count == 0:
        return replace(config, validation_fraction=0.0), split
    exact_fraction = split.validation_sample_count / int(episode_dataset.sample_count)
    aligned_fraction = math.nextafter(exact_fraction, 1.0)
    if not aligned_fraction < 0.5:
        raise ValueError("episode-aligned validation fraction must remain below 0.5")
    return replace(config, validation_fraction=aligned_fraction), split


__all__ = [
    "BehaviorCloningSplit",
    "align_behavior_cloning_validation",
    "behavior_cloning_split",
]
