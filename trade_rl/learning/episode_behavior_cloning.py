"""Episode-aware train/validation contracts for behavior cloning."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from trade_rl.learning.behavior_cloning import BehaviorCloningConfig


class EpisodeDataset(Protocol):
    sample_count: int
    episode_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class BehaviorCloningSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_episode_ids: np.ndarray
    validation_episode_ids: np.ndarray

    def __post_init__(self) -> None:
        resolved: dict[str, np.ndarray] = {}
        for name in (
            "train_indices",
            "validation_indices",
            "train_episode_ids",
            "validation_episode_ids",
        ):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must be an integer vector")
            value = np.asarray(raw, dtype=np.int64).copy(order="C")
            if np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")
            value.setflags(write=False)
            resolved[name] = value
        if resolved["train_indices"].size == 0:
            raise ValueError("behavior cloning split requires training samples")
        if np.intersect1d(
            resolved["train_indices"], resolved["validation_indices"]
        ).size:
            raise ValueError("behavior cloning split overlaps samples")
        if np.intersect1d(
            resolved["train_episode_ids"], resolved["validation_episode_ids"]
        ).size:
            raise ValueError("behavior cloning split overlaps episodes")
        for name, value in resolved.items():
            object.__setattr__(self, name, value)

    @property
    def validation_sample_count(self) -> int:
        return int(self.validation_indices.size)


def _validation_count(sample_count: int, validation_fraction: float) -> int:
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be finite and in [0, 0.5)")
    count = int(math.floor(sample_count * validation_fraction))
    if validation_fraction > 0.0:
        count = max(1, count)
    if count >= sample_count:
        raise ValueError("validation split leaves no training samples")
    return count


def behavior_cloning_split(
    dataset: EpisodeDataset,
    *,
    validation_fraction: float,
) -> BehaviorCloningSplit:
    """Hold out complete tail episodes, or preserve the legacy one-path tail."""

    sample_count = int(dataset.sample_count)
    if sample_count <= 0:
        raise ValueError("behavior cloning dataset must not be empty")
    episode_ids = np.asarray(
        getattr(dataset, "episode_ids", np.zeros(sample_count, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    if len(episode_ids) != sample_count or np.any(episode_ids < 0):
        raise ValueError("episode ids do not cover the behavior cloning dataset")
    ordered_episodes = np.unique(episode_ids)
    if ordered_episodes.size <= 1:
        validation_count = _validation_count(sample_count, validation_fraction)
        train_stop = sample_count - validation_count
        return BehaviorCloningSplit(
            train_indices=np.arange(train_stop, dtype=np.int64),
            validation_indices=np.arange(train_stop, sample_count, dtype=np.int64),
            train_episode_ids=np.asarray([0], dtype=np.int64),
            validation_episode_ids=np.asarray([], dtype=np.int64),
        )
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be finite and in [0, 0.5)")
    if validation_fraction == 0.0:
        return BehaviorCloningSplit(
            train_indices=np.arange(sample_count, dtype=np.int64),
            validation_indices=np.asarray([], dtype=np.int64),
            train_episode_ids=ordered_episodes,
            validation_episode_ids=np.asarray([], dtype=np.int64),
        )
    validation_episode_count = max(
        1,
        int(math.floor(len(ordered_episodes) * validation_fraction)),
    )
    if validation_episode_count >= len(ordered_episodes):
        raise ValueError("episode validation split leaves no training episodes")
    validation_episode_ids = ordered_episodes[-validation_episode_count:]
    train_episode_ids = ordered_episodes[:-validation_episode_count]
    validation_mask = np.isin(episode_ids, validation_episode_ids)
    return BehaviorCloningSplit(
        train_indices=np.flatnonzero(~validation_mask),
        validation_indices=np.flatnonzero(validation_mask),
        train_episode_ids=train_episode_ids,
        validation_episode_ids=validation_episode_ids,
    )


def align_behavior_cloning_validation(
    config: BehaviorCloningConfig,
    dataset: EpisodeDataset,
) -> tuple[BehaviorCloningConfig, BehaviorCloningSplit]:
    """Adjust the scalar tail fraction so the existing trainer cuts at an episode edge."""

    split = behavior_cloning_split(
        dataset,
        validation_fraction=config.validation_fraction,
    )
    if split.validation_sample_count == 0:
        return replace(config, validation_fraction=0.0), split
    exact_fraction = split.validation_sample_count / int(dataset.sample_count)
    aligned_fraction = math.nextafter(exact_fraction, 1.0)
    if not aligned_fraction < 0.5:
        raise ValueError("episode-aligned validation fraction must remain below 0.5")
    return replace(config, validation_fraction=aligned_fraction), split


__all__ = [
    "BehaviorCloningSplit",
    "align_behavior_cloning_validation",
    "behavior_cloning_split",
]
