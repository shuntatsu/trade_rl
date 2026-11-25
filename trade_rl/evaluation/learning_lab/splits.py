"""Frozen walk-forward split of the learning lab on Tier D row timestamps.

The three windows are half-open on the row timestamp, so a row stamped exactly at a
boundary belongs to the later window and no row is in two windows:

* ``train``    2022-01-01T00:00Z <= t < 2024-01-01T00:00Z  (2022-01-01..2023-12-31)
* ``validate`` 2024-01-01T00:00Z <= t < 2025-07-01T00:00Z  (2024-01-01..2025-06-30)
* ``test``     2025-07-01T00:00Z <= t < 2026-08-01T00:00Z  (2025-07-01..2026-07-31)

``test`` ends at the last row strictly before ``DATA_CUTOFF`` = 2026-08-01T00:00Z. No row at
or after the cutoff may be read; :func:`require_before_cutoff` is the guard. The module is
pure (NumPy and the standard library) and reads no data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

TRAIN: Final = "train"
VALIDATE: Final = "validate"
TEST: Final = "test"
SPLIT_NAMES: Final = (TRAIN, VALIDATE, TEST)

#: No row stamped at or after this instant may be read by the learning lab.
DATA_CUTOFF: Final = np.datetime64("2026-08-01T00:00:00", "ns")


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One half-open window ``[start, stop)`` on row timestamps."""

    name: str
    start: np.datetime64
    stop: np.datetime64

    def __post_init__(self) -> None:
        if np.isnat(self.start) or np.isnat(self.stop):
            raise ValueError("split bounds must not be NaT")
        if not self.start < self.stop:
            raise ValueError("split start must precede its stop")

    def contains(self, timestamp: np.datetime64) -> bool:
        return bool(self.start <= timestamp < self.stop)

    def indices(self, timestamps: np.ndarray) -> tuple[int, int]:
        """``(first_row, last_row)`` (both inclusive) of the rows inside the window."""
        stamps = _sorted_stamps(timestamps)
        first = int(np.searchsorted(stamps, self.start, side="left"))
        stop = int(np.searchsorted(stamps, self.stop, side="left"))
        if stop <= first:
            raise ValueError(f"split {self.name!r} holds no row of the dataset")
        return first, stop - 1

    def payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start": str(self.start),
            "stop_exclusive": str(self.stop),
        }


def _sorted_stamps(timestamps: np.ndarray) -> np.ndarray:
    stamps = np.asarray(timestamps).astype("datetime64[ns]")
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    if np.isnat(stamps).any():
        raise ValueError("timestamps must not contain NaT")
    if stamps.size > 1 and not bool(np.all(stamps[1:] > stamps[:-1])):
        raise ValueError("timestamps must be strictly increasing")
    return stamps


def _split(name: str, start: str, stop: str) -> WalkForwardSplit:
    return WalkForwardSplit(
        name=name,
        start=np.datetime64(start, "ns"),
        stop=np.datetime64(stop, "ns"),
    )


FROZEN_SPLITS: Final[dict[str, WalkForwardSplit]] = {
    TRAIN: _split(TRAIN, "2022-01-01T00:00:00", "2024-01-01T00:00:00"),
    VALIDATE: _split(VALIDATE, "2024-01-01T00:00:00", "2025-07-01T00:00:00"),
    TEST: _split(TEST, "2025-07-01T00:00:00", "2026-08-01T00:00:00"),
}


def require_before_cutoff(timestamps: np.ndarray) -> None:
    """Refuse a dataset that holds any row at or after :data:`DATA_CUTOFF`."""
    stamps = _sorted_stamps(timestamps)
    if bool((stamps >= DATA_CUTOFF).any()):
        raise ValueError(
            f"dataset holds rows at or after the data cutoff {DATA_CUTOFF}"
        )


def validate_split_set(splits: dict[str, WalkForwardSplit]) -> None:
    """The set must be exactly train < validate < test, contiguous and disjoint."""
    if tuple(splits) != SPLIT_NAMES:
        raise ValueError(f"splits must be exactly {SPLIT_NAMES}")
    ordered = [splits[name] for name in SPLIT_NAMES]
    for earlier, later in zip(ordered[:-1], ordered[1:], strict=True):
        if earlier.stop != later.start:
            raise ValueError("splits must be contiguous and not overlap")


def split_indices(
    timestamps: np.ndarray,
    splits: dict[str, WalkForwardSplit] | None = None,
) -> dict[str, tuple[int, int]]:
    """Inclusive ``(first_row, last_row)`` of every split; rows past the cutoff refused."""
    require_before_cutoff(timestamps)
    chosen = FROZEN_SPLITS if splits is None else splits
    validate_split_set(chosen)
    return {name: chosen[name].indices(timestamps) for name in SPLIT_NAMES}


def validation_halves(first: int, last: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two disjoint halves of an inclusive row range (the second gets an odd row)."""
    if last - first < 3:
        raise ValueError("a range needs at least four rows to be halved")
    middle = first + (last - first + 1) // 2
    return (first, middle - 1), (middle, last)


def splits_payload(
    splits: dict[str, WalkForwardSplit] | None = None,
) -> list[dict[str, str]]:
    chosen = FROZEN_SPLITS if splits is None else splits
    return [chosen[name].payload() for name in SPLIT_NAMES]


__all__ = [
    "DATA_CUTOFF",
    "FROZEN_SPLITS",
    "SPLIT_NAMES",
    "TEST",
    "TRAIN",
    "VALIDATE",
    "WalkForwardSplit",
    "require_before_cutoff",
    "split_indices",
    "splits_payload",
    "validate_split_set",
    "validation_halves",
]
