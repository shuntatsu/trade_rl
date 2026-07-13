"""Chronological aggregation of outer-OOS fold return series."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trade_rl.evaluation.series import ReturnSeries


class StitchMode(str, Enum):
    """Account-state identity represented by a stitched OOS series."""

    INDEPENDENT_FOLDS = "independent_folds"
    CONTINUOUS_ACCOUNT = "continuous_account"


@dataclass(frozen=True, slots=True)
class FoldOOSResult:
    """One fold's sealed outer-OOS return series and source range."""

    fold_index: int
    start: int
    stop: int
    returns: ReturnSeries
    opening_state_digest: str | None = None
    closing_state_digest: str | None = None

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("OOS range must be a valid half-open range")
        if self.stop - self.start != len(self.returns.values):
            raise ValueError("OOS range length must equal return series length")
        for field_name, value in (
            ("opening_state_digest", self.opening_state_digest),
            ("closing_state_digest", self.closing_state_digest),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class StitchedOOS:
    """Chronologically stitched OOS series with explicit account identity."""

    returns: ReturnSeries
    fold_indices: tuple[int, ...]
    boundaries: tuple[tuple[int, int], ...]
    mode: StitchMode
    gaps: tuple[tuple[int, int], ...]


def stitch_oos(
    results: tuple[FoldOOSResult, ...],
    *,
    mode: StitchMode = StitchMode.INDEPENDENT_FOLDS,
) -> StitchedOOS:
    """Sort and concatenate compatible OOS folds without hiding resets or gaps."""

    if not results:
        raise ValueError("at least one OOS fold result is required")
    ordered = tuple(
        sorted(results, key=lambda result: (result.start, result.fold_index))
    )
    fold_indices = tuple(result.fold_index for result in ordered)
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("fold indices must be unique")

    first = ordered[0]
    if mode is StitchMode.CONTINUOUS_ACCOUNT and (
        first.opening_state_digest is None or first.closing_state_digest is None
    ):
        raise ValueError("continuous account folds require state handoff digests")

    previous = first
    combined: list[float] = list(first.returns.values)
    boundaries: list[tuple[int, int]] = [(first.start, first.stop)]
    gaps: list[tuple[int, int]] = []

    for current in ordered[1:]:
        if current.start < previous.stop:
            raise ValueError("OOS fold ranges overlap")
        if current.returns.kind is not first.returns.kind:
            raise ValueError("OOS return series kind mismatch")
        if current.returns.periods_per_year != first.returns.periods_per_year:
            raise ValueError("OOS return series annualization mismatch")
        if current.start > previous.stop:
            gaps.append((previous.stop, current.start))
        if mode is StitchMode.CONTINUOUS_ACCOUNT:
            if current.start != previous.stop:
                raise ValueError("continuous account OOS ranges must be contiguous")
            if (
                previous.closing_state_digest is None
                or current.opening_state_digest is None
                or current.closing_state_digest is None
            ):
                raise ValueError("continuous account folds require state handoff digests")
            if previous.closing_state_digest != current.opening_state_digest:
                raise ValueError("continuous account state handoff identity mismatch")
        combined.extend(current.returns.values)
        boundaries.append((current.start, current.stop))
        previous = current

    return StitchedOOS(
        returns=ReturnSeries(
            values=tuple(combined),
            kind=first.returns.kind,
            periods_per_year=first.returns.periods_per_year,
        ),
        fold_indices=fold_indices,
        boundaries=tuple(boundaries),
        mode=mode,
        gaps=tuple(gaps),
    )
