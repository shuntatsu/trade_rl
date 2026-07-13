"""Chronological aggregation of outer-OOS fold return series."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.evaluation.series import ReturnSeries


@dataclass(frozen=True, slots=True)
class FoldOOSResult:
    """One fold's sealed outer-OOS return series and source range."""

    fold_index: int
    start: int
    stop: int
    returns: ReturnSeries

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("OOS range must be a valid half-open range")
        if self.stop - self.start != len(self.returns.values):
            raise ValueError("OOS range length must equal return series length")


@dataclass(frozen=True, slots=True)
class StitchedOOS:
    """Chronologically stitched OOS series with fold provenance."""

    returns: ReturnSeries
    fold_indices: tuple[int, ...]
    boundaries: tuple[tuple[int, int], ...]


def stitch_oos(results: tuple[FoldOOSResult, ...]) -> StitchedOOS:
    """Sort and concatenate compatible, non-overlapping OOS fold results."""

    if not results:
        raise ValueError("at least one OOS fold result is required")
    ordered = tuple(
        sorted(results, key=lambda result: (result.start, result.fold_index))
    )
    fold_indices = tuple(result.fold_index for result in ordered)
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("fold indices must be unique")

    first = ordered[0]
    previous = first
    combined: list[float] = list(first.returns.values)
    boundaries: list[tuple[int, int]] = [(first.start, first.stop)]

    for current in ordered[1:]:
        if current.start < previous.stop:
            raise ValueError("OOS fold ranges overlap")
        if current.returns.kind is not first.returns.kind:
            raise ValueError("OOS return series kind mismatch")
        if current.returns.periods_per_year != first.returns.periods_per_year:
            raise ValueError("OOS return series annualization mismatch")
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
    )
