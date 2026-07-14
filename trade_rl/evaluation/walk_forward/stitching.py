"""Chronological aggregation of outer-OOS fold return series."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnSeries




@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Fold-local execution totals retained with return evidence."""

    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    borrow_cost: float = 0.0
    dividend_pnl: float = 0.0
    cash_interest: float = 0.0
    n_trades: int = 0
    rebalance_events: int = 0
    max_participation: float = 0.0
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
            ("funding_pnl", self.funding_pnl),
            ("borrow_cost", self.borrow_cost),
            ("dividend_pnl", self.dividend_pnl),
            ("cash_interest", self.cash_interest),
            ("max_participation", self.max_participation),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.turnover_total < 0.0 or self.total_cost < 0.0 or self.borrow_cost < 0.0:
            raise ValueError("execution costs and turnover must be non-negative")
        if self.n_trades < 0 or self.rebalance_events < 0:
            raise ValueError("execution counters must be non-negative")

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
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    opening_state_digest: str | None = None
    closing_state_digest: str | None = None
    evidence: ExecutionEvidence = ExecutionEvidence()

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
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)


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
        if (current.returns.elapsed_years is None) != (first.returns.elapsed_years is None):
            raise ValueError("OOS elapsed-time metadata mismatch")
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
                raise ValueError(
                    "continuous account folds require state handoff digests"
                )
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
            elapsed_years=(
                None
                if first.returns.elapsed_years is None
                else sum(result.returns.elapsed_years or 0.0 for result in ordered)
            ),
        ),
        fold_indices=fold_indices,
        boundaries=tuple(boundaries),
        mode=mode,
        gaps=tuple(gaps),
        diagnostics=ExecutionDiagnostics.combine(
            result.diagnostics for result in ordered
        ),
    )
