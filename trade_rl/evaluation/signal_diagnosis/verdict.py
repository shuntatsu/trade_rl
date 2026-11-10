"""Pure section-5 decision rule of the signal diagnosis (Issue #1146, card SD0).

The function maps already computed, saved evidence onto one of the four
preregistered labels. It reads no data, fits nothing and has no side effect; every
threshold comes from the frozen :class:`SignalDiagnosisProtocol`.

The decision unit is a *decision cell*: one feature family fitted by one model at the
primary horizon and transform. A cell has a signal only when all of these hold:

* the out-of-sample rank IC increment over the drift baseline is positive and its
  Benjamini-Hochberg adjusted p-value is at most ``q`` (equality counts);
* the same holds for the cross-sectionally demeaned target (the same-time mean
  return is removed; heterogeneous market beta is **not**);
* the rank-IC increment on the demeaned target is positive in every eligible full
  year (``min_full_years_same_sign`` of the protocol's full years). A full year is
  eligible only if it has the protocol's minimum covered months; the start-up year
  of the first fit and the truncated last year are partial, reported only, and are
  refused as decisive evidence.

Among signal cells, a cell is a *candidate* when the 95% interval of the after-cost
payoff relative to the zero-cost drift-sign policy at cost x1 lies strictly above
zero and the point estimate at cost x2 is strictly positive. This is not a
beta-adjusted alpha. At most ``max_candidates`` candidates are kept, ranked by the
interval's lower bound (descending) and then by cell id. Even the best label is a
known-used, post hoc screening label; it is never an edge, G4, G5 or live claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trade_rl.evaluation.signal_diagnosis.protocol import (
    VERDICT_CANDIDATE,
    VERDICT_INVALID,
    VERDICT_NO_SIGNAL,
    VERDICT_NOT_TRADABLE,
    SignalDiagnosisProtocol,
    SignalDiagnosisProtocolError,
)


def _fail(message: str) -> SignalDiagnosisProtocolError:
    return SignalDiagnosisProtocolError(message)


def _real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(f"{field} must be finite")
    return number


def _probability(value: object, field: str) -> float:
    number = _real(value, field)
    if not 0.0 <= number <= 1.0:
        raise _fail(f"{field} must lie in [0, 1]")
    return number


def _optional_real(value: object, field: str) -> float | None:
    return None if value is None else _real(value, field)


@dataclass(frozen=True, slots=True)
class YearStabilityEvidence:
    """Saved D5 evidence of one decisive calendar year for one decision cell.

    ``covered_months`` counts the UTC calendar months of the year in which the cell
    has enough valid daily rank-IC increments (the protocol's coverage rule).
    ``increment_positive`` is true only if the year's pooled rank-IC increment over
    the drift baseline on the persistence target is strictly greater than zero.
    """

    year: int
    covered_months: int
    increment_positive: bool

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year < 1:
            raise _fail("year must be a positive integer")
        if type(self.covered_months) is not int or not 0 <= self.covered_months <= 12:
            raise _fail("covered_months must be an integer in [0, 12]")
        if type(self.increment_positive) is not bool:
            raise _fail("increment_positive must be a boolean")


@dataclass(frozen=True, slots=True)
class ValidityFlags:
    """Any set flag makes the whole diagnosis INVALID and its results unusable."""

    leakage_detected: bool
    dataset_identity_mismatch: bool
    protocol_digest_mismatch: bool
    available_at_violation: bool
    fs1_or_warmup_contact: bool

    def __post_init__(self) -> None:
        for name in (
            "leakage_detected",
            "dataset_identity_mismatch",
            "protocol_digest_mismatch",
            "available_at_violation",
            "fs1_or_warmup_contact",
        ):
            if type(getattr(self, name)) is not bool:
                raise _fail(f"{name} must be a boolean")

    def raised(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "leakage_detected",
                "dataset_identity_mismatch",
                "protocol_digest_mismatch",
                "available_at_violation",
                "fs1_or_warmup_contact",
            )
            if getattr(self, name)
        )


@dataclass(frozen=True, slots=True)
class DecisionCellEvidence:
    """Saved evidence of one decision cell (family x model) at the primary settings.

    ``rank_ic_*`` are the out-of-sample rank IC increment over the drift baseline and
    its BH-adjusted p-value on the raw target; ``demeaned_*`` repeat them on the
    cross-sectionally demeaned target. ``year_evidence`` is the complete D5 roster of
    the protocol's full years; it may be empty only for a cell that does not clear
    both significance tests. The three ``drift_excess_*`` values are the D4 after-cost
    payoff relative to the zero-cost drift-sign policy (95% interval at cost x1,
    point at cost x2). They are required for a cell that has a signal and may be
    absent for a cell that stopped at an upper layer.
    """

    family_id: str
    model_id: str
    rank_ic_vs_drift: float
    rank_ic_bh_adjusted_p: float
    demeaned_rank_ic: float
    demeaned_bh_adjusted_p: float
    year_evidence: tuple[YearStabilityEvidence, ...]
    drift_excess_ci_lower_cost_x1: float | None
    drift_excess_ci_upper_cost_x1: float | None
    drift_excess_point_cost_x2: float | None

    def __post_init__(self) -> None:
        for name in ("family_id", "model_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or ":" in value:
                raise _fail(f"{name} must be non-empty text without ':'")
        _real(self.rank_ic_vs_drift, "rank_ic_vs_drift")
        _probability(self.rank_ic_bh_adjusted_p, "rank_ic_bh_adjusted_p")
        _real(self.demeaned_rank_ic, "demeaned_rank_ic")
        _probability(self.demeaned_bh_adjusted_p, "demeaned_bh_adjusted_p")
        evidence = self.year_evidence
        if type(evidence) is not tuple or any(
            not isinstance(item, YearStabilityEvidence) for item in evidence
        ):
            raise _fail("year_evidence must be a tuple of YearStabilityEvidence")
        years = [item.year for item in evidence]
        if len(set(years)) != len(years):
            raise _fail("year_evidence must not repeat a year")
        lower = _optional_real(
            self.drift_excess_ci_lower_cost_x1, "drift_excess_ci_lower_cost_x1"
        )
        upper = _optional_real(
            self.drift_excess_ci_upper_cost_x1, "drift_excess_ci_upper_cost_x1"
        )
        point = _optional_real(
            self.drift_excess_point_cost_x2, "drift_excess_point_cost_x2"
        )
        present = [value is not None for value in (lower, upper, point)]
        if any(present) and not all(present):
            raise _fail("D4 evidence must be fully present or fully absent")
        if lower is not None and upper is not None and lower > upper:
            raise _fail("the cost x1 alpha interval must have lower <= upper")

    @property
    def cell_id(self) -> str:
        return f"{self.family_id}:{self.model_id}"

    @property
    def has_d4(self) -> bool:
        return self.drift_excess_ci_lower_cost_x1 is not None


@dataclass(frozen=True, slots=True)
class VerdictResult:
    verdict: str
    signal_cell_ids: tuple[str, ...]
    candidate_cell_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    claim_ceiling: str


def decision_cell_ids(protocol: SignalDiagnosisProtocol) -> tuple[str, ...]:
    """Every decision cell id in protocol order: families outer, models inner."""

    return tuple(
        f"{family.family_id}:{model_id}"
        for family in protocol.families
        for model_id in protocol.models.model_ids
    )


def _is_significant(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> bool:
    q = protocol.multiple_testing.bh_fdr_q
    return (
        cell.rank_ic_vs_drift > 0.0
        and cell.rank_ic_bh_adjusted_p <= q
        and cell.demeaned_rank_ic > 0.0
        and cell.demeaned_bh_adjusted_p <= q
    )


def _check_year_roster(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> None:
    """Year evidence is the complete full-year roster or absent; never a partial year."""

    years = [item.year for item in cell.year_evidence]
    if not years:
        return
    stability = protocol.stability
    partial = sorted(set(years) & set(stability.partial_years_reported_only))
    if partial:
        raise _fail(
            f"cell {cell.cell_id}: partial year(s) {partial} are reported only and "
            "cannot be decisive year evidence"
        )
    unknown = sorted(set(years) - set(stability.full_years))
    if unknown:
        raise _fail(f"cell {cell.cell_id}: year(s) {unknown} are not full years")
    missing = sorted(set(stability.full_years) - set(years))
    if missing:
        raise _fail(f"cell {cell.cell_id}: missing full-year evidence for {missing}")


def _eligible_positive_years(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> int:
    floor = protocol.stability.min_covered_months_per_full_year
    return sum(
        1
        for item in cell.year_evidence
        if item.covered_months >= floor and item.increment_positive
    )


def _has_ineligible_year(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> bool:
    floor = protocol.stability.min_covered_months_per_full_year
    return any(item.covered_months < floor for item in cell.year_evidence)


def _has_signal(cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol) -> bool:
    return (
        _is_significant(cell, protocol)
        and _eligible_positive_years(cell, protocol)
        >= protocol.decision_rules.min_full_years_same_sign
    )


def _is_candidate(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> bool:
    rules = protocol.decision_rules
    lower = cell.drift_excess_ci_lower_cost_x1
    point = cell.drift_excess_point_cost_x2
    if lower is None or point is None:
        raise _fail(f"D4 evidence is missing for signal cell {cell.cell_id}")
    return (
        lower > rules.candidate_ci_lower_floor
        and point > rules.candidate_stress_point_floor
    )


def _check_roster(
    protocol: SignalDiagnosisProtocol, cells: Sequence[DecisionCellEvidence]
) -> dict[str, DecisionCellEvidence]:
    expected = decision_cell_ids(protocol)
    by_id: dict[str, DecisionCellEvidence] = {}
    duplicates: list[str] = []
    for cell in cells:
        if not isinstance(cell, DecisionCellEvidence):
            raise _fail("cells must be DecisionCellEvidence records")
        if cell.cell_id in by_id:
            duplicates.append(cell.cell_id)
        by_id[cell.cell_id] = cell
    unknown = sorted(set(by_id) - set(expected))
    if unknown:
        raise _fail(f"unknown decision cell(s): {unknown}")
    if duplicates:
        raise _fail(f"duplicate decision cell(s): {sorted(set(duplicates))}")
    missing = [cell_id for cell_id in expected if cell_id not in by_id]
    if missing:
        raise _fail(f"missing decision cell(s): {missing}")
    return by_id


def classify_signal_diagnosis(
    protocol: SignalDiagnosisProtocol,
    *,
    validity: ValidityFlags,
    cells: Sequence[DecisionCellEvidence],
) -> VerdictResult:
    """Apply the preregistered table; INVALID first, then the strongest evidence."""

    if not isinstance(protocol, SignalDiagnosisProtocol):
        raise _fail("protocol must be a SignalDiagnosisProtocol")
    if not isinstance(validity, ValidityFlags):
        raise _fail("validity must be ValidityFlags")
    ceiling = protocol.claim_ceiling
    raised = validity.raised()
    if raised:
        return VerdictResult(
            verdict=VERDICT_INVALID,
            signal_cell_ids=(),
            candidate_cell_ids=(),
            reasons=tuple(f"validity flag {name} is set" for name in raised),
            claim_ceiling=ceiling,
        )
    supplied: object = cells
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise _fail("cells must be a sequence of DecisionCellEvidence")
    by_id = _check_roster(protocol, cells)
    ordered = [by_id[cell_id] for cell_id in decision_cell_ids(protocol)]

    for cell in ordered:
        _check_year_roster(cell, protocol)
        if _is_significant(cell, protocol) and not cell.year_evidence:
            raise _fail(f"year evidence is missing for significant cell {cell.cell_id}")
    signal_cells = [cell for cell in ordered if _has_signal(cell, protocol)]
    for cell in signal_cells:
        if not cell.has_d4:
            raise _fail(f"D4 evidence is missing for signal cell {cell.cell_id}")
    signal_ids = tuple(cell.cell_id for cell in signal_cells)
    q = protocol.multiple_testing.bh_fdr_q
    years = protocol.decision_rules.min_full_years_same_sign
    floor = protocol.stability.min_covered_months_per_full_year
    if not signal_cells:
        reasons = [
            "no decision cell is positive and BH-significant (q="
            f"{q}) on both the raw and the cross-sectionally demeaned target with a "
            f"positive demeaned rank-IC increment in at least {years} eligible full "
            "years"
        ]
        short = [
            cell.cell_id
            for cell in ordered
            if _is_significant(cell, protocol) and _has_ineligible_year(cell, protocol)
        ]
        if short:
            reasons.append(
                f"coverage: {len(short)} significant cell(s) have a full year with "
                f"fewer than {floor} covered months, which does not count"
            )
        return VerdictResult(
            verdict=VERDICT_NO_SIGNAL,
            signal_cell_ids=(),
            candidate_cell_ids=(),
            reasons=tuple(reasons),
            claim_ceiling=ceiling,
        )

    qualifying = [cell for cell in signal_cells if _is_candidate(cell, protocol)]
    if not qualifying:
        return VerdictResult(
            verdict=VERDICT_NOT_TRADABLE,
            signal_cell_ids=signal_ids,
            candidate_cell_ids=(),
            reasons=(
                f"{len(signal_cells)} signal cell(s); none has an after-cost "
                "payoff interval relative to the zero-cost drift-sign policy strictly "
                "above zero at cost x1 together with a positive point estimate at "
                "cost x2",
            ),
            claim_ceiling=ceiling,
        )

    ranked = sorted(
        qualifying,
        key=lambda cell: (-(cell.drift_excess_ci_lower_cost_x1 or 0.0), cell.cell_id),
    )
    cap = protocol.budget.max_candidates
    kept = ranked[:cap]
    reasons = [
        f"{len(qualifying)} cell(s) qualify as candidates for a separate "
        "prospective validation request"
    ]
    if len(qualifying) > cap:
        reasons.append(
            f"candidate cap {cap} applied; kept the highest interval lower bounds "
            f"and cut {len(qualifying) - cap} cell(s)"
        )
    return VerdictResult(
        verdict=VERDICT_CANDIDATE,
        signal_cell_ids=signal_ids,
        candidate_cell_ids=tuple(cell.cell_id for cell in kept),
        reasons=tuple(reasons),
        claim_ceiling=ceiling,
    )


__all__ = [
    "DecisionCellEvidence",
    "ValidityFlags",
    "VerdictResult",
    "YearStabilityEvidence",
    "classify_signal_diagnosis",
    "decision_cell_ids",
]
