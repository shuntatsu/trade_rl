"""Pure section-5 decision rule of the signal diagnosis (Issue #1146, card SD0).

The function maps already computed, saved evidence onto one of the four
preregistered labels. It reads no data, fits nothing and has no side effect; every
threshold comes from the frozen :class:`SignalDiagnosisProtocol`.

The decision unit is a *decision cell*: one feature family fitted by one model at the
primary horizon and transform. A cell has a signal only when all of these hold:

* the out-of-sample rank IC increment over the drift baseline is positive and its
  Benjamini-Hochberg adjusted p-value is at most ``q`` (equality counts);
* the same holds for the beta-removed target;
* the sign of the pooled estimate agrees in at least four of the five full years.

Among signal cells, a cell is a *candidate* when the 95% interval of the after-cost
alpha at cost x1 lies strictly above zero and the point estimate at cost x2 is
strictly positive. At most ``max_candidates`` candidates are kept, ranked by the
interval's lower bound (descending) and then by cell id. Even the best label is a
known-used, post hoc screening label; it is never an edge, G4, G5 or live claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trade_rl.evaluation.signal_diagnosis.protocol import (
    FULL_YEARS,
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
    its BH-adjusted p-value on the raw target; ``beta_removed_*`` repeat them on the
    beta-removed target. ``years_same_sign`` counts the full years whose sign agrees
    with the pooled estimate. The three ``alpha_*`` values are the D4 after-cost
    alpha (95% interval at cost x1, point at cost x2). They are required for a cell
    that has a signal and may be absent for a cell that stopped at an upper layer.
    """

    family_id: str
    model_id: str
    rank_ic_vs_drift: float
    rank_ic_bh_adjusted_p: float
    beta_removed_rank_ic: float
    beta_removed_bh_adjusted_p: float
    years_same_sign: int
    alpha_ci_lower_cost_x1: float | None
    alpha_ci_upper_cost_x1: float | None
    alpha_point_cost_x2: float | None

    def __post_init__(self) -> None:
        for name in ("family_id", "model_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or ":" in value:
                raise _fail(f"{name} must be non-empty text without ':'")
        _real(self.rank_ic_vs_drift, "rank_ic_vs_drift")
        _probability(self.rank_ic_bh_adjusted_p, "rank_ic_bh_adjusted_p")
        _real(self.beta_removed_rank_ic, "beta_removed_rank_ic")
        _probability(self.beta_removed_bh_adjusted_p, "beta_removed_bh_adjusted_p")
        years = self.years_same_sign
        if type(years) is not int or not 0 <= years <= len(FULL_YEARS):
            raise _fail(f"years_same_sign must be an integer in [0, {len(FULL_YEARS)}]")
        lower = _optional_real(self.alpha_ci_lower_cost_x1, "alpha_ci_lower_cost_x1")
        upper = _optional_real(self.alpha_ci_upper_cost_x1, "alpha_ci_upper_cost_x1")
        point = _optional_real(self.alpha_point_cost_x2, "alpha_point_cost_x2")
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
        return self.alpha_ci_lower_cost_x1 is not None


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


def _has_signal(cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol) -> bool:
    q = protocol.multiple_testing.bh_fdr_q
    return (
        cell.rank_ic_vs_drift > 0.0
        and cell.rank_ic_bh_adjusted_p <= q
        and cell.beta_removed_rank_ic > 0.0
        and cell.beta_removed_bh_adjusted_p <= q
        and cell.years_same_sign >= protocol.decision_rules.min_full_years_same_sign
    )


def _is_candidate(
    cell: DecisionCellEvidence, protocol: SignalDiagnosisProtocol
) -> bool:
    rules = protocol.decision_rules
    lower = cell.alpha_ci_lower_cost_x1
    point = cell.alpha_point_cost_x2
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

    signal_cells = [cell for cell in ordered if _has_signal(cell, protocol)]
    for cell in signal_cells:
        if not cell.has_d4:
            raise _fail(f"D4 evidence is missing for signal cell {cell.cell_id}")
    signal_ids = tuple(cell.cell_id for cell in signal_cells)
    q = protocol.multiple_testing.bh_fdr_q
    years = protocol.decision_rules.min_full_years_same_sign
    if not signal_cells:
        return VerdictResult(
            verdict=VERDICT_NO_SIGNAL,
            signal_cell_ids=(),
            candidate_cell_ids=(),
            reasons=(
                "no decision cell is positive and BH-significant (q="
                f"{q}) on both the raw and the beta-removed target with the same "
                f"sign in at least {years} full years",
            ),
            claim_ceiling=ceiling,
        )

    qualifying = [cell for cell in signal_cells if _is_candidate(cell, protocol)]
    if not qualifying:
        return VerdictResult(
            verdict=VERDICT_NOT_TRADABLE,
            signal_cell_ids=signal_ids,
            candidate_cell_ids=(),
            reasons=(
                f"{len(signal_cells)} signal cell(s); none has an after-cost alpha "
                "interval strictly above zero at cost x1 together with a positive "
                "point estimate at cost x2",
            ),
            claim_ceiling=ceiling,
        )

    ranked = sorted(
        qualifying,
        key=lambda cell: (-(cell.alpha_ci_lower_cost_x1 or 0.0), cell.cell_id),
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
    "classify_signal_diagnosis",
    "decision_cell_ids",
]
