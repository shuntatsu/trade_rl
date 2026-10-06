"""Pure preregistered winner gate for Issue #810 allocation comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationCandidateSummary,
    AllocationComparisonContract,
    AllocationComparisonEvidence,
    AllocationValidity,
    summarize_allocation_comparison,
    validate_allocation_comparison_evidence,
)

_TIE_BREAK = (
    AllocationCandidateKind.NONRL,
    AllocationCandidateKind.RESIDUAL_PPO,
    AllocationCandidateKind.DIRECT_PPO,
)


class AllocationSelectionOutcome(StrEnum):
    WINNER = "WINNER"
    NO_WINNER = "NO_WINNER"
    INVALID = "INVALID"


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


@dataclass(frozen=True, slots=True)
class AllocationSelectionRule:
    comparison_contract_digest: str
    minimum_incremental_vs_nonrl_base: float
    minimum_incremental_vs_cash_base: float

    def __post_init__(self) -> None:
        require_sha256(
            self.comparison_contract_digest, field="comparison_contract_digest"
        )
        object.__setattr__(
            self,
            "minimum_incremental_vs_nonrl_base",
            _finite_nonnegative(
                self.minimum_incremental_vs_nonrl_base,
                "minimum_incremental_vs_nonrl_base",
            ),
        )
        object.__setattr__(
            self,
            "minimum_incremental_vs_cash_base",
            _finite_nonnegative(
                self.minimum_incremental_vs_cash_base,
                "minimum_incremental_vs_cash_base",
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_selection_rule_v1",
            "selection_id": "after_cost_terminal_profit_v1",
            "comparison_contract_digest": self.comparison_contract_digest,
            "minimum_incremental_vs_nonrl_base": self.minimum_incremental_vs_nonrl_base,
            "minimum_incremental_vs_cash_base": self.minimum_incremental_vs_cash_base,
            "tie_break_order": [value.value for value in _TIE_BREAK],
            "strict_thresholds": True,
            "threshold_comparison": "strict_gt_isclose_1e-12_v1",
            "required_validity_policy": "all_candidates_and_cash_v1",
            "required_risk_policy": "candidate_eligibility_v1",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationSelectionRule:
        data = _mapping(
            value,
            {
                "schema",
                "selection_id",
                "comparison_contract_digest",
                "minimum_incremental_vs_nonrl_base",
                "minimum_incremental_vs_cash_base",
                "tie_break_order",
                "strict_thresholds",
                "threshold_comparison",
                "required_validity_policy",
                "required_risk_policy",
            },
            "allocation selection rule",
        )
        if data["schema"] != "allocation_selection_rule_v1":
            raise ValueError("unsupported allocation selection rule schema")
        if data["selection_id"] != "after_cost_terminal_profit_v1":
            raise ValueError("unsupported allocation selection rule identity")
        if data["tie_break_order"] != [value.value for value in _TIE_BREAK]:
            raise ValueError("allocation selection tie-break order changed")
        if data["strict_thresholds"] is not True:
            raise ValueError("allocation selection thresholds must remain strict")
        if data["threshold_comparison"] != "strict_gt_isclose_1e-12_v1":
            raise ValueError("allocation selection threshold comparison changed")
        if data["required_validity_policy"] != "all_candidates_and_cash_v1":
            raise ValueError("allocation selection validity policy changed")
        if data["required_risk_policy"] != "candidate_eligibility_v1":
            raise ValueError("allocation selection risk policy changed")
        try:
            result = cls(
                comparison_contract_digest=data["comparison_contract_digest"],
                minimum_incremental_vs_nonrl_base=data[
                    "minimum_incremental_vs_nonrl_base"
                ],
                minimum_incremental_vs_cash_base=data[
                    "minimum_incremental_vs_cash_base"
                ],
            )
        except TypeError as error:
            raise ValueError("allocation selection rule is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("allocation selection rule differs from canonical form")
        return result


@dataclass(frozen=True, slots=True)
class AllocationCashReference:
    contract_digest: str
    scenario: str
    control_evidence_digest: str
    oos_source_digest: str
    opening_state_digest: str
    terminal_profit_rate: float
    max_drawdown: float
    validity: AllocationValidity
    coverage_complete: bool
    termination_reason: str | None

    def __post_init__(self) -> None:
        for field in (
            "contract_digest",
            "control_evidence_digest",
            "oos_source_digest",
            "opening_state_digest",
        ):
            require_sha256(getattr(self, field), field=field)
        object.__setattr__(
            self, "scenario", require_non_empty(self.scenario, field="scenario")
        )
        object.__setattr__(
            self,
            "terminal_profit_rate",
            _finite(self.terminal_profit_rate, "terminal_profit_rate"),
        )
        drawdown = _finite(self.max_drawdown, "max_drawdown")
        if drawdown < 0.0:
            raise ValueError("cash reference max_drawdown must be non-negative")
        object.__setattr__(self, "max_drawdown", drawdown)
        if type(self.validity) is not AllocationValidity:
            raise ValueError("cash reference validity must be AllocationValidity")
        if type(self.coverage_complete) is not bool:
            raise ValueError("cash reference coverage_complete must be a native bool")
        if self.termination_reason is not None:
            object.__setattr__(
                self,
                "termination_reason",
                require_non_empty(self.termination_reason, field="termination_reason"),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_cash_reference_v1",
            "contract_digest": self.contract_digest,
            "scenario": self.scenario,
            "control_evidence_digest": self.control_evidence_digest,
            "oos_source_digest": self.oos_source_digest,
            "opening_state_digest": self.opening_state_digest,
            "terminal_profit_rate": self.terminal_profit_rate,
            "max_drawdown": self.max_drawdown,
            "validity": self.validity.value,
            "coverage_complete": self.coverage_complete,
            "termination_reason": self.termination_reason,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


@dataclass(frozen=True, slots=True)
class AllocationCandidateEligibility:
    candidate: AllocationCandidateKind
    score: float
    incremental_vs_nonrl: float
    incremental_vs_cash: float
    eligible: bool
    reasons: tuple[str, ...]
    diagnostic_failures: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "score": self.score,
            "incremental_vs_nonrl": self.incremental_vs_nonrl,
            "incremental_vs_cash": self.incremental_vs_cash,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "diagnostic_failures": list(self.diagnostic_failures),
        }


@dataclass(frozen=True, slots=True)
class AllocationSelectionDecision:
    rule_digest: str
    comparison_contract_digest: str
    comparison_summary_digest: str
    cash_reference_set_digest: str
    outcome: AllocationSelectionOutcome
    selected_candidate: AllocationCandidateKind | None
    winner_score: float | None
    eligible_candidates: tuple[AllocationCandidateKind, ...]
    candidates: tuple[AllocationCandidateEligibility, ...]
    invalid_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "rule_digest",
            "comparison_contract_digest",
            "comparison_summary_digest",
            "cash_reference_set_digest",
        ):
            require_sha256(getattr(self, field), field=field)
        if type(self.outcome) is not AllocationSelectionOutcome:
            raise ValueError("selection outcome must be AllocationSelectionOutcome")
        if self.outcome is AllocationSelectionOutcome.WINNER:
            if self.selected_candidate is None or self.winner_score is None:
                raise ValueError("WINNER requires a selected candidate and score")
            if self.selected_candidate not in self.eligible_candidates:
                raise ValueError("WINNER candidate must be eligible")
        else:
            if self.selected_candidate is not None or self.winner_score is not None:
                raise ValueError("non-WINNER outcome forbids selected candidate/score")
        if self.outcome is AllocationSelectionOutcome.INVALID:
            if not self.invalid_reasons or self.eligible_candidates:
                raise ValueError(
                    "INVALID requires reasons and forbids eligible candidates"
                )
        elif self.invalid_reasons:
            raise ValueError("valid selection outcome forbids invalid reasons")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_selection_decision_v1",
            "rule_digest": self.rule_digest,
            "comparison_contract_digest": self.comparison_contract_digest,
            "comparison_summary_digest": self.comparison_summary_digest,
            "cash_reference_set_digest": self.cash_reference_set_digest,
            "outcome": self.outcome.value,
            "selected_candidate": (
                None
                if self.selected_candidate is None
                else self.selected_candidate.value
            ),
            "winner_score": self.winner_score,
            "eligible_candidates": [value.value for value in self.eligible_candidates],
            "candidates": [value.payload() for value in self.candidates],
            "invalid_reasons": list(self.invalid_reasons),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


def _validate_cash_references(
    contract: AllocationComparisonContract,
    evidence: tuple[AllocationComparisonEvidence, ...],
    cash: tuple[AllocationCashReference, ...],
) -> tuple[AllocationCashReference, ...]:
    if (
        type(cash) is not tuple
        or len(cash) != len(contract.scenarios)
        or any(type(value) is not AllocationCashReference for value in cash)
    ):
        raise ValueError("cash reference matrix must contain one row per scenario")
    by_name: dict[str, AllocationCashReference] = {}
    for row in cash:
        if row.contract_digest != contract.digest:
            raise ValueError("cash reference belongs to another comparison contract")
        if row.scenario in by_name:
            raise ValueError("cash reference matrix contains a duplicate scenario")
        by_name[row.scenario] = row
    expected_names = tuple(value.name for value in contract.scenarios)
    if set(by_name) != set(expected_names):
        raise ValueError("cash reference matrix is incomplete")

    for scenario in contract.scenarios:
        row = by_name[scenario.name]
        comparison_rows = tuple(
            value for value in evidence if value.scenario == scenario.name
        )
        if not comparison_rows:
            raise ValueError(
                "comparison matrix lacks scenario needed by cash reference"
            )
        expected_source = comparison_rows[0].oos_source_digest
        expected_opening = comparison_rows[0].opening_state_digest
        if row.oos_source_digest != expected_source:
            raise ValueError("cash reference OOS source differs from comparison")
        if row.opening_state_digest != expected_opening:
            raise ValueError("cash reference opening state differs from comparison")
    return tuple(by_name[name] for name in expected_names)


def _cash_set_digest(rows: tuple[AllocationCashReference, ...]) -> str:
    return content_digest(
        {
            "schema": "allocation_cash_reference_set_v1",
            "references": [value.payload() for value in rows],
        }
    )


def _strictly_above(value: float, threshold: float) -> bool:
    return value > threshold and not math.isclose(
        value,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _candidate_eligibility(
    summary: AllocationCandidateSummary,
    *,
    cash_base_profit: float,
    rule: AllocationSelectionRule,
) -> AllocationCandidateEligibility:
    reasons: list[str] = []
    incremental_cash = summary.median_base_profit_rate - cash_base_profit
    if not summary.risk_execution_passed:
        reasons.append("risk_execution")
    if not _strictly_above(
        incremental_cash,
        rule.minimum_incremental_vs_cash_base,
    ):
        reasons.append("cash")
    if summary.candidate is not AllocationCandidateKind.NONRL and not _strictly_above(
        summary.median_incremental_vs_nonrl_base,
        rule.minimum_incremental_vs_nonrl_base,
    ):
        reasons.append("nonrl")
    return AllocationCandidateEligibility(
        candidate=summary.candidate,
        score=summary.median_base_profit_rate,
        incremental_vs_nonrl=summary.median_incremental_vs_nonrl_base,
        incremental_vs_cash=incremental_cash,
        eligible=summary.validity_passed and not reasons,
        reasons=tuple(reasons),
        diagnostic_failures=summary.diagnostic_failures,
    )


def select_allocation_candidate(
    contract: AllocationComparisonContract,
    evidence: tuple[AllocationComparisonEvidence, ...],
    cash_references: tuple[AllocationCashReference, ...],
    rule: AllocationSelectionRule,
) -> AllocationSelectionDecision:
    """Apply the frozen development gate; this function has no Study/final authority."""
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("selection requires AllocationComparisonContract")
    if type(rule) is not AllocationSelectionRule:
        raise ValueError("selection requires AllocationSelectionRule")
    if rule.comparison_contract_digest != contract.digest:
        raise ValueError("selection rule belongs to another comparison contract")

    normalized = validate_allocation_comparison_evidence(contract, evidence)
    summary = summarize_allocation_comparison(contract, normalized)
    cash = _validate_cash_references(contract, normalized, cash_references)
    summary_digest = content_digest(summary.payload())
    cash_digest = _cash_set_digest(cash)

    invalid_reasons: list[str] = []
    if any(not value.validity_passed for value in summary.candidates):
        invalid_reasons.append("required_candidate_evidence_invalid")
    required_names = {
        scenario.name for scenario in contract.scenarios if scenario.required
    }
    required_cash = tuple(value for value in cash if value.scenario in required_names)
    if any(
        value.validity is not AllocationValidity.VALID
        or not value.coverage_complete
        or value.termination_reason is not None
        or value.max_drawdown > contract.maximum_drawdown
        for value in required_cash
    ):
        invalid_reasons.append("required_cash_reference_invalid")

    cash_base = next(value for value in cash if value.scenario == "base")
    candidates = tuple(
        _candidate_eligibility(
            value,
            cash_base_profit=cash_base.terminal_profit_rate,
            rule=rule,
        )
        for value in summary.candidates
    )

    if invalid_reasons:
        return AllocationSelectionDecision(
            rule_digest=rule.digest,
            comparison_contract_digest=contract.digest,
            comparison_summary_digest=summary_digest,
            cash_reference_set_digest=cash_digest,
            outcome=AllocationSelectionOutcome.INVALID,
            selected_candidate=None,
            winner_score=None,
            eligible_candidates=(),
            candidates=tuple(
                AllocationCandidateEligibility(
                    candidate=value.candidate,
                    score=value.score,
                    incremental_vs_nonrl=value.incremental_vs_nonrl,
                    incremental_vs_cash=value.incremental_vs_cash,
                    eligible=False,
                    reasons=value.reasons,
                    diagnostic_failures=value.diagnostic_failures,
                )
                for value in candidates
            ),
            invalid_reasons=tuple(invalid_reasons),
        )

    eligible = tuple(value for value in candidates if value.eligible)
    eligible_kinds = tuple(value.candidate for value in eligible)
    if not eligible:
        return AllocationSelectionDecision(
            rule_digest=rule.digest,
            comparison_contract_digest=contract.digest,
            comparison_summary_digest=summary_digest,
            cash_reference_set_digest=cash_digest,
            outcome=AllocationSelectionOutcome.NO_WINNER,
            selected_candidate=None,
            winner_score=None,
            eligible_candidates=(),
            candidates=candidates,
            invalid_reasons=(),
        )

    priority = {candidate: index for index, candidate in enumerate(_TIE_BREAK)}
    winner = min(
        eligible,
        key=lambda value: (-value.score, priority[value.candidate]),
    )
    return AllocationSelectionDecision(
        rule_digest=rule.digest,
        comparison_contract_digest=contract.digest,
        comparison_summary_digest=summary_digest,
        cash_reference_set_digest=cash_digest,
        outcome=AllocationSelectionOutcome.WINNER,
        selected_candidate=winner.candidate,
        winner_score=winner.score,
        eligible_candidates=eligible_kinds,
        candidates=candidates,
        invalid_reasons=(),
    )


__all__ = [
    "AllocationCandidateEligibility",
    "AllocationCashReference",
    "AllocationSelectionDecision",
    "AllocationSelectionOutcome",
    "AllocationSelectionRule",
    "select_allocation_candidate",
]
