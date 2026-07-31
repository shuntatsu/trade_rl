"""Validation-selection and sealed-test decision artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation._stage_a_zero_shot_gate_values import (
    STAGE_A_SEALED_TEST_DECISION_SCHEMA,
    STAGE_A_VALIDATION_SELECTION_SCHEMA,
    StageACandidateSummary,
    _finite,
    _fraction,
    _summary_meets_gate,
)

@dataclass(frozen=True, slots=True)
class StageAValidationSelection:
    """Validation-only deterministic candidate selection artifact."""

    plan_digest: str
    validation_evidence_digest: str
    candidate_summaries: tuple[StageACandidateSummary, ...]
    minimum_lower_bound: float
    minimum_worst_triplet_excess: float
    minimum_worst_seed_excess: float
    minimum_triplet_pass_fraction: float
    selected_candidate_id: str | None
    passed: bool
    reason: str
    schema_version: str = STAGE_A_VALIDATION_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_VALIDATION_SELECTION_SCHEMA:
            raise ValueError("unsupported Stage A validation selection schema")
        require_sha256(self.plan_digest, field="stage_a_selection.plan_digest")
        require_sha256(
            self.validation_evidence_digest,
            field="stage_a_selection.validation_evidence_digest",
        )
        if not self.candidate_summaries:
            raise ValueError("Stage A validation summaries must not be empty")
        summaries = tuple(
            sorted(self.candidate_summaries, key=lambda item: item.candidate_id)
        )
        candidate_ids = tuple(item.candidate_id for item in summaries)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Stage A validation summary candidates must be unique")
        if any(item.split != "validation" for item in summaries):
            raise ValueError("Stage A selection requires validation summaries")
        if any(
            item.plan_digest != self.plan_digest
            or item.evidence_digest != self.validation_evidence_digest
            for item in summaries
        ):
            raise ValueError("Stage A validation summary identity mismatch")
        minimum_lower_bound = _finite(
            self.minimum_lower_bound, field="stage_a_selection.minimum_lower_bound"
        )
        minimum_worst_triplet = _finite(
            self.minimum_worst_triplet_excess,
            field="stage_a_selection.minimum_worst_triplet_excess",
        )
        minimum_worst_seed = _finite(
            self.minimum_worst_seed_excess,
            field="stage_a_selection.minimum_worst_seed_excess",
        )
        minimum_pass_fraction = _fraction(
            self.minimum_triplet_pass_fraction,
            field="stage_a_selection.minimum_triplet_pass_fraction",
        )
        if not isinstance(self.passed, bool):
            raise ValueError("Stage A validation passed must be boolean")
        reason = require_non_empty(self.reason, field="stage_a_selection.reason")
        eligible = tuple(
            item
            for item in summaries
            if _summary_meets_gate(
                item,
                minimum_lower_bound=minimum_lower_bound,
                minimum_worst_triplet_excess=minimum_worst_triplet,
                minimum_worst_seed_excess=minimum_worst_seed,
                minimum_triplet_pass_fraction=minimum_pass_fraction,
            )
        )
        if eligible:
            if not self.passed:
                raise ValueError("Stage A validation selection cannot fail with an eligible candidate")
            expected = min(
                eligible,
                key=lambda item: (
                    -item.lower_confidence_bound,
                    -item.worst_triplet_excess_log_growth,
                    -item.worst_seed_excess_log_growth,
                    -item.mean_excess_log_growth,
                    item.candidate_id,
                ),
            )
            if self.selected_candidate_id != expected.candidate_id:
                raise ValueError("Stage A validation selection must use the deterministic winner")
            if reason != "candidate_selected_by_validation_gate":
                raise ValueError("Stage A validation selection reason mismatch")
        else:
            if self.passed or self.selected_candidate_id is not None:
                raise ValueError("Stage A validation selection cannot pass without an eligible candidate")
            if reason != "no_candidate_met_validation_gate":
                raise ValueError("Stage A validation selection reason mismatch")
        object.__setattr__(self, "candidate_summaries", summaries)
        object.__setattr__(self, "minimum_lower_bound", minimum_lower_bound)
        object.__setattr__(self, "minimum_worst_triplet_excess", minimum_worst_triplet)
        object.__setattr__(self, "minimum_worst_seed_excess", minimum_worst_seed)
        object.__setattr__(self, "minimum_triplet_pass_fraction", minimum_pass_fraction)
        object.__setattr__(self, "reason", reason)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A validation selection digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def summary(self, candidate_id: str) -> StageACandidateSummary:
        resolved = require_non_empty(candidate_id, field="stage_a_candidate_id")
        for summary in self.candidate_summaries:
            if summary.candidate_id == resolved:
                return summary
        raise ValueError("Stage A selection candidate summary is missing")

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_summaries": tuple(item.to_json_dict() for item in self.candidate_summaries),
            "minimum_lower_bound": self.minimum_lower_bound,
            "minimum_triplet_pass_fraction": self.minimum_triplet_pass_fraction,
            "minimum_worst_seed_excess": self.minimum_worst_seed_excess,
            "minimum_worst_triplet_excess": self.minimum_worst_triplet_excess,
            "passed": self.passed,
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "validation_evidence_digest": self.validation_evidence_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageASealedTestDecision:
    """Final unseen-symbol gate for the already selected validation winner."""

    plan_digest: str
    validation_selection_digest: str
    test_evidence_digest: str
    selected_candidate_id: str
    candidate_summary: StageACandidateSummary
    minimum_lower_bound: float
    minimum_worst_triplet_excess: float
    minimum_worst_seed_excess: float
    minimum_triplet_pass_fraction: float
    passed: bool
    reason: str
    schema_version: str = STAGE_A_SEALED_TEST_DECISION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_SEALED_TEST_DECISION_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test decision schema")
        require_sha256(self.plan_digest, field="stage_a_test.plan_digest")
        require_sha256(
            self.validation_selection_digest,
            field="stage_a_test.validation_selection_digest",
        )
        require_sha256(self.test_evidence_digest, field="stage_a_test.test_evidence_digest")
        candidate_id = require_non_empty(
            self.selected_candidate_id, field="stage_a_test.selected_candidate_id"
        )
        if self.candidate_summary.split != "test":
            raise ValueError("Stage A sealed-test decision requires a test summary")
        if (
            self.candidate_summary.plan_digest != self.plan_digest
            or self.candidate_summary.evidence_digest != self.test_evidence_digest
            or self.candidate_summary.candidate_id != candidate_id
        ):
            raise ValueError("Stage A sealed-test summary identity mismatch")
        minimum_lower_bound = _finite(
            self.minimum_lower_bound, field="stage_a_test.minimum_lower_bound"
        )
        minimum_worst_triplet = _finite(
            self.minimum_worst_triplet_excess,
            field="stage_a_test.minimum_worst_triplet_excess",
        )
        minimum_worst_seed = _finite(
            self.minimum_worst_seed_excess,
            field="stage_a_test.minimum_worst_seed_excess",
        )
        minimum_pass_fraction = _fraction(
            self.minimum_triplet_pass_fraction,
            field="stage_a_test.minimum_triplet_pass_fraction",
        )
        if not isinstance(self.passed, bool):
            raise ValueError("Stage A sealed-test passed must be boolean")
        reason = require_non_empty(self.reason, field="stage_a_test.reason")
        expected_passed = _summary_meets_gate(
            self.candidate_summary,
            minimum_lower_bound=minimum_lower_bound,
            minimum_worst_triplet_excess=minimum_worst_triplet,
            minimum_worst_seed_excess=minimum_worst_seed,
            minimum_triplet_pass_fraction=minimum_pass_fraction,
        )
        expected_reason = (
            "selected_candidate_met_test_gate"
            if expected_passed
            else "selected_candidate_missed_test_gate"
        )
        if self.passed != expected_passed or reason != expected_reason:
            raise ValueError("Stage A sealed-test outcome mismatch")
        object.__setattr__(self, "selected_candidate_id", candidate_id)
        object.__setattr__(self, "minimum_lower_bound", minimum_lower_bound)
        object.__setattr__(self, "minimum_worst_triplet_excess", minimum_worst_triplet)
        object.__setattr__(self, "minimum_worst_seed_excess", minimum_worst_seed)
        object.__setattr__(self, "minimum_triplet_pass_fraction", minimum_pass_fraction)
        object.__setattr__(self, "reason", reason)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A sealed-test decision digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_summary": self.candidate_summary.to_json_dict(),
            "minimum_lower_bound": self.minimum_lower_bound,
            "minimum_triplet_pass_fraction": self.minimum_triplet_pass_fraction,
            "minimum_worst_seed_excess": self.minimum_worst_seed_excess,
            "minimum_worst_triplet_excess": self.minimum_worst_triplet_excess,
            "passed": self.passed,
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "test_evidence_digest": self.test_evidence_digest,
            "validation_selection_digest": self.validation_selection_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


