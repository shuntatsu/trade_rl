"""Typed contracts for Stage A zero-shot evaluation orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAEvaluationEvidence,
    StageAEvaluationSplit,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    StageASealedTestDecision,
    StageAValidationSelection,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestAccessRecord

_STAGE_A_CELL_REQUEST_SCHEMA = "stage_a_evaluation_cell_request_v1"
_STAGE_A_TEST_SCHEDULE_SCHEMA = "stage_a_test_schedule_v1"
_STAGE_A_VALIDATION_RUN_SCHEMA = "stage_a_validation_run_v1"
_STAGE_A_SEALED_TEST_RUN_SCHEMA = "stage_a_sealed_test_run_v1"
_SPLITS = {"validation", "test"}


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class StageAEvaluationCellRequest:
    """One exact baseline or policy evaluation request."""

    plan_digest: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    candidate_id: str | None
    checkpoint_digest: str | None
    dataset_identity: str
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    schema_version: str = _STAGE_A_CELL_REQUEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_CELL_REQUEST_SCHEMA:
            raise ValueError("unsupported Stage A evaluation cell request schema")
        if self.split not in _SPLITS:
            raise ValueError("Stage A evaluation cell split is invalid")
        for field_name, value in (
            ("plan_digest", self.plan_digest),
            ("triplet_id", self.triplet_id),
            ("dataset_identity", self.dataset_identity),
            ("feature_identity", self.feature_identity),
            ("execution_identity", self.execution_identity),
            ("evaluation_identity", self.evaluation_identity),
        ):
            require_sha256(value, field=f"stage_a_cell_request.{field_name}")
        fold = _non_negative_int(self.fold, field="stage_a_cell_request.fold")
        seed = _non_negative_int(self.seed, field="stage_a_cell_request.seed")
        if (self.candidate_id is None) != (self.checkpoint_digest is None):
            raise ValueError(
                "Stage A policy request requires candidate and checkpoint identities"
            )
        candidate_id = self.candidate_id
        if candidate_id is not None:
            candidate_id = require_non_empty(
                candidate_id, field="stage_a_cell_request.candidate_id"
            )
            checkpoint_digest = self.checkpoint_digest
            if checkpoint_digest is None:
                raise ValueError(
                    "Stage A policy request requires candidate and checkpoint identities"
                )
            require_sha256(
                checkpoint_digest,
                field="stage_a_cell_request.checkpoint_digest",
            )
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "candidate_id", candidate_id)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A evaluation cell request digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def is_baseline(self) -> bool:
        return self.candidate_id is None

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_identity": self.dataset_identity,
            "evaluation_identity": self.evaluation_identity,
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "fold": self.fold,
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split": self.split,
            "triplet_id": self.triplet_id,
        }


@dataclass(frozen=True, slots=True)
class StageAEvaluationCellResult:
    """Identity-bound result returned by an A6b evaluator."""

    request_digest: str
    execution_evidence_digest: str
    log_growth: float

    def __post_init__(self) -> None:
        require_sha256(self.request_digest, field="stage_a_cell_result.request_digest")
        require_sha256(
            self.execution_evidence_digest,
            field="stage_a_cell_result.execution_evidence_digest",
        )
        if isinstance(self.log_growth, bool) or not isinstance(
            self.log_growth, (int, float)
        ):
            raise ValueError("Stage A cell result log growth must be numeric")
        growth = float(self.log_growth)
        if not math.isfinite(growth):
            raise ValueError("Stage A cell result log growth must be finite")
        object.__setattr__(self, "log_growth", growth)


class StageAEvaluationCellEvaluator(Protocol):
    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult: ...


@dataclass(frozen=True, slots=True)
class StageATestFoldRange:
    fold: int
    test_range: IndexRange

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fold", _non_negative_int(self.fold, field="stage_a_test_fold.fold")
        )
        if not isinstance(self.test_range, IndexRange):
            raise ValueError("Stage A test fold range must be an IndexRange")


@dataclass(frozen=True, slots=True)
class StageATestSchedule:
    """Fold ranges bound to one Stage A plan and evaluation identity."""

    plan_digest: str
    evaluation_identity: str
    fold_ranges: tuple[StageATestFoldRange, ...]
    schema_version: str = _STAGE_A_TEST_SCHEDULE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_TEST_SCHEDULE_SCHEMA:
            raise ValueError("unsupported Stage A test schedule schema")
        require_sha256(self.plan_digest, field="stage_a_test_schedule.plan_digest")
        require_sha256(
            self.evaluation_identity,
            field="stage_a_test_schedule.evaluation_identity",
        )
        if not self.fold_ranges:
            raise ValueError("Stage A test schedule must contain fold ranges")
        ranges = tuple(sorted(self.fold_ranges, key=lambda item: item.fold))
        folds = tuple(item.fold for item in ranges)
        if len(set(folds)) != len(folds):
            raise ValueError("Stage A test schedule folds must be unique")
        object.__setattr__(self, "fold_ranges", ranges)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A test schedule digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(item.fold for item in self.fold_ranges)

    def validate_plan(self, plan: StageAZeroShotEvaluationPlan) -> None:
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A test schedule plan digest mismatch")
        if self.evaluation_identity != plan.evaluation_identity:
            raise ValueError("Stage A test schedule evaluation identity mismatch")
        if self.folds != plan.folds:
            raise ValueError("Stage A test schedule fold closure mismatch")

    def range_for(self, fold: int) -> IndexRange:
        resolved = _non_negative_int(fold, field="stage_a_test_schedule.fold")
        for item in self.fold_ranges:
            if item.fold == resolved:
                return item.test_range
        raise ValueError("Stage A test schedule fold is not declared")

    def digest_payload(self) -> dict[str, object]:
        return {
            "evaluation_identity": self.evaluation_identity,
            "fold_ranges": tuple(
                {
                    "fold": item.fold,
                    "test_range": (item.test_range.start, item.test_range.stop),
                }
                for item in self.fold_ranges
            ),
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class StageAValidationRun:
    evidence: StageAEvaluationEvidence
    selection: StageAValidationSelection
    schema_version: str = _STAGE_A_VALIDATION_RUN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_VALIDATION_RUN_SCHEMA:
            raise ValueError("unsupported Stage A validation run schema")
        if self.evidence.split != "validation":
            raise ValueError("Stage A validation run requires validation evidence")
        if self.selection.plan_digest != self.evidence.plan_digest:
            raise ValueError("Stage A validation run plan identity mismatch")
        if self.selection.validation_evidence_digest != self.evidence.digest:
            raise ValueError("Stage A validation run evidence identity mismatch")
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A validation run digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "evidence_digest": self.evidence.digest,
            "schema_version": self.schema_version,
            "selection_digest": self.selection.digest,
        }


@dataclass(frozen=True, slots=True)
class StageASealedTestRun:
    validation_run: StageAValidationRun
    access_records: tuple[SealedTestAccessRecord, ...]
    evidence: StageAEvaluationEvidence
    decision: StageASealedTestDecision
    schema_version: str = _STAGE_A_SEALED_TEST_RUN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_SEALED_TEST_RUN_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test run schema")
        if self.evidence.split != "test":
            raise ValueError("Stage A sealed-test run requires test evidence")
        selected_id = self.validation_run.selection.selected_candidate_id
        if not self.validation_run.selection.passed or selected_id is None:
            raise ValueError("Stage A sealed-test run requires passed validation")
        if self.evidence.candidate_ids != (selected_id,):
            raise ValueError("Stage A sealed-test run candidate closure mismatch")
        if self.decision.plan_digest != self.evidence.plan_digest:
            raise ValueError("Stage A sealed-test run plan identity mismatch")
        if (
            self.decision.validation_selection_digest
            != self.validation_run.selection.digest
        ):
            raise ValueError("Stage A sealed-test run selection identity mismatch")
        if self.decision.test_evidence_digest != self.evidence.digest:
            raise ValueError("Stage A sealed-test run evidence identity mismatch")
        if self.decision.selected_candidate_id != selected_id:
            raise ValueError("Stage A sealed-test run selected candidate mismatch")
        if not self.access_records:
            raise ValueError("Stage A sealed-test run requires access records")
        folds = tuple(record.fold_index for record in self.access_records)
        if len(set(folds)) != len(folds):
            raise ValueError("Stage A sealed-test access folds must be unique")
        for record in self.access_records:
            if (
                record.experiment_plan_digest != self.evidence.plan_digest
                or record.selected_configuration != selected_id
                or record.selected_policy_digest is None
            ):
                raise ValueError("Stage A sealed-test access identity mismatch")
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A sealed-test run digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "access_digests": tuple(
                record.access_digest for record in self.access_records
            ),
            "decision_digest": self.decision.digest,
            "evidence_digest": self.evidence.digest,
            "schema_version": self.schema_version,
            "validation_run_digest": self.validation_run.digest,
        }


__all__ = [
    "StageAEvaluationCellEvaluator",
    "StageAEvaluationCellRequest",
    "StageAEvaluationCellResult",
    "StageASealedTestRun",
    "StageATestFoldRange",
    "StageATestSchedule",
    "StageAValidationRun",
]
