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
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
)

_STAGE_A_CELL_REQUEST_SCHEMA = "stage_a_evaluation_cell_request_v2"
_STAGE_A_TEST_SCHEDULE_SCHEMA = "stage_a_test_schedule_v2"
_STAGE_A_ACCESS_RECORD_SCHEMA = "stage_a_sealed_test_access_record_v2"
_STAGE_A_VALIDATION_RUN_SCHEMA = "stage_a_validation_run_v2"
_STAGE_A_SEALED_TEST_RUN_SCHEMA = "stage_a_sealed_test_run_v3"
_SPLITS = {"validation", "test"}


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class StageAEvaluationCellRequest:
    """One exact baseline or policy evaluation request."""

    plan_digest: str
    evaluation_dataset_manifest_digest: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    candidate_id: str | None
    checkpoint_digest: str | None
    dataset_id: str
    evaluation_range: IndexRange
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
            (
                "evaluation_dataset_manifest_digest",
                self.evaluation_dataset_manifest_digest,
            ),
            ("triplet_id", self.triplet_id),
            ("dataset_id", self.dataset_id),
            ("feature_identity", self.feature_identity),
            ("execution_identity", self.execution_identity),
            ("evaluation_identity", self.evaluation_identity),
        ):
            require_sha256(value, field=f"stage_a_cell_request.{field_name}")
        if not isinstance(self.evaluation_range, IndexRange):
            raise ValueError("Stage A evaluation cell range must be an IndexRange")
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

    def validate_manifest(
        self,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
    ) -> None:
        plan.validate_manifest(manifest)
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A evaluation cell plan digest mismatch")
        if self.evaluation_dataset_manifest_digest != manifest.digest:
            raise ValueError("Stage A evaluation cell manifest digest mismatch")
        if self.feature_identity != plan.feature_identity:
            raise ValueError("Stage A evaluation cell feature identity mismatch")
        if self.execution_identity != plan.execution_identity:
            raise ValueError("Stage A evaluation cell execution identity mismatch")
        if self.evaluation_identity != plan.evaluation_identity:
            raise ValueError("Stage A evaluation cell evaluation identity mismatch")
        if self.fold not in plan.folds:
            raise ValueError("Stage A evaluation cell fold is not declared")
        if self.seed not in plan.seeds:
            raise ValueError("Stage A evaluation cell seed is not declared")
        if self.triplet_id not in plan.triplet_ids_for(self.split):
            raise ValueError("Stage A evaluation cell triplet is not declared")
        if self.dataset_id != manifest.dataset_id_for(self.split, self.triplet_id):
            raise ValueError("Stage A evaluation cell dataset identity mismatch")
        if self.evaluation_range != manifest.range_for(self.split, self.fold):
            raise ValueError("Stage A evaluation cell evaluation range mismatch")
        if self.is_baseline:
            return
        candidate_id = self.candidate_id
        checkpoint_digest = self.checkpoint_digest
        if candidate_id is None or checkpoint_digest is None:
            raise ValueError("Stage A evaluation cell policy identity is incomplete")
        candidate = plan.candidate(candidate_id)
        if checkpoint_digest != candidate.checkpoint_digest(self.seed):
            raise ValueError("Stage A evaluation cell checkpoint identity mismatch")

    def constructor_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_id": self.dataset_id,
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "evaluation_identity": self.evaluation_identity,
            "evaluation_range": self.evaluation_range,
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "fold": self.fold,
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split": self.split,
            "triplet_id": self.triplet_id,
        }

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_id": self.dataset_id,
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "evaluation_identity": self.evaluation_identity,
            "evaluation_range": (
                self.evaluation_range.start,
                self.evaluation_range.stop,
            ),
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
    """Manifest-derived test ranges bound to one Stage A plan."""

    plan_digest: str
    evaluation_dataset_manifest_digest: str
    evaluation_identity: str
    fold_ranges: tuple[StageATestFoldRange, ...]
    schema_version: str = _STAGE_A_TEST_SCHEDULE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_TEST_SCHEDULE_SCHEMA:
            raise ValueError("unsupported Stage A test schedule schema")
        require_sha256(self.plan_digest, field="stage_a_test_schedule.plan_digest")
        require_sha256(
            self.evaluation_dataset_manifest_digest,
            field="stage_a_test_schedule.evaluation_dataset_manifest_digest",
        )
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

    @classmethod
    def from_manifest(
        cls,
        *,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
    ) -> StageATestSchedule:
        plan.validate_manifest(manifest)
        return cls(
            plan_digest=plan.digest,
            evaluation_dataset_manifest_digest=manifest.digest,
            evaluation_identity=plan.evaluation_identity,
            fold_ranges=tuple(
                StageATestFoldRange(
                    fold=fold,
                    test_range=manifest.range_for("test", fold),
                )
                for fold in manifest.folds_declared
            ),
        )

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(item.fold for item in self.fold_ranges)

    def validate_manifest(
        self,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
    ) -> None:
        plan.validate_manifest(manifest)
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A test schedule plan digest mismatch")
        if self.evaluation_dataset_manifest_digest != manifest.digest:
            raise ValueError("Stage A test schedule manifest digest mismatch")
        if self.evaluation_identity != plan.evaluation_identity:
            raise ValueError("Stage A test schedule evaluation identity mismatch")
        if self.folds != plan.folds:
            raise ValueError("Stage A test schedule fold closure mismatch")
        for item in self.fold_ranges:
            if item.test_range != manifest.range_for("test", item.fold):
                raise ValueError("Stage A test schedule range mismatch")

    def range_for(self, fold: int) -> IndexRange:
        resolved = _non_negative_int(fold, field="stage_a_test_schedule.fold")
        for item in self.fold_ranges:
            if item.fold == resolved:
                return item.test_range
        raise ValueError("Stage A test schedule fold is not declared")

    def digest_payload(self) -> dict[str, object]:
        return {
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
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
class StageASealedTestAccessRecord:
    """Stage A-specific access identity around the generic one-shot ledger record."""

    authorization_batch_digest: str
    evaluation_dataset_manifest_digest: str
    triplet_id: str
    dataset_id: str
    fold: int
    test_range: IndexRange
    ledger_record: SealedTestAccessRecord
    schema_version: str = _STAGE_A_ACCESS_RECORD_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _STAGE_A_ACCESS_RECORD_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test access schema")
        for field, value in (
            ("authorization_batch_digest", self.authorization_batch_digest),
            (
                "evaluation_dataset_manifest_digest",
                self.evaluation_dataset_manifest_digest,
            ),
            ("triplet_id", self.triplet_id),
            ("dataset_id", self.dataset_id),
        ):
            require_sha256(value, field=f"stage_a_sealed_test_access.{field}")
        fold = _non_negative_int(self.fold, field="stage_a_sealed_test_access.fold")
        if not isinstance(self.test_range, IndexRange):
            raise ValueError("Stage A sealed-test access range must be IndexRange")
        if self.ledger_record.dataset_id != self.dataset_id:
            raise ValueError("Stage A sealed-test access dataset identity mismatch")
        if self.ledger_record.fold_index != fold:
            raise ValueError("Stage A sealed-test access fold mismatch")
        if self.ledger_record.test_range != self.test_range:
            raise ValueError("Stage A sealed-test access range mismatch")
        object.__setattr__(self, "fold", fold)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A sealed-test access digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def access_digest(self) -> str:
        return self.digest

    def digest_payload(self) -> dict[str, object]:
        return {
            "authorization_batch_digest": self.authorization_batch_digest,
            "dataset_id": self.dataset_id,
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "fold": self.fold,
            "ledger_access_digest": self.ledger_record.access_digest,
            "schema_version": self.schema_version,
            "test_range": (self.test_range.start, self.test_range.stop),
            "triplet_id": self.triplet_id,
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
    access_records: tuple[StageASealedTestAccessRecord, ...]
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
        batch_digests = {
            record.authorization_batch_digest for record in self.access_records
        }
        if len(batch_digests) != 1:
            raise ValueError("Stage A sealed-test access records must share one batch")
        keys = tuple((record.triplet_id, record.fold) for record in self.access_records)
        if len(set(keys)) != len(keys):
            raise ValueError("Stage A sealed-test access cells must be unique")
        expected_keys = {
            (triplet_id, fold)
            for triplet_id in self.evidence.triplet_ids
            for fold in self.evidence.folds
        }
        if set(keys) != expected_keys:
            raise ValueError("Stage A sealed-test access cell closure mismatch")
        for record in self.access_records:
            generic = record.ledger_record
            if (
                generic.experiment_plan_digest != self.evidence.plan_digest
                or generic.selected_configuration != selected_id
                or generic.selected_policy_digest is None
            ):
                raise ValueError("Stage A sealed-test access identity mismatch")
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A sealed-test run digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def authorization_batch_digest(self) -> str:
        return self.access_records[0].authorization_batch_digest

    def digest_payload(self) -> dict[str, object]:
        return {
            "access_digests": tuple(
                record.access_digest for record in self.access_records
            ),
            "authorization_batch_digest": (self.authorization_batch_digest),
            "decision_digest": self.decision.digest,
            "evidence_digest": self.evidence.digest,
            "schema_version": self.schema_version,
            "validation_run_digest": self.validation_run.digest,
        }


__all__ = [
    "StageAEvaluationCellEvaluator",
    "StageAEvaluationCellRequest",
    "StageAEvaluationCellResult",
    "StageASealedTestAccessRecord",
    "StageASealedTestRun",
    "StageATestFoldRange",
    "StageATestSchedule",
    "StageAValidationRun",
]
