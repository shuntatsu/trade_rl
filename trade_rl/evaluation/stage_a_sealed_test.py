"""Atomic Stage A sealed-test authorization contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestAccessRecord,
    build_sealed_test_access_record,
)

_CELL_SCHEMA = "stage_a_sealed_test_cell_authorization_v1"
_BATCH_SCHEMA = "stage_a_sealed_test_authorization_batch_v1"


def _fold_index(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class StageASealedTestCellSpec:
    """One manifest-derived test triplet/fold cell before authorization."""

    triplet_id: str
    dataset_id: str
    fold_index: int
    test_range: IndexRange

    def __post_init__(self) -> None:
        require_sha256(self.triplet_id, field="stage_a_sealed_cell.triplet_id")
        require_sha256(self.dataset_id, field="stage_a_sealed_cell.dataset_id")
        object.__setattr__(
            self,
            "fold_index",
            _fold_index(
                self.fold_index,
                field_name="stage_a_sealed_cell.fold_index",
            ),
        )
        if not isinstance(self.test_range, IndexRange):
            raise ValueError("Stage A sealed-test cell range must be IndexRange")


@dataclass(frozen=True, slots=True)
class StageASealedTestCellAuthorization:
    """One authorized Stage A cell bound to the generic access record."""

    evaluation_dataset_manifest_digest: str
    triplet_id: str
    dataset_id: str
    fold_index: int
    test_range: IndexRange
    access_record: SealedTestAccessRecord
    schema_version: str = _CELL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _CELL_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test cell schema")
        require_sha256(
            self.evaluation_dataset_manifest_digest,
            field="stage_a_sealed_cell.manifest_digest",
        )
        require_sha256(self.triplet_id, field="stage_a_sealed_cell.triplet_id")
        require_sha256(self.dataset_id, field="stage_a_sealed_cell.dataset_id")
        fold_index = _fold_index(
            self.fold_index,
            field_name="stage_a_sealed_cell.fold_index",
        )
        if not isinstance(self.test_range, IndexRange):
            raise ValueError("Stage A sealed-test cell range must be IndexRange")
        if self.access_record.dataset_id != self.dataset_id:
            raise ValueError("Stage A sealed-test cell dataset identity mismatch")
        if self.access_record.fold_index != fold_index:
            raise ValueError("Stage A sealed-test cell fold mismatch")
        if self.access_record.test_range != self.test_range:
            raise ValueError("Stage A sealed-test cell range mismatch")
        object.__setattr__(self, "fold_index", fold_index)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A sealed-test cell digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "access_digest": self.access_record.access_digest,
            "dataset_id": self.dataset_id,
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "fold_index": self.fold_index,
            "schema_version": self.schema_version,
            "test_range": (self.test_range.start, self.test_range.stop),
            "triplet_id": self.triplet_id,
        }


@dataclass(frozen=True, slots=True)
class StageASealedTestAuthorizationBatch:
    """Complete immutable Stage A test opening authorized as one operation."""

    experiment_plan_digest: str
    evaluation_dataset_manifest_digest: str
    evaluation_identity: str
    selected_configuration: str
    selected_policy_digest: str
    cells: tuple[StageASealedTestCellAuthorization, ...]
    schema_version: str = _BATCH_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _BATCH_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test batch schema")
        require_sha256(
            self.experiment_plan_digest,
            field="stage_a_sealed_batch.experiment_plan_digest",
        )
        require_sha256(
            self.evaluation_dataset_manifest_digest,
            field="stage_a_sealed_batch.manifest_digest",
        )
        require_sha256(
            self.evaluation_identity,
            field="stage_a_sealed_batch.evaluation_identity",
        )
        require_non_empty(
            self.selected_configuration,
            field="stage_a_sealed_batch.selected_configuration",
        )
        require_sha256(
            self.selected_policy_digest,
            field="stage_a_sealed_batch.selected_policy_digest",
        )
        if not self.cells:
            raise ValueError("Stage A sealed-test batch must contain cells")
        cells = tuple(
            sorted(self.cells, key=lambda item: (item.triplet_id, item.fold_index))
        )
        keys = tuple((cell.triplet_id, cell.fold_index) for cell in cells)
        if len(set(keys)) != len(keys):
            raise ValueError("Stage A sealed-test batch cells must be unique")
        for cell in cells:
            if (
                cell.evaluation_dataset_manifest_digest
                != self.evaluation_dataset_manifest_digest
            ):
                raise ValueError("Stage A sealed-test batch manifest mismatch")
            record = cell.access_record
            if record.experiment_plan_digest != self.experiment_plan_digest:
                raise ValueError("Stage A sealed-test batch plan mismatch")
            if record.selected_configuration != self.selected_configuration:
                raise ValueError("Stage A sealed-test batch selection mismatch")
            if record.selected_policy_digest != self.selected_policy_digest:
                raise ValueError("Stage A sealed-test batch policy mismatch")
        object.__setattr__(self, "cells", cells)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A sealed-test batch digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def batch_digest(self) -> str:
        return self.digest

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def records(self) -> tuple[SealedTestAccessRecord, ...]:
        return tuple(cell.access_record for cell in self.cells)

    def digest_payload(self) -> dict[str, object]:
        return {
            "cell_digests": tuple(cell.digest for cell in self.cells),
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "evaluation_identity": self.evaluation_identity,
            "experiment_plan_digest": self.experiment_plan_digest,
            "schema_version": self.schema_version,
            "selected_configuration": self.selected_configuration,
            "selected_policy_digest": self.selected_policy_digest,
        }


class StageASealedTestLedgerProtocol(Protocol):
    @property
    def records(self) -> tuple[StageASealedTestAuthorizationBatch, ...]: ...

    def authorize_once(
        self,
        batch: StageASealedTestAuthorizationBatch,
    ) -> StageASealedTestAuthorizationBatch: ...


def build_stage_a_sealed_test_authorization_batch(
    *,
    experiment_plan_digest: str,
    evaluation_dataset_manifest_digest: str,
    evaluation_identity: str,
    selected_configuration: str,
    selected_policy_digest: str,
    cells: tuple[StageASealedTestCellSpec, ...],
) -> StageASealedTestAuthorizationBatch:
    """Build the deterministic complete authorization batch."""

    if not cells:
        raise ValueError("Stage A sealed-test batch must contain cells")
    ordered = tuple(sorted(cells, key=lambda item: (item.triplet_id, item.fold_index)))
    keys = tuple((cell.triplet_id, cell.fold_index) for cell in ordered)
    if len(set(keys)) != len(keys):
        raise ValueError("Stage A sealed-test batch cells must be unique")
    authorized = tuple(
        StageASealedTestCellAuthorization(
            evaluation_dataset_manifest_digest=(evaluation_dataset_manifest_digest),
            triplet_id=cell.triplet_id,
            dataset_id=cell.dataset_id,
            fold_index=cell.fold_index,
            test_range=cell.test_range,
            access_record=build_sealed_test_access_record(
                experiment_plan_digest=experiment_plan_digest,
                dataset_id=cell.dataset_id,
                fold_index=cell.fold_index,
                test_range=cell.test_range,
                selected_configuration=selected_configuration,
                selected_policy_digest=selected_policy_digest,
            ),
        )
        for cell in ordered
    )
    return StageASealedTestAuthorizationBatch(
        experiment_plan_digest=experiment_plan_digest,
        evaluation_dataset_manifest_digest=evaluation_dataset_manifest_digest,
        evaluation_identity=evaluation_identity,
        selected_configuration=selected_configuration,
        selected_policy_digest=selected_policy_digest,
        cells=authorized,
    )


@dataclass(slots=True)
class StageASealedTestLedger:
    """In-memory batch ledger with the same one-shot key as PostgreSQL."""

    _opened: set[str] = field(default_factory=set, init=False)
    _records: list[StageASealedTestAuthorizationBatch] = field(
        default_factory=list,
        init=False,
    )

    @property
    def records(self) -> tuple[StageASealedTestAuthorizationBatch, ...]:
        return tuple(self._records)

    def authorize_once(
        self,
        batch: StageASealedTestAuthorizationBatch,
    ) -> StageASealedTestAuthorizationBatch:
        if not isinstance(batch, StageASealedTestAuthorizationBatch):
            raise ValueError("Stage A sealed-test authorization must be a batch")
        if batch.experiment_plan_digest in self._opened:
            raise ValueError("Stage A sealed test was already opened for this plan")
        self._opened.add(batch.experiment_plan_digest)
        self._records.append(batch)
        return batch


__all__ = [
    "StageASealedTestAuthorizationBatch",
    "StageASealedTestCellAuthorization",
    "StageASealedTestCellSpec",
    "StageASealedTestLedger",
    "StageASealedTestLedgerProtocol",
    "build_stage_a_sealed_test_authorization_batch",
]
