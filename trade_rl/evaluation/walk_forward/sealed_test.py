"""One-shot authorization ledger for sealed outer-test evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.walk_forward.folds import IndexRange


@dataclass(frozen=True, slots=True)
class SealedTestAccessRecord:
    experiment_plan_digest: str
    dataset_id: str
    fold_index: int
    test_range: IndexRange
    selected_configuration: str
    selected_policy_digest: str | None
    access_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.experiment_plan_digest, field="experiment_plan_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.access_digest, field="access_digest")
        require_non_empty(self.selected_configuration, field="selected_configuration")
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.selected_policy_digest is not None:
            require_sha256(self.selected_policy_digest, field="selected_policy_digest")


@dataclass(slots=True)
class SealedTestLedger:
    _opened: set[tuple[str, str, int]] = field(default_factory=set, init=False)
    _records: list[SealedTestAccessRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SealedTestAccessRecord, ...]:
        return tuple(self._records)

    def authorize_once(
        self,
        *,
        experiment_plan_digest: str,
        dataset_id: str,
        fold_index: int,
        test_range: IndexRange,
        selected_configuration: str,
        selected_policy_digest: str | None,
    ) -> SealedTestAccessRecord:
        key = (experiment_plan_digest, dataset_id, fold_index)
        if key in self._opened:
            raise ValueError("sealed outer test was already opened for this plan")
        payload = {
            "dataset_id": dataset_id,
            "experiment_plan_digest": experiment_plan_digest,
            "fold_index": fold_index,
            "schema_version": "sealed_test_access_v1",
            "selected_configuration": selected_configuration,
            "selected_policy_digest": selected_policy_digest,
            "test_range": (test_range.start, test_range.stop),
        }
        record = SealedTestAccessRecord(
            experiment_plan_digest=experiment_plan_digest,
            dataset_id=dataset_id,
            fold_index=fold_index,
            test_range=test_range,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
            access_digest=content_digest(payload),
        )
        self._opened.add(key)
        self._records.append(record)
        return record


__all__ = ["SealedTestAccessRecord", "SealedTestLedger"]
