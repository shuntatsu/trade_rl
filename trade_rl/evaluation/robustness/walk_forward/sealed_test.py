"""One-shot authorization ledger for sealed outer-test evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.robustness.walk_forward.folds import IndexRange


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


class SealedTestLedgerProtocol(Protocol):
    @property
    def records(self) -> tuple[SealedTestAccessRecord, ...]: ...

    @property
    def consumed_access_digests(self) -> tuple[str, ...]: ...

    def authorize_once(
        self,
        *,
        experiment_plan_digest: str,
        dataset_id: str,
        fold_index: int,
        test_range: IndexRange,
        selected_configuration: str,
        selected_policy_digest: str | None,
    ) -> SealedTestAccessRecord: ...

    def consume_once(
        self, record: SealedTestAccessRecord
    ) -> SealedTestAccessRecord: ...

    def consume_all_once(
        self, records: tuple[SealedTestAccessRecord, ...]
    ) -> tuple[SealedTestAccessRecord, ...]: ...


def build_sealed_test_access_record(
    *,
    experiment_plan_digest: str,
    dataset_id: str,
    fold_index: int,
    test_range: IndexRange,
    selected_configuration: str,
    selected_policy_digest: str | None,
) -> SealedTestAccessRecord:
    payload = {
        "dataset_id": dataset_id,
        "experiment_plan_digest": experiment_plan_digest,
        "fold_index": fold_index,
        "schema_version": "sealed_test_access_v1",
        "selected_configuration": selected_configuration,
        "selected_policy_digest": selected_policy_digest,
        "test_range": (test_range.start, test_range.stop),
    }
    return SealedTestAccessRecord(
        experiment_plan_digest=experiment_plan_digest,
        dataset_id=dataset_id,
        fold_index=fold_index,
        test_range=test_range,
        selected_configuration=selected_configuration,
        selected_policy_digest=selected_policy_digest,
        access_digest=content_digest(payload),
    )


@dataclass(slots=True)
class SealedTestLedger:
    _opened: set[tuple[str, str, int]] = field(default_factory=set, init=False)
    _records: list[SealedTestAccessRecord] = field(default_factory=list, init=False)
    _consumed_access_digests: list[str] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SealedTestAccessRecord, ...]:
        return tuple(self._records)

    @property
    def consumed_access_digests(self) -> tuple[str, ...]:
        return tuple(self._consumed_access_digests)

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
        record = build_sealed_test_access_record(
            experiment_plan_digest=experiment_plan_digest,
            dataset_id=dataset_id,
            fold_index=fold_index,
            test_range=test_range,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
        )
        self._opened.add(key)
        self._records.append(record)
        return record

    @staticmethod
    def _canonical_record(record: SealedTestAccessRecord) -> SealedTestAccessRecord:
        if type(record) is not SealedTestAccessRecord:
            raise ValueError("sealed access consumption requires an access record")
        rebuilt = build_sealed_test_access_record(
            experiment_plan_digest=record.experiment_plan_digest,
            dataset_id=record.dataset_id,
            fold_index=record.fold_index,
            test_range=record.test_range,
            selected_configuration=record.selected_configuration,
            selected_policy_digest=record.selected_policy_digest,
        )
        if rebuilt != record:
            raise ValueError("sealed access record digest is inconsistent")
        return record

    def consume_all_once(
        self, records: tuple[SealedTestAccessRecord, ...]
    ) -> tuple[SealedTestAccessRecord, ...]:
        if type(records) is not tuple or not records:
            raise ValueError("sealed access consumption requires immutable records")
        canonical = tuple(self._canonical_record(record) for record in records)
        if len({record.access_digest for record in canonical}) != len(canonical):
            raise ValueError("sealed access consumption records must be unique")

        admitted: list[SealedTestAccessRecord] = []
        for record in canonical:
            match = next((value for value in self._records if value == record), None)
            if match is None:
                raise ValueError(
                    "sealed access record was not authorized by this ledger"
                )
            if record.access_digest in self._consumed_access_digests:
                raise ValueError("sealed access record was already consumed")
            admitted.append(match)

        self._consumed_access_digests.extend(
            record.access_digest for record in admitted
        )
        return tuple(admitted)

    def consume_once(self, record: SealedTestAccessRecord) -> SealedTestAccessRecord:
        return self.consume_all_once((record,))[0]


__all__ = [
    "SealedTestAccessRecord",
    "SealedTestLedger",
    "SealedTestLedgerProtocol",
    "build_sealed_test_access_record",
]
