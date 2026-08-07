"""Persistent sealed outer-test authorization backed by PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestAccessRecord,
    build_sealed_test_access_record,
)


class SealedTestReservationStore(Protocol):
    def reserve_sealed_test_access(self, record: SealedTestAccessRecord) -> None: ...


@dataclass(slots=True)
class PostgresSealedTestLedger:
    """Reserve each plan/dataset/fold key atomically across processes."""

    store: SealedTestReservationStore
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
        record = build_sealed_test_access_record(
            experiment_plan_digest=experiment_plan_digest,
            dataset_id=dataset_id,
            fold_index=fold_index,
            test_range=test_range,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
        )
        self.store.reserve_sealed_test_access(record)
        self._records.append(record)
        return record


__all__ = ["PostgresSealedTestLedger", "SealedTestReservationStore"]
