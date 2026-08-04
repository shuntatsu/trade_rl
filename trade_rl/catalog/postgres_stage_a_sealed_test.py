"""Atomic PostgreSQL ledger for complete Stage A sealed-test openings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trade_rl.catalog.postgres_connection import default_connection_factory
from trade_rl.evaluation.stage_a_sealed_test import (
    StageASealedTestAuthorizationBatch,
)


@dataclass(slots=True)
class PostgresStageASealedTestLedger:
    """Persist one complete Stage A authorization batch per plan."""

    database_url: str
    connection_factory: Callable[[str], Any] | None = None
    _records: list[StageASealedTestAuthorizationBatch] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.database_url, str) or not self.database_url.strip():
            raise ValueError("database_url must be non-empty")
        if self.connection_factory is None:
            self.connection_factory = default_connection_factory

    @property
    def records(self) -> tuple[StageASealedTestAuthorizationBatch, ...]:
        return tuple(self._records)

    def _connect(self) -> Any:
        factory = self.connection_factory
        if factory is None:
            raise RuntimeError("Stage A PostgreSQL connection factory is missing")
        return factory(self.database_url)

    @staticmethod
    def _insert_batch(cursor: Any, batch: StageASealedTestAuthorizationBatch) -> None:
        cursor.execute(
            """
            INSERT INTO catalog_stage_a_sealed_test_batches (
                experiment_plan_digest, batch_digest, schema_version,
                evaluation_dataset_manifest_digest, evaluation_identity,
                selected_configuration, selected_policy_digest, cell_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING batch_digest
            """,
            (
                batch.experiment_plan_digest,
                batch.batch_digest,
                batch.schema_version,
                batch.evaluation_dataset_manifest_digest,
                batch.evaluation_identity,
                batch.selected_configuration,
                batch.selected_policy_digest,
                batch.cell_count,
            ),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError("Stage A sealed test was already opened for this plan")
        if str(row[0]) != batch.batch_digest:
            raise RuntimeError("Stage A sealed-test batch insert returned wrong digest")

    @staticmethod
    def _insert_cells(cursor: Any, batch: StageASealedTestAuthorizationBatch) -> None:
        for cell in batch.cells:
            record = cell.access_record
            cursor.execute(
                """
                INSERT INTO catalog_sealed_test_access (
                    experiment_plan_digest, dataset_id, fold_index,
                    test_start, test_stop, selected_configuration,
                    selected_policy_digest, selection_evidence_digest,
                    access_digest, schema_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    record.experiment_plan_digest,
                    record.dataset_id,
                    record.fold_index,
                    record.test_start,
                    record.test_stop,
                    record.selected_configuration,
                    record.selected_policy_digest,
                    record.selection_evidence_digest,
                    record.digest,
                    record.schema_version,
                ),
            )
            cursor.execute(
                """
                INSERT INTO catalog_stage_a_sealed_test_cells (
                    experiment_plan_digest, cell_digest, market_regime,
                    zero_shot_regime, evaluation_seed, fold_index,
                    access_digest, schema_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    batch.experiment_plan_digest,
                    cell.cell_digest,
                    cell.key.market_regime,
                    cell.key.zero_shot_regime,
                    cell.key.evaluation_seed,
                    cell.key.fold_index,
                    record.digest,
                    cell.schema_version,
                ),
            )

    def reserve_batch(self, batch: StageASealedTestAuthorizationBatch) -> None:
        if not isinstance(batch, StageASealedTestAuthorizationBatch):
            raise TypeError("batch must be StageASealedTestAuthorizationBatch")
        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    self._insert_batch(cursor, batch)
                    self._insert_cells(cursor, batch)
        self._records.append(batch)


__all__ = ["PostgresStageASealedTestLedger"]
