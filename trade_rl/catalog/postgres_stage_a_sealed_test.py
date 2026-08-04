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
                    selected_policy_digest, access_digest
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING access_digest
                """,
                (
                    record.experiment_plan_digest,
                    record.dataset_id,
                    record.fold_index,
                    record.test_range.start,
                    record.test_range.stop,
                    record.selected_configuration,
                    record.selected_policy_digest,
                    record.access_digest,
                ),
            )
            generic_row = cursor.fetchone()
            if generic_row is None:
                raise ValueError("sealed outer test was already opened for this plan")
            if str(generic_row[0]) != record.access_digest:
                raise RuntimeError(
                    "Stage A generic sealed-test insert returned wrong digest"
                )

            cursor.execute(
                """
                INSERT INTO catalog_stage_a_sealed_test_cells (
                    experiment_plan_digest, batch_digest, cell_digest,
                    schema_version, evaluation_dataset_manifest_digest,
                    triplet_id, dataset_id, fold_index, test_start,
                    test_stop, generic_access_digest
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING cell_digest
                """,
                (
                    batch.experiment_plan_digest,
                    batch.batch_digest,
                    cell.digest,
                    cell.schema_version,
                    cell.evaluation_dataset_manifest_digest,
                    cell.triplet_id,
                    cell.dataset_id,
                    cell.fold_index,
                    cell.test_range.start,
                    cell.test_range.stop,
                    record.access_digest,
                ),
            )
            cell_row = cursor.fetchone()
            if cell_row is None:
                raise ValueError(
                    "Stage A sealed-test cell was already opened for this plan"
                )
            if str(cell_row[0]) != cell.digest:
                raise RuntimeError(
                    "Stage A sealed-test cell insert returned wrong digest"
                )

    @staticmethod
    def _verify_read_back(
        cursor: Any,
        batch: StageASealedTestAuthorizationBatch,
    ) -> None:
        cursor.execute(
            """
            SELECT batch_digest, schema_version,
                   evaluation_dataset_manifest_digest, evaluation_identity,
                   selected_configuration, selected_policy_digest, cell_count
            FROM catalog_stage_a_sealed_test_batches
            WHERE experiment_plan_digest = %s
            FOR UPDATE
            """,
            (batch.experiment_plan_digest,),
        )
        batch_row = cursor.fetchone()
        expected_batch = (
            batch.batch_digest,
            batch.schema_version,
            batch.evaluation_dataset_manifest_digest,
            batch.evaluation_identity,
            batch.selected_configuration,
            batch.selected_policy_digest,
            batch.cell_count,
        )
        if batch_row is None or tuple(batch_row) != expected_batch:
            raise RuntimeError("stored Stage A sealed-test batch does not match")

        cursor.execute(
            """
            SELECT cell_digest, schema_version,
                   evaluation_dataset_manifest_digest, triplet_id,
                   dataset_id, fold_index, test_start, test_stop,
                   generic_access_digest
            FROM catalog_stage_a_sealed_test_cells
            WHERE experiment_plan_digest = %s
            ORDER BY triplet_id, fold_index
            FOR UPDATE
            """,
            (batch.experiment_plan_digest,),
        )
        rows = tuple(cursor.fetchall())
        expected_cells = tuple(
            (
                cell.digest,
                cell.schema_version,
                cell.evaluation_dataset_manifest_digest,
                cell.triplet_id,
                cell.dataset_id,
                cell.fold_index,
                cell.test_range.start,
                cell.test_range.stop,
                cell.access_record.access_digest,
            )
            for cell in batch.cells
        )
        if rows != expected_cells:
            raise RuntimeError("stored Stage A sealed-test cells do not match")

    def authorize_once(
        self,
        batch: StageASealedTestAuthorizationBatch,
    ) -> StageASealedTestAuthorizationBatch:
        if not isinstance(batch, StageASealedTestAuthorizationBatch):
            raise ValueError("Stage A sealed-test authorization must be a batch")
        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    self._insert_batch(cursor, batch)
                    self._insert_cells(cursor, batch)
                    self._verify_read_back(cursor, batch)
        self._records.append(batch)
        return batch


__all__ = ["PostgresStageASealedTestLedger"]
