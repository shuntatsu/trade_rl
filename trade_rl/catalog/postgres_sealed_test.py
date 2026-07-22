"""Dedicated PostgreSQL adapter for one-time sealed-test reservations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trade_rl.catalog.postgres import _default_connection_factory
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestAccessRecord


class PostgresSealedTestReservationStore:
    """Reserve sealed-test access independently of artifact catalog operations."""

    def __init__(
        self,
        database_url: str,
        *,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be non-empty")
        self._database_url = database_url
        self._connection_factory = connection_factory or _default_connection_factory

    def reserve_sealed_test_access(self, record: SealedTestAccessRecord) -> None:
        with self._connection_factory(self._database_url) as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO catalog_sealed_test_access (
                            experiment_plan_digest, dataset_id, fold_index,
                            test_start, test_stop, selected_configuration,
                            selected_policy_digest, access_digest
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (
                            experiment_plan_digest, dataset_id, fold_index
                        ) DO NOTHING
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
                    if cursor.fetchone() is None:
                        raise ValueError(
                            "sealed outer test was already opened for this plan"
                        )


__all__ = ["PostgresSealedTestReservationStore"]
