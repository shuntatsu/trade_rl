from __future__ import annotations

import hashlib
import os

import pytest

from trade_rl.catalog.postgres import PostgresArtifactCatalog
from trade_rl.catalog.postgres_stage_a_sealed_test import (
    PostgresStageASealedTestLedger,
)
from trade_rl.catalog.sealed_test import PostgresSealedTestLedger
from trade_rl.evaluation.stage_a_sealed_test import (
    StageASealedTestCellSpec,
    build_stage_a_sealed_test_authorization_batch,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange

pytestmark = pytest.mark.postgres


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _database_url() -> str:
    value = os.environ.get("TRADE_RL_TEST_DATABASE_URL")
    if not value:
        pytest.skip("TRADE_RL_TEST_DATABASE_URL is not configured")
    pytest.importorskip("psycopg")
    return value


def _batch(label: str):
    return build_stage_a_sealed_test_authorization_batch(
        experiment_plan_digest=_digest(f"{label}:plan"),
        evaluation_dataset_manifest_digest=_digest(f"{label}:manifest"),
        evaluation_identity=_digest(f"{label}:evaluation"),
        selected_configuration="candidate-a",
        selected_policy_digest=_digest(f"{label}:policy"),
        cells=(
            StageASealedTestCellSpec(
                triplet_id=_digest(f"{label}:triplet-a"),
                dataset_id=_digest(f"{label}:dataset-a"),
                fold_index=0,
                test_range=IndexRange(20, 40),
            ),
            StageASealedTestCellSpec(
                triplet_id=_digest(f"{label}:triplet-b"),
                dataset_id=_digest(f"{label}:dataset-b"),
                fold_index=1,
                test_range=IndexRange(40, 60),
            ),
        ),
    )


def _counts(database_url: str, plan_digest: str) -> tuple[int, int, int]:
    import psycopg

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM catalog_stage_a_sealed_test_batches
                WHERE experiment_plan_digest = %s
                """,
                (plan_digest,),
            )
            batch_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM catalog_stage_a_sealed_test_cells
                WHERE experiment_plan_digest = %s
                """,
                (plan_digest,),
            )
            cell_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM catalog_sealed_test_access
                WHERE experiment_plan_digest = %s
                """,
                (plan_digest,),
            )
            generic_count = int(cursor.fetchone()[0])
    return batch_count, cell_count, generic_count


def test_postgres_stage_a_ledger_persists_complete_batch_once() -> None:
    database_url = _database_url()
    catalog = PostgresArtifactCatalog(database_url)
    catalog.migrate()
    batch = _batch("stage-a-ledger-once")
    first = PostgresStageASealedTestLedger(database_url)
    second = PostgresStageASealedTestLedger(database_url)

    assert first.authorize_once(batch) == batch
    assert first.records == (batch,)
    assert _counts(database_url, batch.experiment_plan_digest) == (1, 2, 2)

    with pytest.raises(ValueError, match="already opened"):
        second.authorize_once(batch)
    assert second.records == ()
    assert _counts(database_url, batch.experiment_plan_digest) == (1, 2, 2)


def test_postgres_stage_a_ledger_rolls_back_complete_batch_on_generic_conflict() -> (
    None
):
    database_url = _database_url()
    catalog = PostgresArtifactCatalog(database_url)
    catalog.migrate()
    batch = _batch("stage-a-ledger-rollback")
    conflicting = batch.cells[1].access_record
    generic = PostgresSealedTestLedger(catalog)
    generic.authorize_once(
        experiment_plan_digest=conflicting.experiment_plan_digest,
        dataset_id=conflicting.dataset_id,
        fold_index=conflicting.fold_index,
        test_range=conflicting.test_range,
        selected_configuration=conflicting.selected_configuration,
        selected_policy_digest=conflicting.selected_policy_digest,
    )

    ledger = PostgresStageASealedTestLedger(database_url)
    with pytest.raises(ValueError, match="already opened"):
        ledger.authorize_once(batch)

    assert ledger.records == ()
    assert _counts(database_url, batch.experiment_plan_digest) == (0, 0, 1)
