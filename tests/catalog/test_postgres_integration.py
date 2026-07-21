from __future__ import annotations

import os

import pytest

from trade_rl.catalog import ArtifactKind, ArtifactQuery, ArtifactRegistration
from trade_rl.catalog.postgres import CatalogConflictError, PostgresArtifactCatalog
from trade_rl.catalog.sealed_test import PostgresSealedTestLedger
from trade_rl.evaluation.walk_forward.folds import IndexRange

pytestmark = pytest.mark.postgres


def _database_url() -> str:
    value = os.environ.get("TRADE_RL_TEST_DATABASE_URL")
    if not value:
        pytest.skip("TRADE_RL_TEST_DATABASE_URL is not configured")
    pytest.importorskip("psycopg")
    return value


def test_postgres_catalog_migrates_registers_queries_and_links_artifacts() -> None:
    catalog = PostgresArtifactCatalog(_database_url())
    applied = catalog.migrate()
    assert applied in {(1, 2), (2,), ()}

    parent = ArtifactRegistration(
        artifact_digest="1" * 64,
        artifact_kind=ArtifactKind.MARKET_DATASET,
        schema_version="market_dataset_artifact_v3",
        cache_key={"integration": "dataset-v1"},
        metadata={"rows": 100},
        location="/tmp/integration-dataset",
        size_bytes=1000,
        dataset_id="2" * 64,
    )
    child = ArtifactRegistration(
        artifact_digest="3" * 64,
        artifact_kind=ArtifactKind.NORMALIZER,
        schema_version="sequence_feature_normalizer_v2",
        cache_key={"dataset_id": "2" * 64, "train_start": 0, "train_end": 80},
        metadata={"channels": 226},
        location="/tmp/integration-normalizer",
        size_bytes=200,
        dataset_id="2" * 64,
    )

    parent_record = catalog.register(parent)
    assert catalog.register(parent).registration == parent
    child_record = catalog.register(child)
    assert catalog.find(parent.artifact_kind, parent.cache_key) is not None
    assert child_record in catalog.list(
        ArtifactQuery(artifact_kind=ArtifactKind.NORMALIZER, dataset_id="2" * 64)
    )
    catalog.add_dependency(
        parent_record.registration.artifact_digest,
        child_record.registration.artifact_digest,
        "source_dataset",
    )

    with pytest.raises(CatalogConflictError, match="cache key"):
        catalog.register(
            ArtifactRegistration(
                artifact_digest="4" * 64,
                artifact_kind=ArtifactKind.MARKET_DATASET,
                schema_version=parent.schema_version,
                cache_key=parent.cache_key,
                metadata=parent.metadata,
                location="/tmp/conflict",
                size_bytes=1,
                dataset_id=parent.dataset_id,
            )
        )

    health = catalog.health()
    assert health["status"] == "ok"
    assert health["migration_version"] == 2


def test_postgres_sealed_test_ledger_rejects_duplicate_across_instances() -> None:
    database_url = _database_url()
    first_catalog = PostgresArtifactCatalog(database_url)
    first_catalog.migrate()
    first = PostgresSealedTestLedger(first_catalog)
    second = PostgresSealedTestLedger(PostgresArtifactCatalog(database_url))

    first.authorize_once(
        experiment_plan_digest="a" * 64,
        dataset_id="b" * 64,
        fold_index=7,
        test_range=IndexRange(100, 120),
        selected_configuration="candidate",
        selected_policy_digest="c" * 64,
    )

    with pytest.raises(ValueError, match="already opened"):
        second.authorize_once(
            experiment_plan_digest="a" * 64,
            dataset_id="b" * 64,
            fold_index=7,
            test_range=IndexRange(100, 120),
            selected_configuration="candidate",
            selected_policy_digest="c" * 64,
        )
