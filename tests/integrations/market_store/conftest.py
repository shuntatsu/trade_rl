"""PostgreSQL fixtures, active only when ``TRADE_RL_TEST_POSTGRES_URL`` is set.

Each test gets its own throw-away schema, so tests are isolated and can run against
any empty or shared database (the CI job uses a ``postgres`` service container).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

POSTGRES_URL_ENV = "TRADE_RL_TEST_POSTGRES_URL"


@pytest.fixture
def postgres_url() -> str:
    url = os.environ.get(POSTGRES_URL_ENV)
    if not url:
        pytest.skip(f"{POSTGRES_URL_ENV} is not set; PostgreSQL integration skipped")
    pytest.importorskip(
        "psycopg", reason="install the `store` extra for PostgreSQL tests"
    )
    return url


@pytest.fixture
def pg_schema(postgres_url: str) -> Iterator[str]:
    import psycopg

    schema = f"mst_{uuid.uuid4().hex[:16]}"
    with psycopg.connect(postgres_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
        try:
            yield schema
        finally:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def pg_store(postgres_url: str, pg_schema: str) -> Iterator[object]:
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as store:
        store.migrate()
        yield store
