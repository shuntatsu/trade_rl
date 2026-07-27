from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.catalog import (
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
)
from trade_rl.cli import app
from trade_rl.data import MarketDataset, publish_market_dataset_artifact


class FakeCatalog:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.registrations: list[ArtifactRegistration] = []

    def migrate(self) -> tuple[int, ...]:
        return (1,)

    def health(self) -> dict[str, object]:
        return {
            "database": "trade_rl",
            "user": "trade_rl",
            "server_version": "PostgreSQL 16",
            "migration_version": 1,
            "status": "ok",
        }

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.registrations.append(registration)
        now = datetime(2026, 7, 21, tzinfo=UTC)
        return ArtifactRecord(registration, now, now)

    def find(self, artifact_kind: ArtifactKind, cache_key: dict[str, object]):
        return None

    def list(self, query: ArtifactQuery = ArtifactQuery()):
        return ()

    def add_dependency(self, parent_digest: str, child_digest: str, role: str) -> None:
        raise AssertionError("not used")


def _dataset() -> MarketDataset:
    n_bars = 8
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.arange(n_bars, dtype=np.float32)[:, None, None],
        global_features=np.ones((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        tick_size=np.full((n_bars, 1), 0.1),
        lot_size=np.full((n_bars, 1), 0.001),
        minimum_notional=np.full((n_bars, 1), 5.0),
    ).with_content_identity()


def test_catalog_migrate_uses_environment_without_printing_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeCatalog] = []
    monkeypatch.setenv(
        "TRADE_RL_DATABASE_URL",
        "postgresql://trade_rl:secret-password@localhost:5432/trade_rl",
    )
    from trade_rl.cli import catalog as catalog_cli

    monkeypatch.setattr(
        catalog_cli,
        "catalog_factory",
        lambda url: created.append(FakeCatalog(url)) or created[-1],
    )
    stdout = io.StringIO()

    exit_code = app.main(["catalog", "migrate"], stdout=stdout)

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload == {
        "applied_versions": [1],
        "schema": "artifact_catalog_migration_result_v1",
    }
    assert created[0].database_url.endswith("/trade_rl")
    assert "secret-password" not in stdout.getvalue()


def test_catalog_register_parses_json_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = FakeCatalog("postgresql://catalog")
    from trade_rl.cli import catalog as catalog_cli

    monkeypatch.setattr(catalog_cli, "catalog_factory", lambda _: catalog)
    stdout = io.StringIO()

    exit_code = app.main(
        [
            "catalog",
            "register",
            "--database-url",
            "postgresql://catalog",
            "--artifact-digest",
            "a" * 64,
            "--kind",
            "market_dataset",
            "--schema-version",
            "market_v1",
            "--cache-key-json",
            '{"dataset_id":"b"}',
            "--metadata-json",
            '{"rows":10}',
            "--location",
            "/tmp/data",
            "--size-bytes",
            "100",
            "--dataset-id",
            "b" * 64,
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    registration = catalog.registrations[0]
    assert registration.artifact_kind is ArtifactKind.MARKET_DATASET
    assert registration.cache_key == {"dataset_id": "b"}
    assert json.loads(stdout.getvalue())["artifact_digest"] == "a" * 64


def test_catalog_reconciles_existing_market_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = publish_market_dataset_artifact(tmp_path / "dataset", _dataset())
    catalog = FakeCatalog("postgresql://catalog")
    from trade_rl.cli import catalog as catalog_cli

    monkeypatch.setattr(catalog_cli, "catalog_factory", lambda _: catalog)
    stdout = io.StringIO()

    exit_code = app.main(
        [
            "catalog",
            "reconcile-market-dataset",
            "--database-url",
            "postgresql://catalog",
            "--artifact-root",
            str(published.root),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["artifact_digest"] == published.artifact_digest
    assert payload["dataset_id"] == _dataset().dataset_id
    assert catalog.registrations[0].location == str(published.root.resolve())


def test_catalog_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADE_RL_DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="TRADE_RL_DATABASE_URL"):
        app.main(["catalog", "health"], stdout=io.StringIO())
