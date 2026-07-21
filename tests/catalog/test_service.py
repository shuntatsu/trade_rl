from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from trade_rl.catalog import ArtifactRecord, ArtifactRegistration, service


class FakeCatalog:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.registrations: list[ArtifactRegistration] = []

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.registrations.append(registration)
        now = datetime(2026, 7, 21, tzinfo=UTC)
        return ArtifactRecord(registration, now, now)


def test_registration_is_noop_without_database_configuration(monkeypatch) -> None:
    called = False

    def forbidden_factory(_: str):
        nonlocal called
        called = True
        raise AssertionError("database factory must not be called")

    monkeypatch.setattr(service, "catalog_factory", forbidden_factory)
    registration = ArtifactRegistration(
        artifact_digest="a" * 64,
        artifact_kind="model",
        schema_version="model_v1",
        cache_key={"model": "a"},
        metadata={},
        location="/tmp/model",
        size_bytes=1,
    )

    assert service.register_artifact_if_configured(registration, environ={}) is None
    assert called is False


def test_registration_uses_configured_database_url(monkeypatch) -> None:
    created: list[FakeCatalog] = []
    monkeypatch.setattr(
        service,
        "catalog_factory",
        lambda url: created.append(FakeCatalog(url)) or created[-1],
    )
    registration = ArtifactRegistration(
        artifact_digest="a" * 64,
        artifact_kind="model",
        schema_version="model_v1",
        cache_key={"model": "a"},
        metadata={},
        location="/tmp/model",
        size_bytes=1,
    )

    record = service.register_artifact_if_configured(
        registration,
        environ={"TRADE_RL_DATABASE_URL": "postgresql://catalog"},
    )

    assert record is not None
    assert created[0].database_url == "postgresql://catalog"
    assert created[0].registrations == [registration]


def test_market_dataset_registration_uses_canonical_dataset_identity(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = root / "manifest.json"
    arrays = root / "arrays.npz"
    manifest.write_text("{}", encoding="utf-8")
    arrays.write_bytes(b"array")
    published = SimpleNamespace(
        root=root,
        manifest_path=manifest,
        arrays_path=arrays,
        artifact_digest="a" * 64,
        schema_version="market_dataset_artifact_v3",
    )
    dataset = SimpleNamespace(
        dataset_id="b" * 64,
        identity_payload_json='{"feature_config_digest":"c","symbols":["BTCUSDT"]}',
        symbols=("BTCUSDT",),
        n_bars=100,
        n_features=226,
        n_symbols=1,
    )

    registration = service.market_dataset_registration(published, dataset)

    assert registration.dataset_id == "b" * 64
    assert registration.cache_key == {
        "feature_config_digest": "c",
        "symbols": ("BTCUSDT",),
    }
    assert registration.size_bytes == manifest.stat().st_size + arrays.stat().st_size
    assert registration.location == str(root.resolve())
